# SLM Pipeline Makefile
# ----------------------
# Usage:
#   make <target>                                        # defaults: SIZE=125m, GPUS=1
#   make <target> SIZE=350m                              # different model size
#   make <target> GPUS=4                                 # multi-GPU
#   make <target> WORKERS=16                             # parallel workers for curation and artifact transfer
#   make pretrain PRETRAIN_CONFIG=pretrain/configs/gpt_125m.yaml  # explicit pretrain config override
#   make config-gen-* GPU=h200                           # override GPU auto-detection
#   make config-gen-* MODE=aggressive                    # 90% VRAM budget (or conservative=70%)
#
# New-host workflows:
#   make curate-all SIZE=125m WORKERS=62 DATA_DIR=/data/slm/data
#   make train-all SIZE=125m GPUS=4 RUN_ID=125m-YYYYMMDD-abcdef
#
# See docs/COMMANDS.md for full target documentation.

.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

SIZE    ?= 125m
DATASET_SIZE ?= $(SIZE)
DATASET_RUN_ID ?= $(RUN_ID)
GPUS    ?= 1
INSTALLER ?= pip
SFT_TOOL_DATA ?=
WORKERS ?=
FORCE   ?=
RUN_ID  ?=
ARTIFACT_STAGES ?= raw,tokenized,tokenizer,metadata
ARTIFACT_OVERWRITE ?=
PROBE_MODEL ?= $(RESULTS_DIR)/runs/$(SIZE)/pretrain/final
PROBE_TOKENIZER ?=
CORPUS_QA ?=
COMPARE_SMOL_MODEL ?= HuggingFaceTB/SmolLM2-135M
COMPARE_TOHIO_MODEL ?= $(EXPORTS_DIR)/125m/base
COMPARE_OUTPUT_DIR ?= $(RESULTS_DIR)/diagnostics/sft-comparison
COMPARE_TRAIN_EXAMPLES ?= 32
COMPARE_EVAL_EXAMPLES ?= 32
COMPARE_MAX_STEPS ?= 60
EXPORT_VARIANT ?= base

# Integrity values published in the CCNet objects' S3 metadata. These are
# content checks, not security primitives; they prevent truncated or mismatched
# model files from being accepted as a usable pair.
CCNET_EN_ARPA_MD5 := 3f5f659c62cf72d1446fdd36d6e04a57
CCNET_EN_SP_MD5   := e55b10980b6bdbd8599a3fd3a54eb9ed

REQUIRED_ENV_VARS := \
	DATA_DIR RESULTS_DIR EXPORTS_DIR HF_HOME \
	HF_DATASETS_CACHE HF_XET_HIGH_PERFORMANCE WANDB_API_KEY WANDB_PROJECT \
	HF_TOKEN

# Read path settings from .env when they are not already set in the
# environment or on the make command line. Preserve spaces in paths and strip
# optional inline comments.
_env_value = $(strip $(shell sed -n 's/^[[:space:]]*$(1)[[:space:]]*=[[:space:]]*//p' .env 2>/dev/null | sed 's/[[:space:]]*\#.*$$//' | head -1))
ifeq ($(origin DATA_DIR),undefined)
  DATA_DIR := $(or $(call _env_value,DATA_DIR),data)
endif
ifeq ($(origin RESULTS_DIR),undefined)
  RESULTS_DIR := $(or $(call _env_value,RESULTS_DIR),results)
endif
ifeq ($(origin EXPORTS_DIR),undefined)
  EXPORTS_DIR := $(or $(call _env_value,EXPORTS_DIR),$(RESULTS_DIR)/exports)
endif
ARTIFACT_BACKEND ?= $(or $(call _env_value,ARTIFACT_BACKEND),s3)
# Transfer/publication code validates only its selected destination. W&B stays
# required for the workflow; local curation does not require unused S3/Hub repos.
ifeq ($(filter $(ARTIFACT_BACKEND),s3 hf),)
  $(error ARTIFACT_BACKEND must be s3 or hf)
endif
export DATA_DIR RESULTS_DIR EXPORTS_DIR ARTIFACT_BACKEND
PYTHON     ?= .venv/bin/python
_ACCELERATE = .venv/bin/accelerate

CONFIG ?=
PRETRAIN_CONFIG ?= $(if $(CONFIG),$(CONFIG),pretrain/configs/gpt_$(SIZE).yaml)
SFT_INSTRUCT_CONFIG ?= finetune/configs/sft_instruct_$(SIZE).yaml
SFT_CODE_CONFIG ?= finetune/configs/sft_code_$(SIZE).yaml
DPO_CHAT_CONFIG ?= alignment/configs/dpo_chat_$(SIZE).yaml
DPO_CONFIG      ?= $(DPO_CHAT_CONFIG)
DPO_BASE_MODEL ?=
_DPO_BASE_FLAG = $(if $(DPO_BASE_MODEL),--base-model "$(DPO_BASE_MODEL)",)
_DATASET_FLAGS = --dataset-size "$(DATASET_SIZE)" $(if $(DATASET_RUN_ID),--dataset-run-id "$(DATASET_RUN_ID)",)
_ARTIFACT_FLAGS = --backend "$(ARTIFACT_BACKEND)"


ACCELERATE_CONFIG ?= accelerate_configs/$(if $(filter 1,$(GPUS)),single_gpu,multi_gpu).yaml
ACCELERATE = $(_ACCELERATE) launch --config_file "$(ACCELERATE_CONFIG)" --num_processes $(GPUS) --num_machines 1 --mixed_precision bf16 --dynamo_backend no

ifdef WORKERS
  WORKERS_FLAG = --workers $(WORKERS)
else
  WORKERS_FLAG =
endif

FORCE_FLAG = $(if $(filter 1 true yes,$(FORCE)),--force,)

# Reference-data controls: SIZE selects the real generated recipe, not a second
# architecture table. A stage may be run separately without retraining.
SANITY_STAGE ?= all
SANITY_CONFIG ?= $(PRETRAIN_CONFIG)
SANITY_RUN_DIR ?= $(RESULTS_DIR)/diagnostics/hf-$(SIZE)
SANITY_TOKENIZER ?= $(DATA_DIR)/runs/$(SIZE)/tokenizer
SANITY_TOKENIZED ?=
SANITY_EVAL_TOKENIZED ?=
SANITY_TARGET_TOKENS ?=
SANITY_MAX_STEPS ?=
SANITY_PROBE_EVERY_STEPS ?=
SANITY_BACKEND ?= slm
SANITY_REUSE_TOKENS ?= 0

# Fully independent model-complete pretrain control. Raw JSONL data is shared;
# tokenizer, packing, model initialization, dataset and Trainer are independent.
SANITY_INDEPENDENT_RUN_DIR ?= $(RESULTS_DIR)/diagnostics/independent-$(SIZE)
SANITY_INDEPENDENT_REFERENCE_DATA ?=
SANITY_INDEPENDENT_TOKENIZER ?= TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T
SANITY_INDEPENDENT_TOKENIZER_REVISION ?= main
SANITY_INDEPENDENT_EPOCHS ?= 2
SANITY_INDEPENDENT_LR ?=
SANITY_INDEPENDENT_BATCH_SIZE ?=
SANITY_INDEPENDENT_GRAD_ACCUM ?=
SANITY_INDEPENDENT_WARMUP_RATIO ?=
SANITY_INDEPENDENT_MAX_STEPS ?=
SANITY_INDEPENDENT_REUSE_DATA ?= 0

# GPU pipeline tests should stay mini-focused by default.
# If SIZE is explicitly supplied on the command line or environment, use it.
# Otherwise test targets validate mini artifacts even though the pipeline
# default SIZE remains 125m.
TEST_SIZE ?= $(if $(filter command line environment,$(origin SIZE)),$(SIZE),mini)

# config-gen flags
#   GPU=h200|b200|...        force a specific GPU (otherwise auto-detect via nvidia-smi)
#   MODE=conservative|balanced|aggressive   (default: balanced)
#   AGGRESSIVE=1             alias for MODE=aggressive (backwards compat)
GPU         ?=
MODE        ?=
AGGRESSIVE  ?=

# Build flag fragments used by all four config-gen-* targets.
ifeq ($(GPU),)
  _GPU_FLAG = --detect
else
  _GPU_FLAG = --gpu $(GPU)
endif

# AGGRESSIVE=1 wins over MODE if both are set; matches old behaviour.
ifdef AGGRESSIVE
  _MODE_FLAG = --mode aggressive
else ifneq ($(MODE),)
  _MODE_FLAG = --mode $(MODE)
else
  _MODE_FLAG =
endif

.PHONY: setup-curate setup-train _config-gen-launch all check-env check-curation-prereqs curate-all train-all curate curate-download curate-filter curate-dedup \
        curate-blend curate-upload validate validate-upload \
        tokenizer tokenizer-test tokenize artifacts-upload artifacts-download \
        config-gen config-gen-pretrain config-gen-sft config-gen-dpo \
        pretrain pretrain-preflight pretrain-mini pretrain-smoke pretrain-resume pretrain-resume-preflight smoke-gen prepare-sft sft sft-instruct sft-mini sft-instruct-mini sft-resume sft-instruct-resume sft-code sft-code-mini sft-code-resume \
        prepare-dpo dpo-chat dpo-chat-resume dpo-chat-mini dpo dpo-mini dpo-resume eval eval-base eval-instruct eval-chat eval-code eval-sanity eval-sanity-base eval-sanity-instruct eval-sanity-chat eval-sanity-code serve serve-local \
        export export-base export-instruct export-chat export-code \
        export-local export-base-local export-instruct-local export-chat-local export-code-local \
        check-training-env test-upgrade-gpu \
        download-kenlm-model download-fasttext-model \
        test-curator test-validate test-tokenizer test-data-pipeline \
        test-training test-sft-instruct test-sft-chat test-sft-code test-dpo-chat test-dpo test-gpu-pipeline test-model test-export test-export-acceptance test-vllm-export test-data-unit test-curation-unit test-training-args test-config-gen test-accel-gen test-comparison test-misc test-unit test-gpu-gate test-pretrain-ready test-pretrain-resume-ready test-artifacts \
        compare-sft-preflight compare-sft \
        sanity-train sanity-train-small sanity-train-tiny sanity-train-save sanity-pretrain-independent \
        clean clean-data clean-results clean-logs help

# ── New-host workflows ────────────────────────────────────────────────────────

all:
	@echo "The data and training workflows run on different host types."
	@echo "Use 'make curate-all ...' on the curation host, then"
	@echo "'make train-all ...' on the GPU host."

check-env:
	@test -f .env || (echo "Missing .env. Copy .env.sample to .env and configure paths, HF access, and required W&B credentials."; exit 1)
	@missing=""; \
	for var in $(REQUIRED_ENV_VARS); do \
		value=$$(sed -n "s/^$${var}=//p" .env | head -1 | sed 's/[[:space:]]*#.*$$//' | sed 's/^[[:space:]]*//;s/[[:space:]]*$$//'); \
		if [ -z "$$value" ] || [ "$$value" = "..." ]; then \
			missing="$$missing $$var"; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "Fill these required .env values:$$missing"; \
		exit 1; \
	fi

check-curation-prereqs: check-env
	@$(PYTHON) infra/verify_environment.py --profile curation
	@missing=""; \
	fasttext="$(DATA_DIR)/models/lid.176.ftz"; \
	kenlm_arpa="$(DATA_DIR)/models/en.arpa.bin"; \
	kenlm_sp="$(DATA_DIR)/models/en.sp.model"; \
	[ -s "$$fasttext" ] || missing="$$missing\n  FastText: $$fasttext"; \
	[ -s "$$kenlm_arpa" ] || missing="$$missing\n  KenLM:    $$kenlm_arpa"; \
	[ -s "$$kenlm_sp" ] || missing="$$missing\n  KenLM SP: $$kenlm_sp"; \
	if [ -n "$$missing" ]; then \
		printf 'ERROR: curation prerequisites are missing:%b\n\n' "$$missing"; \
		echo "Prepare the curation environment and assets with:"; \
		echo "  make setup-curate DATA_DIR=$(DATA_DIR)"; \
		exit 1; \
	fi

curate-all: check-curation-prereqs
	@case "$(SIZE)" in smoke|mini|125m|350m|1b) ;; *) echo "SIZE must be smoke, mini, 125m, 350m, or 1b"; exit 1;; esac
	@test -n "$(strip $(WORKERS))" || (echo "WORKERS is required. Example: WORKERS=62"; exit 1)
	@case "$(WORKERS)" in *[!0-9]*|'') echo "WORKERS must be a positive integer"; exit 1;; esac
	@test "$(WORKERS)" -gt 0 || (echo "WORKERS must be greater than zero"; exit 1)
	@test -n "$(strip $(DATA_DIR))" || (echo "DATA_DIR is required"; exit 1)
	@echo "==> Complete curation workflow: SIZE=$(SIZE), WORKERS=$(WORKERS), DATA_DIR=$(DATA_DIR)"
	$(PYTHON) infra/verify_environment.py --profile curation
	$(MAKE) curate SIZE="$(SIZE)" WORKERS="$(WORKERS)"
	$(MAKE) test-curator SIZE="$(SIZE)"
	$(MAKE) validate SIZE="$(SIZE)"
	$(MAKE) test-validate SIZE="$(SIZE)"
	$(MAKE) tokenizer SIZE="$(SIZE)"
	$(MAKE) tokenizer-test SIZE="$(SIZE)"
	$(MAKE) tokenize SIZE="$(SIZE)"
	$(MAKE) test-data-pipeline SIZE="$(SIZE)"
	$(MAKE) artifacts-upload \
		SIZE="$(SIZE)" \
		WORKERS="$(WORKERS)" \
		ARTIFACT_STAGES="$(ARTIFACT_STAGES)"
	@echo "==> Curation workflow complete. Record this RUN_ID:"
	@cat "$(DATA_DIR)/runs/$(SIZE)/RUN_ID"

train-all: check-env
	@case "$(SIZE)" in mini|125m|350m|1b) ;; *) echo "SIZE must be mini, 125m, 350m, or 1b"; exit 1;; esac
	@test -n "$(strip $(DATASET_RUN_ID))" || (echo "DATASET_RUN_ID (or RUN_ID) is required. Use the source dataset run ID."; exit 1)
	@case "$(GPUS)" in *[!0-9]*|'') echo "GPUS must be a positive integer"; exit 1;; esac
	@test "$(GPUS)" -gt 0 || (echo "GPUS must be greater than zero"; exit 1)
	@test -n "$(strip $(DATA_DIR))" || (echo "DATA_DIR is required"; exit 1)
	@test -n "$(strip $(RESULTS_DIR))" || (echo "RESULTS_DIR is required"; exit 1)
	@test ! -e "$(RESULTS_DIR)/runs/$(SIZE)/pretrain" || \
		(echo "train-all starts new runs only, but pretraining output already exists."; \
		 echo "Use the stage-specific resume command in docs/TRAIN.md."; exit 1)
	@echo "==> Complete training workflow: SIZE=$(SIZE), GPUS=$(GPUS), DATASET_SIZE=$(DATASET_SIZE), DATASET_RUN_ID=$(DATASET_RUN_ID)"
	$(MAKE) setup-train \
		DATA_DIR="$(DATA_DIR)" \
		SIZE="$(SIZE)" \
		DATASET_SIZE="$(DATASET_SIZE)" DATASET_RUN_ID="$(DATASET_RUN_ID)"
	$(MAKE) test-gpu-gate
	$(MAKE) config-gen SIZE="$(SIZE)" GPUS="$(GPUS)"
	$(MAKE) pretrain-preflight SIZE="$(SIZE)" GPUS="$(GPUS)"
	$(MAKE) pretrain SIZE="$(SIZE)" GPUS="$(GPUS)"
	$(MAKE) test-training SIZE="$(SIZE)"
	$(MAKE) prepare-sft SIZE="$(SIZE)"
	$(MAKE) sft-instruct SIZE="$(SIZE)" GPUS="$(GPUS)"
	$(MAKE) test-sft-instruct SIZE="$(SIZE)"
	$(MAKE) sft-code SIZE="$(SIZE)" GPUS="$(GPUS)"
	$(MAKE) test-sft-code SIZE="$(SIZE)"
	$(MAKE) prepare-dpo SIZE="$(SIZE)"
	$(MAKE) dpo-chat SIZE="$(SIZE)" GPUS="$(GPUS)"
	$(MAKE) test-dpo-chat SIZE="$(SIZE)"
	@echo "==> Training workflow complete for slm-$(SIZE)"
	@echo "Next: evaluate and export the completed variants. See docs/TRAIN.md."

# ── Stage 1: Data curation ────────────────────────────────────────────────────

curate: check-curation-prereqs
	@case "$(SIZE)" in smoke|mini|125m|350m|1b) ;; *) echo "SIZE must be smoke, mini, 125m, 350m, or 1b"; exit 1;; esac
	@echo "==> Stage 1: Curation (target=$(SIZE))"
	ulimit -n 65536 && $(PYTHON) curator/scripts/curate.py --target $(SIZE) $(WORKERS_FLAG) $(FORCE_FLAG)

curate-download: check-curation-prereqs
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage download $(FORCE_FLAG)

curate-filter: check-curation-prereqs
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage filter $(WORKERS_FLAG)

curate-dedup: check-curation-prereqs
	ulimit -n 65536 && $(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage dedup $(WORKERS_FLAG)

curate-blend: check-curation-prereqs
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage blend $(WORKERS_FLAG)

curate-upload:
	@echo "==> Uploading curated artifacts by RUN_ID (target=$(SIZE))"
	$(MAKE) artifacts-upload ARTIFACT_STAGES=curated,metadata

# ── Stage 2: Validation ───────────────────────────────────────────────────────

validate:
	@echo "==> Stage 2: Validation (train + val + test splits)"
	$(PYTHON) validation/scripts/validate.py --size $(SIZE)

validate-upload:
	@echo "==> Uploading validated artifacts by RUN_ID (target=$(SIZE))"
	$(MAKE) artifacts-upload ARTIFACT_STAGES=validated,metadata

# ── Stage 3: Tokenizer ────────────────────────────────────────────────────────

tokenizer:
	@echo "==> Stage 3: Tokenizer training"
	$(PYTHON) tokenizer/train_tokenizer.py --size $(SIZE)

tokenizer-test:
	$(PYTHON) tokenizer/test_tokenizer.py --size $(SIZE)

# ── Stage 4: Pretrain ─────────────────────────────────────────────────────────

tokenize:
	@echo "==> Stage 4a: Tokenize dataset (train + val + test splits)"
	$(PYTHON) pretrain/data/tokenize_data.py --size $(SIZE) --chunk-size 256 --verify

.PHONY: artifacts-index
artifacts-index:
	$(PYTHON) curator/scripts/upload_s3.py artifacts-index --size "$(DATASET_SIZE)" $(if $(DATASET_RUN_ID),--run-id "$(DATASET_RUN_ID)",)

artifacts-upload:
	@echo "==> Uploading $(ARTIFACT_BACKEND) artifacts: dataset=$(DATASET_SIZE), stages=$(ARTIFACT_STAGES)"
	$(PYTHON) curator/scripts/upload_s3.py artifacts-upload \
		--size "$(DATASET_SIZE)" $(if $(DATASET_RUN_ID),--run-id "$(DATASET_RUN_ID)",) \
		--stages "$(ARTIFACT_STAGES)" $(WORKERS_FLAG) $(_ARTIFACT_FLAGS) \
		$(if $(filter 1 true yes,$(ARTIFACT_OVERWRITE)),--overwrite,)

artifacts-download:
	@test -n "$(DATASET_RUN_ID)" || (echo "DATASET_RUN_ID or RUN_ID is required for artifacts-download"; exit 1)
	@echo "==> Restoring $(ARTIFACT_BACKEND) artifacts: model=$(SIZE), dataset=$(DATASET_SIZE), run=$(DATASET_RUN_ID)"
	$(PYTHON) curator/scripts/upload_s3.py artifacts-download \
		--size "$(DATASET_SIZE)" --run-id "$(DATASET_RUN_ID)" \
		--stages "$(ARTIFACT_STAGES)" $(WORKERS_FLAG) $(_ARTIFACT_FLAGS) \
		$(if $(filter 1 true yes,$(ARTIFACT_OVERWRITE)),--overwrite,)

.PHONY: eval-pretrain-final pretrain-probes
pretrain-probes:
	$(PYTHON) eval/eval.py --mode pretrain-probes --model "$(PROBE_MODEL)" \
		--size "$(SIZE)" $(_DATASET_FLAGS) $(if $(PROBE_TOKENIZER),--tokenizer-dir "$(PROBE_TOKENIZER)",)

eval-pretrain-final:
	@echo "==> Final pretraining evaluation: model=$(SIZE), dataset=$(DATASET_SIZE)"
	$(PYTHON) eval/eval.py --mode pretraining-final --model "$(PROBE_MODEL)" \
		--size "$(SIZE)" $(_DATASET_FLAGS) $(if $(CORPUS_QA),--corpus-qa "$(CORPUS_QA)",)

# ── Config generation ─────────────────────────────────────────────────────────
# Auto-generates training configs tuned for the current GPU and GPU count.
# `config-gen` (no suffix) is a convenience target that runs all three stages.
#
#   make config-gen-pretrain SIZE=125m GPUS=1                  # auto-detect GPU
#   make config-gen-sft      SIZE=350m GPUS=4 GPU=h200         # explicit GPU
#   make config-gen-dpo      SIZE=1b   GPUS=N GPU=b200 MODE=aggressive
#   make config-gen          SIZE=125m GPUS=1                  # generates all three

config-gen-pretrain: _config-gen-launch
	@echo "==> Generating pretrain config for SIZE=$(SIZE) GPUS=$(GPUS)"
	$(PYTHON) -m config_gen.config_gen \
		--stage pretrain \
		$(_GPU_FLAG) \
		--size $(SIZE) \
		--gpus $(GPUS) \
		$(_MODE_FLAG) \
		-o $(PRETRAIN_CONFIG)

config-gen-sft: _config-gen-launch
	@echo "==> Generating SFT instruct + code configs for SIZE=$(SIZE) GPUS=$(GPUS)"
	$(PYTHON) -m config_gen.config_gen \
		--stage sft \
		$(_GPU_FLAG) \
		--size $(SIZE) \
		--gpus $(GPUS) \
		$(_MODE_FLAG) \
		-o $(SFT_INSTRUCT_CONFIG) \
		--output-code $(SFT_CODE_CONFIG)

config-gen-dpo: _config-gen-launch
	@echo "==> Generating DPO config for SIZE=$(SIZE) GPUS=$(GPUS)"
	$(PYTHON) -m config_gen.config_gen \
		--stage dpo \
		$(_GPU_FLAG) \
		--size $(SIZE) \
		--gpus $(GPUS) \
		$(_MODE_FLAG) \
		-o $(DPO_CONFIG)

config-gen:
	@set -e; if [ "$(SIZE)" = "smoke" ]; then \
		$(MAKE) config-gen-pretrain SIZE="$(SIZE)" GPUS="$(GPUS)"; \
		echo "==> $(SIZE) pretraining config generated for GPUS=$(GPUS)"; \
	else \
		$(MAKE) config-gen-pretrain SIZE="$(SIZE)" GPUS="$(GPUS)"; \
		$(MAKE) config-gen-sft SIZE="$(SIZE)" GPUS="$(GPUS)"; \
		$(MAKE) config-gen-dpo SIZE="$(SIZE)" GPUS="$(GPUS)"; \
		echo "==> All training configs generated for SIZE=$(SIZE) GPUS=$(GPUS)"; \
	fi

# Internal: used by every stage-specific generator, never writes global HF config.
_config-gen-launch:
	@if [ "$(GPUS)" -gt 1 ]; then \
		$(PYTHON) -m config_gen.accel_gen --gpus "$(GPUS)" -o "$(ACCELERATE_CONFIG)"; \
	fi

# Pretrain
pretrain-preflight:
	@echo "==> Pretraining preflight ($(SIZE), $(GPUS) GPU(s), config=$(PRETRAIN_CONFIG))"
	$(PYTHON) pretrain/train.py \
		--config "$(PRETRAIN_CONFIG)" $(_DATASET_FLAGS) \
		--preflight-only \
		--expected-gpus "$(GPUS)"

pretrain-resume-preflight:
	@echo "==> Pretraining resume preflight ($(SIZE), $(GPUS) GPU(s), config=$(PRETRAIN_CONFIG))"
	$(PYTHON) pretrain/train.py \
		--config "$(PRETRAIN_CONFIG)" $(_DATASET_FLAGS) \
		--resume $(if $(RESUME_CHECKPOINT),"$(RESUME_CHECKPOINT)",) \
		--preflight-only \
		--expected-gpus "$(GPUS)"

pretrain:
	@echo "==> Stage 4b: Pretraining ($(SIZE), $(GPUS) GPU(s), config=$(PRETRAIN_CONFIG))"
	$(ACCELERATE) pretrain/train.py \
		--config "$(PRETRAIN_CONFIG)" $(_DATASET_FLAGS)

pretrain-resume:
	$(ACCELERATE) pretrain/train.py \
		--config "$(PRETRAIN_CONFIG)" $(_DATASET_FLAGS) \
		--resume $(if $(RESUME_CHECKPOINT),"$(RESUME_CHECKPOINT)",)

pretrain-mini:
	@echo "==> Generate the Mini config for the selected GPUS=$(GPUS), then use the standard pretraining path"
	$(MAKE) config-gen-pretrain SIZE=mini GPUS="$(GPUS)"
	$(MAKE) pretrain SIZE=mini GPUS="$(GPUS)"

pretrain-smoke:
	@echo "==> Smoke remains the separate bounded configuration"
	$(MAKE) pretrain SIZE=smoke GPUS="$(GPUS)"

smoke-gen:
	@echo "==> Smoke generation test for slm-$(SIZE)"
	@echo "    Purpose: detect training-objective bugs early. Output won't be"
	@echo "    fluent on a small/short run, but should be topical, not pure"
	@echo "    repetition like 'of of of of'. If outputs are total gibberish,"
	@echo "    investigate before proceeding to SFT."
	@echo ""
	@for prompt in \
		"The capital of France is" \
		"Once upon a time" \
		"Python is a programming language" \
		"The history of artificial intelligence"; do \
		echo "--- prompt: $$prompt ---"; \
		echo "$$prompt" | $(PYTHON) inference/generate.py \
			--model "$(RESULTS_DIR)/runs/$(SIZE)/pretrain/final" \
			--max-new-tokens 30 \
			--greedy; \
		echo ""; \
	done	

# ── Stage 5: SFT ──────────────────────────────────────────────────────────────

prepare-sft:
	@echo "==> Stage 5a: Prepare SFT data ($(SIZE))"
	$(PYTHON) finetune/data/prepare_sft.py --stage both --size $(SIZE) $(if $(SFT_TOOL_DATA),--tool-data "$(SFT_TOOL_DATA)",)

sft-instruct:
	@echo "==> Stage 5b: Instruct SFT ($(SIZE), $(GPUS) GPU(s), config=$(SFT_INSTRUCT_CONFIG))"
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_INSTRUCT_CONFIG)

sft: sft-instruct


sft-instruct-resume:
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_INSTRUCT_CONFIG) \
		--resume $(if $(RESUME_CHECKPOINT),"$(RESUME_CHECKPOINT)",)

sft-instruct-mini:
	@echo "==> Stage 5b: Mini instruct SFT (pipeline validation)"
	$(ACCELERATE) finetune/train_sft.py \
		--config finetune/configs/sft_instruct_mini.yaml

sft-mini: sft-instruct-mini


sft-code:
	@echo "==> Stage 5c: Code SFT ($(SIZE), $(GPUS) GPU(s), config=$(SFT_CODE_CONFIG))"
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_CODE_CONFIG)

sft-code-resume:
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_CODE_CONFIG) \
		--resume $(if $(RESUME_CHECKPOINT),"$(RESUME_CHECKPOINT)",)

sft-code-mini:
	@echo "==> Stage 5c: Mini code SFT (pipeline validation)"
	$(ACCELERATE) finetune/train_sft.py \
		--config finetune/configs/sft_code_mini.yaml

# ── Stage 5d: Raw code-completion SFT ─────────────────────────────────────────

.PHONY: prepare-code-completion sft-code-completion eval-code-completion

prepare-code-completion:
	@echo "==> Stage 5d: Prepare raw code-completion data ($(SIZE))"
	$(PYTHON) finetune/data/prepare_code_completion.py --size $(SIZE)

sft-code-completion:
	@echo "==> Stage 5e: Raw code-completion SFT ($(SIZE))"
	DATA_DIR=$(DATA_DIR) RESULTS_DIR=$(RESULTS_DIR) $(PYTHON) finetune/train_code_completion.py --config finetune/configs/code_completion_$(SIZE).yaml

eval-code-completion:
	@echo "==> Stage 7: HumanEval for raw code-completion checkpoint ($(SIZE))"
	$(PYTHON) eval/eval.py \
		--model $(RESULTS_DIR)/runs/$(SIZE)/sft_code_completion/final \
		--tasks humaneval \
		--batch-size 1 \
		--log-samples

# ── Stage 6: DPO ──────────────────────────────────────────────────────────────

prepare-dpo:
	@echo "==> Stage 6a: Prepare DPO data ($(SIZE))"
	$(PYTHON) alignment/data/prepare_dpo.py --size $(SIZE) \
		--training-config "$(DPO_CHAT_CONFIG)" $(_DPO_BASE_FLAG) $(FORCE_FLAG)

dpo-chat:
	@echo "==> Stage 6b: Chat DPO alignment ($(SIZE), $(GPUS) GPU(s), config=$(DPO_CHAT_CONFIG))"
	$(ACCELERATE) alignment/train_dpo.py \
		--config $(DPO_CHAT_CONFIG) $(_DPO_BASE_FLAG)

dpo-chat-resume:
	$(ACCELERATE) alignment/train_dpo.py \
		--config $(DPO_CHAT_CONFIG) $(_DPO_BASE_FLAG) \
		--resume $(if $(RESUME_CHECKPOINT),"$(RESUME_CHECKPOINT)",)

dpo-chat-mini:
	@echo "==> Stage 6b: Mini chat DPO (pipeline validation)"
	$(ACCELERATE) alignment/train_dpo.py \
		--config alignment/configs/dpo_chat_mini.yaml $(_DPO_BASE_FLAG)

dpo: dpo-chat

dpo-resume: dpo-chat-resume

dpo-mini: dpo-chat-mini

# ── Stage 7: Evaluation ───────────────────────────────────────────────────────

# Default eval target — alias for eval-chat (the final aligned variant).
eval: eval-chat

eval-base:
	@echo "==> Stage 7: Evaluation (base, $(SIZE))"
	$(PYTHON) eval/eval.py --model "$(RESULTS_DIR)/runs/$(SIZE)/pretrain/final"

eval-instruct:
	@echo "==> Stage 7: Evaluation (instruct, $(SIZE))"
	$(PYTHON) eval/eval.py --model "$(RESULTS_DIR)/runs/$(SIZE)/sft_instruct/final"


eval-chat:
	@echo "==> Stage 7: Evaluation (chat, $(SIZE))"
	$(PYTHON) eval/eval.py --model "$(RESULTS_DIR)/runs/$(SIZE)/dpo_chat/final"

eval-code:
	@echo "==> Stage 7: Evaluation (code, $(SIZE))"
	$(PYTHON) eval/eval.py --model "$(RESULTS_DIR)/runs/$(SIZE)/sft_code/final"


# Behavior sanity eval targets.
# These are deterministic generation checks for factuality, task format,
# code behavior, repetition, and clean stopping. They complement lm-eval
# benchmarks; they do not replace them.
eval-sanity: eval-sanity-chat

eval-sanity-base:
	@echo "==> Stage 7: Sanity evaluation (base, $(SIZE))"
	$(PYTHON) eval/sanity_eval.py \
		--model "$(RESULTS_DIR)/runs/$(SIZE)/pretrain/final" \
		--json-out "$(RESULTS_DIR)/runs/$(SIZE)/eval/sanity/base.json"

eval-sanity-instruct:
	@echo "==> Stage 7: Sanity evaluation (instruct, $(SIZE))"
	$(PYTHON) eval/sanity_eval.py \
		--model "$(RESULTS_DIR)/runs/$(SIZE)/sft_instruct/final" \
		--json-out "$(RESULTS_DIR)/runs/$(SIZE)/eval/sanity/instruct.json"


eval-sanity-chat:
	@echo "==> Stage 7: Sanity evaluation (chat, $(SIZE))"
	$(PYTHON) eval/sanity_eval.py \
		--model "$(RESULTS_DIR)/runs/$(SIZE)/dpo_chat/final" \
		--json-out "$(RESULTS_DIR)/runs/$(SIZE)/eval/sanity/chat.json"

eval-sanity-code:
	@echo "==> Stage 7: Sanity evaluation (code, $(SIZE))"
	$(PYTHON) eval/sanity_eval.py \
		--model "$(RESULTS_DIR)/runs/$(SIZE)/sft_code/final" \
		--json-out "$(RESULTS_DIR)/runs/$(SIZE)/eval/sanity/code.json"

# ── Stage 8: Export ───────────────────────────────────────────────────────────

export: export-base export-instruct export-chat export-code
	@echo "All variants exported for slm-$(SIZE)"

export-base:
	@echo "==> Stage 8: Export base model ($(SIZE))"
	EXPORTS_DIR=$(EXPORTS_DIR) $(PYTHON) export/export.py --size $(SIZE) --variant base

export-instruct:
	@echo "==> Stage 8: Export instruct model ($(SIZE))"
	EXPORTS_DIR=$(EXPORTS_DIR) $(PYTHON) export/export.py --size $(SIZE) --variant instruct

export-chat:
	@echo "==> Stage 8: Export chat model ($(SIZE))"
	EXPORTS_DIR=$(EXPORTS_DIR) $(PYTHON) export/export.py --size $(SIZE) --variant chat

export-code:
	@echo "==> Stage 8: Export code model ($(SIZE))"
	EXPORTS_DIR=$(EXPORTS_DIR) $(PYTHON) export/export.py --size $(SIZE) --variant code

export-local: export-base-local export-instruct-local export-chat-local export-code-local
	@echo "All native local variants exported for slm-$(SIZE)"

export-base-local:
	@echo "==> Stage 8: Build native local base model ($(SIZE))"
	EXPORTS_DIR=$(EXPORTS_DIR) $(PYTHON) export/export.py --size $(SIZE) --variant base --dry-run

export-instruct-local:
	@echo "==> Stage 8: Build native local instruct model ($(SIZE))"
	EXPORTS_DIR=$(EXPORTS_DIR) $(PYTHON) export/export.py --size $(SIZE) --variant instruct --dry-run

export-chat-local:
	@echo "==> Stage 8: Build native local chat model ($(SIZE))"
	EXPORTS_DIR=$(EXPORTS_DIR) $(PYTHON) export/export.py --size $(SIZE) --variant chat --dry-run

export-code-local:
	@echo "==> Stage 8: Build native local code model ($(SIZE))"
	EXPORTS_DIR=$(EXPORTS_DIR) $(PYTHON) export/export.py --size $(SIZE) --variant code --dry-run

# ── Stage 10: Serve ───────────────────────────────────────────────────────────

serve:
	@echo "==> Stage 10: Serve chat model ($(SIZE))"
	MODEL=tohio/slm-$(SIZE)-chat ./serve/serve.sh


serve-local:
	@echo "==> Stage 10: Serve native local chat export ($(SIZE))"
	MODEL=$(EXPORTS_DIR)/$(SIZE)/chat ./serve/serve.sh


# ── Setup ─────────────────────────────────────────────────────────────────────

setup-curate:
	@echo "==> Curation setup ($(INSTALLER))"
	bash infra/setup_curate.sh --installer "$(INSTALLER)" --data-dir "$(DATA_DIR)"

setup-train train-all: ARTIFACT_STAGES = tokenized,tokenizer,metadata

setup-train:
	@echo "==> Training setup ($(INSTALLER)): model=$(SIZE), dataset=$(DATASET_SIZE)"
	bash infra/setup_train.sh --installer "$(INSTALLER)" --data-dir "$(DATA_DIR)" \
		--size "$(SIZE)" --dataset-size "$(DATASET_SIZE)" --backend "$(ARTIFACT_BACKEND)" \
		--stages "$(ARTIFACT_STAGES)" \
		$(if $(DATASET_RUN_ID),--run-id "$(DATASET_RUN_ID)",--skip-data)

test-upgrade-gpu: test-gpu-gate

download-kenlm-model:
	@echo "==> Downloading matched CCNet English model pair (~4GB)..."
	mkdir -p "$(DATA_DIR)/models"
	@if [ -s "$(DATA_DIR)/models/en.arpa.bin" ] && \
		echo "$(CCNET_EN_ARPA_MD5)  $(DATA_DIR)/models/en.arpa.bin" | md5sum -c --status; then \
		echo "  Reusing $(DATA_DIR)/models/en.arpa.bin"; \
	else \
		wget -q --show-progress -c \
			https://dl.fbaipublicfiles.com/cc_net/lm/en.arpa.bin \
			-O "$(DATA_DIR)/models/en.arpa.bin.partial"; \
		echo "$(CCNET_EN_ARPA_MD5)  $(DATA_DIR)/models/en.arpa.bin.partial" | \
			md5sum -c - || { rm -f "$(DATA_DIR)/models/en.arpa.bin.partial"; exit 1; }; \
		mv "$(DATA_DIR)/models/en.arpa.bin.partial" \
			"$(DATA_DIR)/models/en.arpa.bin"; \
	fi
	@if [ -s "$(DATA_DIR)/models/en.sp.model" ] && \
		echo "$(CCNET_EN_SP_MD5)  $(DATA_DIR)/models/en.sp.model" | md5sum -c --status; then \
		echo "  Reusing $(DATA_DIR)/models/en.sp.model"; \
	else \
		wget -q --show-progress -c \
			https://dl.fbaipublicfiles.com/cc_net/lm/en.sp.model \
			-O "$(DATA_DIR)/models/en.sp.model.partial"; \
		echo "$(CCNET_EN_SP_MD5)  $(DATA_DIR)/models/en.sp.model.partial" | \
			md5sum -c - || { rm -f "$(DATA_DIR)/models/en.sp.model.partial"; exit 1; }; \
		mv "$(DATA_DIR)/models/en.sp.model.partial" \
			"$(DATA_DIR)/models/en.sp.model"; \
	fi
	@echo "  CCNet KenLM and SentencePiece models are ready"

download-fasttext-model:
	@echo "==> Preparing FastText language identification model (~1MB)..."
	mkdir -p "$(DATA_DIR)/models"
	@if [ -s "$(DATA_DIR)/models/lid.176.ftz" ] && \
		$(PYTHON) -c 'import fasttext; fasttext.load_model("$(DATA_DIR)/models/lid.176.ftz")' >/dev/null 2>&1; then \
		echo "  Reusing $(DATA_DIR)/models/lid.176.ftz"; \
	else \
		wget -q --show-progress \
			https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
			-O "$(DATA_DIR)/models/lid.176.ftz.partial" && \
		$(PYTHON) -c 'import fasttext; fasttext.load_model("$(DATA_DIR)/models/lid.176.ftz.partial")' && \
		mv "$(DATA_DIR)/models/lid.176.ftz.partial" "$(DATA_DIR)/models/lid.176.ftz" || exit 1; \
	fi

# ── Tests ─────────────────────────────────────────────────────────────────────

check-training-env:
	@test -x .venv/bin/python || { \
		echo "Training environment not found at .venv."; \
		echo "Run 'make setup-train' on the training host."; \
		exit 1; \
	}
	@.venv/bin/python infra/verify_environment.py --profile training || { \
		echo "Model, export, and training tests require the pinned training stack."; \
		echo "Run 'make setup-train' on the training host."; \
		exit 1; \
	}

test-curator:
	@echo "==> Validating curated outputs for SIZE=$(TEST_SIZE)..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/data_pipeline/test_pipeline_curator.py --size=$(TEST_SIZE) -v --tb=short

test-validate:
	@echo "==> Validating validate outputs..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/data_pipeline/test_pipeline_validate.py --size=$(TEST_SIZE) -v --tb=short

test-tokenizer:
	@echo "==> Validating tokenizer and tokenized binary outputs (SIZE=$(TEST_SIZE))..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/data_pipeline/test_pipeline_tokenizer.py --size=$(TEST_SIZE) -q --tb=short

test-data-pipeline: test-curator test-validate test-tokenizer
	@echo "==> Data pipeline tests complete"

# GPU pipeline tests respect TEST_SIZE. By default TEST_SIZE=mini so normal
# development stays mini-focused. Passing SIZE=125m / 350m / 1b on the make
# command line also sets TEST_SIZE to that size for full-artifact checks.
test-training: check-training-env
	@echo "==> Validating pretrain outputs (TEST_SIZE=$(TEST_SIZE))..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/gpu_pipeline/test_pipeline_training.py --size=$(TEST_SIZE) --require-artifacts -v --tb=short

test-sft-instruct: check-training-env
	@echo "==> Validating instruct SFT outputs (TEST_SIZE=$(TEST_SIZE))..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/gpu_pipeline/test_pipeline_sft.py::TestChatSFTModel tests/gpu_pipeline/test_pipeline_sft.py::TestSFTData --size=$(TEST_SIZE) --require-artifacts -v --tb=short

test-sft-chat: test-sft-instruct


test-sft-code: check-training-env
	@echo "==> Validating code SFT outputs (TEST_SIZE=$(TEST_SIZE))..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/gpu_pipeline/test_pipeline_sft.py::TestCodeSFTModel --size=$(TEST_SIZE) --require-artifacts -v --tb=short

test-dpo-chat: check-training-env
	@echo "==> Validating chat DPO outputs (TEST_SIZE=$(TEST_SIZE))..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/gpu_pipeline/test_pipeline_dpo.py --size=$(TEST_SIZE) --require-artifacts -v --tb=short

test-dpo: test-dpo-chat


test-gpu-pipeline: test-training test-sft-instruct test-sft-code test-dpo-chat
	@echo "==> GPU pipeline tests complete"

test-model: check-training-env
	@echo "==> Running model unit tests..."
	.venv/bin/pytest tests/model/ -v --tb=short

test-export: check-training-env
	@echo "==> Running native export unit tests..."
	.venv/bin/pytest tests/test_export.py -v --tb=short

test-export-acceptance: check-training-env
	@case "$(EXPORT_VARIANT)" in base|instruct|chat|code) ;; \
		*) echo "EXPORT_VARIANT must be base, instruct, chat, or code"; exit 1;; esac
	@echo "==> Native export/load acceptance ($(SIZE), $(EXPORT_VARIANT))..."
	EXPORTS_DIR=$(EXPORTS_DIR) $(PYTHON) export/export.py \
		--size "$(SIZE)" \
		--variant "$(EXPORT_VARIANT)" \
		--dry-run

test-vllm-export: test-export-acceptance
	@echo "==> vLLM offline generation smoke ($(SIZE), $(EXPORT_VARIANT))..."
	$(PYTHON) scripts/vllm_smoke.py \
		--model "$(EXPORTS_DIR)/$(SIZE)/$(EXPORT_VARIANT)"

test-data-unit:
	@echo "==> Running shared data/tokenization/post-training contract tests..."
	.venv/bin/pytest tests/test_data_config.py tests/test_curator_state.py \
		tests/test_realized_mixture.py tests/test_tokenize_data.py \
		tests/test_sft_data_contract.py tests/test_dpo_data_contract.py -v --tb=short

test-curation-unit: test-data-unit
	@echo "==> Running curation-only unit tests (no corpus/model assets required)..."
	$(PYTHON) infra/verify_environment.py --profile curation
	.venv/bin/pytest tests/test_benchmark_contamination.py tests/test_common_crawl_source.py \
		tests/test_curation_audit_stats.py tests/test_dedup_partitioning.py \
		tests/test_exact_split_overlap.py tests/test_long_document_segmentation.py \
		tests/test_near_overlap.py tests/test_quality_filter.py \
		tests/test_sensitive_content.py tests/test_validation_kenlm.py -v --tb=short

test-training-args: check-training-env
	@echo "==> Running Transformers/TRL compatibility tests..."
	.venv/bin/pytest tests/test_environment_contract.py tests/test_training_args.py tests/test_trl_smoke.py -v --tb=short

test-config-gen:
	@echo "==> Running config_gen unit tests..."
	.venv/bin/pytest tests/test_config_gen.py -v --tb=short

test-accel-gen:
	@echo "==> Running accel_gen unit tests..."
	.venv/bin/pytest tests/test_accel_gen.py -v --tb=short

test-comparison: check-training-env
	@echo "==> Running controlled-comparison harness unit tests..."
	.venv/bin/pytest tests/test_sft_comparison.py -v --tb=short

test-misc: check-training-env
	@echo "==> Running cross-cutting contract tests..."
	.venv/bin/pytest tests/test_misc_contract.py -v --tb=short

test-unit: test-model test-export test-data-unit test-training-args test-config-gen test-accel-gen test-comparison test-misc
	@echo "==> Training/common unit tests complete (curation host: make test-curation-unit)"

test-gpu-gate: check-training-env
	@echo "==> Validating the installed GPU stack without datasets or checkpoints..."
	.venv/bin/python infra/verify_environment.py --require-cuda
	.venv/bin/python infra/gpu_smoke.py

test-pretrain-ready: check-env
	@echo "==> Bounded readiness gate for a new pretraining run..."
	$(MAKE) test-model
	$(MAKE) test-data-unit
	$(MAKE) test-training-args
	$(MAKE) test-config-gen
	$(MAKE) test-accel-gen
	$(MAKE) test-gpu-gate
	$(MAKE) pretrain-preflight SIZE="$(SIZE)" GPUS="$(GPUS)"

test-pretrain-resume-ready: check-env
	@echo "==> Bounded readiness gate for a resumed pretraining run..."
	$(MAKE) test-model
	$(MAKE) test-data-unit
	$(MAKE) test-training-args
	$(MAKE) test-config-gen
	$(MAKE) test-accel-gen
	$(MAKE) test-gpu-gate
	$(MAKE) pretrain-resume-preflight SIZE="$(SIZE)" GPUS="$(GPUS)"

test-artifacts: test-data-pipeline test-gpu-pipeline
	@echo "==> Existing pipeline artifact validation complete"

# Diagnostic comparison. Preflight loads checkpoints only and is the required
# cheap gate before the opt-in training run.
compare-sft-preflight:
	$(PYTHON) scripts/sft_model_comparison.py \
		--smol-model "$(COMPARE_SMOL_MODEL)" \
		--tohio-model "$(COMPARE_TOHIO_MODEL)" \
		--output-dir "$(COMPARE_OUTPUT_DIR)" \
		--preflight-only

compare-sft:
	$(PYTHON) scripts/sft_model_comparison.py \
		--smol-model "$(COMPARE_SMOL_MODEL)" \
		--tohio-model "$(COMPARE_TOHIO_MODEL)" \
		--train-examples $(COMPARE_TRAIN_EXAMPLES) \
		--eval-examples $(COMPARE_EVAL_EXAMPLES) \
		--max-steps $(COMPARE_MAX_STEPS) \
		--output-dir "$(COMPARE_OUTPUT_DIR)"

# ── Sanity check ──────────────────────────────────────────────────────────────

# Small/tiny are token-budget presets only; all use the selected SIZE/config.
sanity-train-small: SANITY_TARGET_TOKENS = 500000000
sanity-train-tiny: SANITY_TARGET_TOKENS = 50000000

sanity-pretrain-independent: check-training-env
	@test -n "$(SANITY_INDEPENDENT_REFERENCE_DATA)" || { echo "ERROR: SANITY_INDEPENDENT_REFERENCE_DATA must point to exact raw train/val/test JSONL data"; exit 2; }
	@echo "==> Independent pretrain control: size=$(SIZE), raw_data=$(SANITY_INDEPENDENT_REFERENCE_DATA), tokenizer=$(SANITY_INDEPENDENT_TOKENIZER)"
	$(PYTHON) scripts/pretrain_reference_independent.py --stage all \
		--config "$(SANITY_CONFIG)" \
		--reference-data-dir "$(SANITY_INDEPENDENT_REFERENCE_DATA)" \
		--run-dir "$(SANITY_INDEPENDENT_RUN_DIR)" \
		--tokenizer "$(SANITY_INDEPENDENT_TOKENIZER)" \
		--tokenizer-revision "$(SANITY_INDEPENDENT_TOKENIZER_REVISION)" \
		--epochs "$(SANITY_INDEPENDENT_EPOCHS)" \
		$(if $(SANITY_INDEPENDENT_LR),--learning-rate "$(SANITY_INDEPENDENT_LR)",) \
		$(if $(SANITY_INDEPENDENT_BATCH_SIZE),--batch-size "$(SANITY_INDEPENDENT_BATCH_SIZE)",) \
		$(if $(SANITY_INDEPENDENT_GRAD_ACCUM),--gradient-accumulation "$(SANITY_INDEPENDENT_GRAD_ACCUM)",) \
		$(if $(SANITY_INDEPENDENT_WARMUP_RATIO),--warmup-ratio "$(SANITY_INDEPENDENT_WARMUP_RATIO)",) \
		$(if $(SANITY_INDEPENDENT_MAX_STEPS),--max-steps "$(SANITY_INDEPENDENT_MAX_STEPS)",) \
		$(if $(filter 1 true yes,$(SANITY_INDEPENDENT_REUSE_DATA)),--reuse-data,)

sanity-train sanity-train-small sanity-train-tiny sanity-train-save: check-training-env
	@echo "==> HF control: stage=$(SANITY_STAGE), size=$(SIZE), backend=$(SANITY_BACKEND), outputs=$(SANITY_RUN_DIR)"
	$(PYTHON) scripts/sanity_train.py --stage "$(SANITY_STAGE)" \
		--config "$(SANITY_CONFIG)" --size "$(SIZE)" --run-dir "$(SANITY_RUN_DIR)" \
		--tokenizer-dir "$(SANITY_TOKENIZER)" --backend "$(SANITY_BACKEND)" \
		$(if $(SANITY_TOKENIZED),--tokenized-dir "$(SANITY_TOKENIZED)",) \
		$(if $(SANITY_EVAL_TOKENIZED),--eval-tokenized-dir "$(SANITY_EVAL_TOKENIZED)",) \
		$(if $(SANITY_TARGET_TOKENS),--target-tokens "$(SANITY_TARGET_TOKENS)",) \
		$(if $(SANITY_MAX_STEPS),--max-steps "$(SANITY_MAX_STEPS)",) \
		$(if $(SANITY_PROBE_EVERY_STEPS),--probe-every-steps "$(SANITY_PROBE_EVERY_STEPS)",) \
		$(if $(filter 1 true yes,$(SANITY_REUSE_TOKENS)),--reuse-tokens,)

# ── Clean ─────────────────────────────────────────────────────────────────────

clean-data:
	rm -rf "$(DATA_DIR)/runs/$(SIZE)" "$(DATA_DIR)/dedup_scratch"

clean-results:
	@test -n "$(strip $(RESULTS_DIR))" || (echo "RESULTS_DIR must not be empty"; exit 1)
	@if [ -e "$(RESULTS_DIR)" ]; then \
		resolved="$$(cd "$(RESULTS_DIR)" && pwd -P)"; \
		test "$$resolved" != "/" || (echo "Refusing to remove /"; exit 1); \
	fi
	rm -rf -- "$(RESULTS_DIR)"

clean-logs:
	rm -rf logs/

clean: clean-logs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "SLM Pipeline"
	@echo "============"
	@echo ""
	@echo "Usage: make <target> [SIZE=smoke|mini|125m|350m|1b] [GPUS=N] [WORKERS=N] [DATA_DIR=path]"
	@echo "       make config-gen-* [GPU=h200|b200|...] [MODE=conservative|balanced|aggressive]"
	@echo ""
	@echo "For full target documentation see: docs/COMMANDS.md"
	@echo ""
	@echo "New-host workflows:"
	@echo "  curate-all               Curate, validate, tokenize, test, and upload after bootstrap"
	@echo "  train-all                Setup, restore, train all model branches, and test"
	@echo ""
	@echo "Config generation (run before pretrain/sft-instruct/dpo-chat/sft-code):"
	@echo "  config-gen-pretrain  Auto-generate pretrain/configs/gpt_$(SIZE).yaml"
	@echo "  config-gen-sft       Auto-generate sft_instruct_$(SIZE).yaml + sft_code_$(SIZE).yaml"
	@echo "  config-gen-dpo       Auto-generate alignment/configs/dpo_chat_$(SIZE).yaml"
	@echo "  config-gen           Generates all stages (Smoke: pretrain only)"
	@echo ""
	@echo "One-time setup (INSTALLER=pip|uv|conda; default pip):"
	@echo "  setup-curate             Curation environment; accepts DATA_DIR=..."
	@echo "  setup-train              GPU training environment; optional dataset RUN_ID restore"
	@echo "  download-fasttext-model  Internal FastText asset repair (called by setup-curate)"
	@echo "  download-kenlm-model     Internal KenLM asset repair (called by setup-curate)"
	@echo "  test-upgrade-gpu         One-shot CUDA/compile/cache acceptance test"
	@echo ""
	@echo "Tests (CPU — data pipeline):"
	@echo "  test-curator             Validate curated outputs for SIZE=<size>"
	@echo "  test-validate            Validate validate outputs"
	@echo "  test-tokenizer           Validate tokenizer outputs and tokenized binaries"
	@echo "  test-data-pipeline       Run all data pipeline tests"
	@echo ""
	@echo "Tests (GPU — training pipeline, use SIZE=<size>, default mini):"
	@echo "  test-training            Validate pretrain outputs"
	@echo "  test-sft-instruct        Validate instruct SFT outputs"
	@echo "  test-sft-code            Validate code SFT outputs"
	@echo "  test-dpo-chat            Validate chat DPO outputs"
	@echo "  test-gpu-pipeline        Run all GPU pipeline tests"
	@echo ""
	@echo "Tests (unit — pinned training stack; no pipeline outputs needed):"
	@echo "  test-model               Model architecture unit tests"
	@echo "  test-export              Native Transformers export contract tests"
	@echo "  test-export-acceptance   Build and clean-load one real native export"
	@echo "  test-vllm-export         Load that export in vLLM and generate one response"
	@echo "  test-data-unit           Shared data, tokenization, and SFT/DPO contract tests"
	@echo "  test-training-args       Transformers/TRL argument compatibility tests"
	@echo "  test-config-gen          Config generator unit tests"
	@echo "  test-accel-gen           Accelerate config generator unit tests"
	@echo "  test-comparison          Controlled SFT comparison unit tests"
	@echo "  test-curation-unit       Curation/common unit tests in the curation environment"
	@echo "  test-unit                Training/common unit tests in the training environment"
	@echo "  test-gpu-gate            Bounded CUDA/compile/cache test; no datasets"
	@echo "  test-pretrain-ready      Bounded new-run contracts, CUDA gate, and preflight"
	@echo "  test-pretrain-resume-ready Bounded resume contracts and provenance preflight"
	@echo "  test-artifacts           Strictly validate existing mini/full artifacts"
	@echo ""
	@echo "Controlled model comparison:"
	@echo "  compare-sft-preflight    Check vocab, parameters, prompt sensitivity, cache"
	@echo "  compare-sft              Run the opt-in controlled SFT response comparison"
	@echo ""
	@echo "Sanity check (model + training code only):"
	@echo "  sanity-train             HF control using SIZE/config; SANITY_STAGE=prepare|check|train|all"
	@echo "  sanity-train-small       same selected SIZE/config, 500M unique train-token target"
	@echo "  sanity-train-tiny        same selected SIZE/config, 50M unique train-token target"
	@echo "  sanity-train-save        compatibility alias; all control training preserves checkpoints"
	@echo "  sanity-pretrain-independent  model-complete native HF control; shares raw data + experiment values only"
	@echo ""
	@echo "Pipeline:"
	@echo "  curate             Stage 1  — curate SIZE=smoke|mini|125m|350m|1b"
	@echo "  validate           Stage 2  — source-aware validation + KenLM audit"
	@echo "  validate-upload    Upload validated artifacts through RUN_ID storage"
	@echo "  tokenizer          Stage 3  — train BPE tokenizer"
	@echo "  tokenize           Stage 4a — tokenize train + val + test to binaries"
	@echo "  eval-pretrain-final Final-only loss/PPL and diagnostic categories"
	@echo "  pretrain-probes    Fixed non-gating prompts on a saved checkpoint"
	@echo "  artifacts-upload   Upload selected S3/HF artifacts using a dataset RUN_ID"
	@echo "  artifacts-download Restore matched S3/HF dataset artifacts using RUN_ID=<run_id>"
	@echo "  pretrain-preflight Validate a new run without allocating model weights"
	@echo "  pretrain           Stage 4b — pretrain from scratch"
	@echo "  pretrain-resume-preflight Validate checkpoint and provenance before resume"
	@echo "  pretrain-resume    Stage 4b — resume complete state (optional RESUME_CHECKPOINT)"
	@echo "  pretrain-smoke     Stage 4b — bounded smoke pretraining run"
	@echo "  pretrain-mini      Stage 4b — 69.9M functional mini pilot"
	@echo "  smoke-gen          Stage 4b — generate from \$$(RESULTS_DIR)/runs/\$$(SIZE)/pretrain/final to spot-check"
	@echo "  prepare-sft        Stage 5a — download SFT datasets"
	@echo "  sft-instruct       Stage 5b — instruct supervised fine-tuning"
	@echo "  sft-instruct-mini  Stage 5b — mini instruct SFT"
	@echo "  sft-code           Stage 5c — code supervised fine-tuning"
	@echo "  sft-code-mini      Stage 5c — mini code SFT"
	@echo "  prepare-code-completion Stage 5d — prepare raw code-completion data"
	@echo "  sft-code-completion     Stage 5e — raw code-completion supervised fine-tuning"
	@echo "  eval-code-completion    Stage 7  — HumanEval for raw code-completion checkpoint"
	@echo "  prepare-dpo        Stage 6a — download DPO datasets"
	@echo "  dpo-chat           Stage 6b — chat DPO alignment"
	@echo "  dpo-chat-mini      Stage 6b — mini chat DPO"
	@echo "  eval-base              Stage 7  — benchmark eval for base variant"
	@echo "  eval-instruct          Stage 7  — benchmark eval for instruct variant"
	@echo "  eval-chat              Stage 7  — benchmark eval for chat variant"
	@echo "  eval                   Stage 7  — alias for eval-chat"
	@echo "  eval-sanity-base       Stage 7  — behavior sanity eval for base variant"
	@echo "  eval-sanity-instruct   Stage 7  — behavior sanity eval for instruct variant"
	@echo "  eval-sanity-chat       Stage 7  — behavior sanity eval for chat variant"
	@echo "  eval-sanity            Stage 7  — alias for eval-sanity-chat"
	@echo "  export             Stage 8  — build, validate, and push all native Hub variants"
	@echo "  export-local       Stage 8  — build and validate all native variants without pushing"
	@echo "  serve              Stage 10 — launch vLLM server"
	@echo ""

.PHONY: restore-size-tokenizer
restore-size-tokenizer:
	@echo "==> Restoring size-specific tokenizer ($(SIZE)) into active tokenizer path..."
	@test -d "$(DATA_DIR)/runs/$(SIZE)/tokenizer" || (echo "Missing $(DATA_DIR)/runs/$(SIZE)/tokenizer — run setup-train or artifacts-download first" && exit 1)
	mkdir -p "$(DATA_DIR)/tokenizer"
	cp -a "$(DATA_DIR)/runs/$(SIZE)/tokenizer/." "$(DATA_DIR)/tokenizer/"
	@echo "  Active tokenizer restored from $(DATA_DIR)/runs/$(SIZE)/tokenizer"
