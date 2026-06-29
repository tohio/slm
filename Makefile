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
# Full pipeline:
#   make all SIZE=125m GPUS=4
#
# See docs/COMMANDS.md for full target documentation.

SIZE    ?= 125m
GPUS    ?= 1
WORKERS ?=
RUN_ID  ?=
ARTIFACT_STAGES ?= raw,curated,validated,tokenized,tokenizer,metadata

# DATA_DIR — read from .env if not set in environment.
DATA_DIR := $(or $(shell grep -v '^\#' .env 2>/dev/null | grep '^DATA_DIR=' | head -1 | cut -d= -f2 | tr -d ' '),data)
RESULTS_DIR := $(or $(shell grep -v '^\#' .env 2>/dev/null | grep '^RESULTS_DIR=' | head -1 | cut -d= -f2 | tr -d ' '),results)
PYTHON     ?= .venv/bin/python
_ACCELERATE = .venv/bin/accelerate

CONFIG ?=
PRETRAIN_CONFIG ?= $(if $(CONFIG),$(CONFIG),pretrain/configs/gpt_$(SIZE).yaml)
SFT_INSTRUCT_CONFIG ?= finetune/configs/sft_instruct_$(SIZE).yaml
SFT_CODE_CONFIG ?= finetune/configs/sft_code_$(SIZE).yaml
DPO_CHAT_CONFIG ?= alignment/configs/dpo_chat_$(SIZE).yaml
DPO_CONFIG      ?= $(DPO_CHAT_CONFIG)

ACCELERATE = $(_ACCELERATE) launch --num_processes $(GPUS) --num_machines 1 --mixed_precision bf16 --dynamo_backend no

ifdef WORKERS
  WORKERS_FLAG = --workers $(WORKERS)
else
  WORKERS_FLAG =
endif

SANITY_SIZE ?= 125m

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

.PHONY: all curate curate-mini curate-download curate-filter curate-dedup \
        curate-blend curate-upload validate validate-upload validate-datatrove \
        tokenizer tokenizer-test tokenize artifacts-upload artifacts-download \
        config-gen config-gen-pretrain config-gen-sft config-gen-dpo \
        accel-gen-ddp accel-gen-fsdp \
        pretrain pretrain-mini pretrain-smoke pretrain-resume reinit-embeds smoke-gen prepare-sft sft sft-instruct sft-mini sft-instruct-mini sft-resume sft-instruct-resume sft-code sft-code-mini sft-code-resume \
        prepare-dpo dpo-chat dpo-chat-resume dpo-chat-mini dpo dpo-mini dpo-resume eval eval-base eval-instruct eval-chat eval-code eval-sanity eval-sanity-base eval-sanity-instruct eval-sanity-chat eval-sanity-code eval-mini serve serve-local \
        export export-base export-instruct export-chat export-code \
        setup setup-data-dir setup-gpu install install-gpu install-uv install-conda install-kenlm install-orjson \
        download-kenlm-model download-fasttext-model accelerate-config accelerate-config-single accelerate-config-multi \
        s3-upload s3-download s3-list \
        test-curator test-validate test-tokenizer test-data-pipeline \
        test-training test-sft-instruct test-sft-chat test-sft-code test-dpo-chat test-dpo test-gpu-pipeline test-model test-config-gen test-accel-gen test-unit \
        sanity-train sanity-train-small sanity-train-tiny sanity-train-save \
        clean clean-data clean-results clean-logs help

# ── Full pipeline ──────────────────────────────────────────────────────────────
# Note: assumes configs exist at $(PRETRAIN_CONFIG), $(SFT_INSTRUCT_CONFIG), etc.
# Run `make config-gen` first to auto-generate them tuned for the current GPU.

all: curate validate tokenizer tokenize pretrain prepare-sft sft-instruct prepare-dpo dpo-chat sft-code
	@echo "Canonical pipeline complete for slm-$(SIZE) on $(GPUS) GPU(s)"

# ── Stage 1: Data curation ────────────────────────────────────────────────────

curate:
	@echo "==> Stage 1: Curation (target=$(SIZE))"
	ulimit -n 65536 && $(PYTHON) curator/scripts/curate.py --target $(SIZE) $(WORKERS_FLAG)

curate-mini:
	@echo "==> Stage 1: Mini curation run (pipeline validation)"
	ulimit -n 65536 && $(PYTHON) curator/scripts/curate.py --target mini --mini $(WORKERS_FLAG)

curate-download:
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage download

curate-filter:
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage filter $(WORKERS_FLAG)

curate-dedup:
	ulimit -n 65536 && $(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage dedup $(WORKERS_FLAG)

curate-blend:
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage blend $(WORKERS_FLAG)

curate-upload:
	@echo "==> Stage 1: Upload curated data to S3 (target=$(SIZE))"
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage upload

# ── Stage 2: Validation ───────────────────────────────────────────────────────

validate:
	@echo "==> Stage 2: Validation (train + val splits)"
	$(PYTHON) validation/scripts/validate.py

validate-upload:
	@echo "==> Stage 2: Upload validated data to S3 (target=$(SIZE))"
	$(PYTHON) validation/scripts/upload_validated.py --target $(SIZE)

validate-datatrove:
	$(PYTHON) validation/scripts/validate.py --use-datatrove

# ── Stage 3: Tokenizer ────────────────────────────────────────────────────────

tokenizer:
	@echo "==> Stage 3: Tokenizer training"
	$(PYTHON) tokenizer/train_tokenizer.py

tokenizer-test:
	$(PYTHON) tokenizer/test_tokenizer.py

# ── Stage 4: Pretrain ─────────────────────────────────────────────────────────

tokenize:
	@echo "==> Stage 4a: Tokenize dataset (train + val splits)"
	$(PYTHON) pretrain/data/tokenize_data.py --chunk-size 256 --verify

artifacts-upload:
	@echo "==> Uploading artifacts to S3 (target=$(SIZE), run_id=$(if $(RUN_ID),$(RUN_ID),auto), stages=$(ARTIFACT_STAGES))"
	$(PYTHON) curator/scripts/upload_s3.py artifacts-upload \
		--size $(SIZE) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) \
		--stages "$(ARTIFACT_STAGES)" $(WORKERS_FLAG) \
		--overwrite

artifacts-download:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required for artifacts-download"; exit 1)
	@echo "==> Downloading artifacts from S3 (target=$(SIZE), run_id=$(RUN_ID), stages=$(ARTIFACT_STAGES))"
	$(PYTHON) curator/scripts/upload_s3.py artifacts-download \
		--size $(SIZE) \
		--run-id $(RUN_ID) \
		--stages "$(ARTIFACT_STAGES)" $(WORKERS_FLAG)

# ── Config generation ─────────────────────────────────────────────────────────
# Auto-generates training configs tuned for the current GPU and GPU count.
# `config-gen` (no suffix) is a convenience target that runs all three stages.
#
#   make config-gen-pretrain SIZE=125m GPUS=1                  # auto-detect GPU
#   make config-gen-sft      SIZE=350m GPUS=4 GPU=h200         # explicit GPU
#   make config-gen-dpo      SIZE=1b   GPUS=8 GPU=b200 MODE=aggressive
#   make config-gen          SIZE=125m GPUS=1                  # generates all three

config-gen-pretrain:
	@echo "==> Generating pretrain config for SIZE=$(SIZE) GPUS=$(GPUS)"
	$(PYTHON) -m config_gen.config_gen \
		--stage pretrain \
		$(_GPU_FLAG) \
		--size $(SIZE) \
		--gpus $(GPUS) \
		$(_MODE_FLAG) \
		-o $(PRETRAIN_CONFIG)

config-gen-sft:
	@echo "==> Generating SFT instruct + code configs for SIZE=$(SIZE) GPUS=$(GPUS)"
	$(PYTHON) -m config_gen.config_gen \
		--stage sft \
		$(_GPU_FLAG) \
		--size $(SIZE) \
		--gpus $(GPUS) \
		$(_MODE_FLAG) \
		-o $(SFT_INSTRUCT_CONFIG) \
		--output-code $(SFT_CODE_CONFIG)

config-gen-dpo:
	@echo "==> Generating DPO config for SIZE=$(SIZE) GPUS=$(GPUS)"
	$(PYTHON) -m config_gen.config_gen \
		--stage dpo \
		$(_GPU_FLAG) \
		--size $(SIZE) \
		--gpus $(GPUS) \
		$(_MODE_FLAG) \
		-o $(DPO_CONFIG)

config-gen: config-gen-pretrain config-gen-sft config-gen-dpo
	@echo "==> All training configs generated for SIZE=$(SIZE) GPUS=$(GPUS)"

# ── Accelerate launch config generation ───────────────────────────────────────
# Generates accelerate_configs/{multi_gpu,fsdp}.yaml from a small generator.
# Replaces the old sed-based accelerate-config-multi flow.
#
#   make accel-gen-ddp  GPUS=8                  # plain DDP
#   make accel-gen-fsdp GPUS=8                  # FullyShardedDataParallel for 1b runs

accel-gen-ddp:
	@echo "==> Generating accelerate DDP config for GPUS=$(GPUS)"
	$(PYTHON) -m config_gen.accel_gen --strategy ddp --gpus $(GPUS)

accel-gen-fsdp:
	@echo "==> Generating accelerate FSDP config for GPUS=$(GPUS)"
	$(PYTHON) -m config_gen.accel_gen --strategy fsdp --gpus $(GPUS)

# Pretrain
pretrain:
	@echo "==> Stage 4b: Pretraining ($(SIZE), $(GPUS) GPU(s), config=$(PRETRAIN_CONFIG))"
	$(ACCELERATE) pretrain/train.py \
		--config $(PRETRAIN_CONFIG)
	@$(MAKE) smoke-gen SIZE=$(SIZE)

pretrain-resume:
	$(ACCELERATE) pretrain/train.py \
		--config $(PRETRAIN_CONFIG) \
		--resume

pretrain-mini:
	@echo "==> Stage 4b: Mini pretraining run (pipeline validation)"
	$(ACCELERATE) pretrain/train.py \
		--config pretrain/configs/gpt_mini.yaml
	@$(MAKE) smoke-gen SIZE=mini

pretrain-smoke:
	@echo "==> Stage 4b: Tiny pretraining smoke run (DDP/pipeline validation)"
	$(ACCELERATE) pretrain/train.py \
		--config pretrain/configs/gpt_smoke.yaml

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
			--model results/runs/$(SIZE)/pretrain/final \
			--max-new-tokens 30 \
			--greedy; \
		echo ""; \
	done	

reinit-embeds:
	@echo "==> Stage 4c: Re-init chat special-token embeddings ($(SIZE))"
	$(PYTHON) scripts/reinit_special_embeds.py --size $(SIZE)

# ── Stage 5: SFT ──────────────────────────────────────────────────────────────

prepare-sft:
	@echo "==> Stage 5a: Prepare SFT data ($(SIZE))"
	$(PYTHON) finetune/data/prepare_sft.py --stage both --size $(SIZE)

sft-instruct:
	@echo "==> Stage 5b: Instruct SFT ($(SIZE), $(GPUS) GPU(s), config=$(SFT_INSTRUCT_CONFIG))"
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_INSTRUCT_CONFIG)

sft: sft-instruct


sft-instruct-resume:
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_INSTRUCT_CONFIG) \
		--resume

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
		--resume

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
	$(PYTHON) alignment/data/prepare_dpo.py --size $(SIZE)

dpo-chat:
	@echo "==> Stage 6b: Chat DPO alignment ($(SIZE), $(GPUS) GPU(s), config=$(DPO_CHAT_CONFIG))"
	$(ACCELERATE) alignment/train_dpo.py \
		--config $(DPO_CHAT_CONFIG)

dpo-chat-resume:
	$(ACCELERATE) alignment/train_dpo.py \
		--config $(DPO_CHAT_CONFIG) \
		--resume

dpo-chat-mini:
	@echo "==> Stage 6b: Mini chat DPO (pipeline validation)"
	$(ACCELERATE) alignment/train_dpo.py \
		--config alignment/configs/dpo_chat_mini.yaml

dpo: dpo-chat

dpo-resume: dpo-chat-resume

dpo-mini: dpo-chat-mini

# ── Stage 7: Evaluation ───────────────────────────────────────────────────────

# Default eval target — alias for eval-chat (the final aligned variant).
eval: eval-chat

eval-base:
	@echo "==> Stage 7: Evaluation (base, $(SIZE))"
	$(PYTHON) eval/eval.py --model results/runs/$(SIZE)/pretrain/final

eval-instruct:
	@echo "==> Stage 7: Evaluation (instruct, $(SIZE))"
	$(PYTHON) eval/eval.py --model results/runs/$(SIZE)/sft_instruct/final


eval-chat:
	@echo "==> Stage 7: Evaluation (chat, $(SIZE))"
	$(PYTHON) eval/eval.py --model results/runs/$(SIZE)/dpo_chat/final

eval-code:
	@echo "==> Stage 7: Evaluation (code, $(SIZE))"
	$(PYTHON) eval/eval.py --model results/runs/$(SIZE)/sft_code/final


# Behavior sanity eval targets.
# These are deterministic generation checks for factuality, task format,
# code behavior, repetition, and clean stopping. They complement lm-eval
# benchmarks; they do not replace them.
eval-sanity: eval-sanity-chat

eval-sanity-base:
	@echo "==> Stage 7: Sanity evaluation (base, $(SIZE))"
	$(PYTHON) eval/sanity_eval.py \
		--model results/runs/$(SIZE)/pretrain/final \
		--json-out results/runs/$(SIZE)/eval/sanity/base.json

eval-sanity-instruct:
	@echo "==> Stage 7: Sanity evaluation (instruct, $(SIZE))"
	$(PYTHON) eval/sanity_eval.py \
		--model results/runs/$(SIZE)/sft_instruct/final \
		--json-out results/runs/$(SIZE)/eval/sanity/instruct.json


eval-sanity-chat:
	@echo "==> Stage 7: Sanity evaluation (chat, $(SIZE))"
	$(PYTHON) eval/sanity_eval.py \
		--model results/runs/$(SIZE)/dpo_chat/final \
		--json-out results/runs/$(SIZE)/eval/sanity/chat.json

eval-sanity-code:
	@echo "==> Stage 7: Sanity evaluation (code, $(SIZE))"
	$(PYTHON) eval/sanity_eval.py \
		--model results/runs/$(SIZE)/sft_code/final \
		--json-out results/runs/$(SIZE)/eval/sanity/code.json

eval-mini:
	@echo "==> Stage 7: Mini evaluation (pipeline validation)"
	$(PYTHON) eval/eval.py --model results/runs/$(SIZE)/dpo_chat/final --tasks hellaswag --limit 50 --batch-size 4

# ── Stage 8: Export ───────────────────────────────────────────────────────────

export: export-base export-instruct export-chat export-code
	@echo "All variants exported for slm-$(SIZE)"

export-base:
	@echo "==> Stage 8: Export base model ($(SIZE))"
	$(PYTHON) export/export.py --size $(SIZE) --variant base

export-instruct:
	@echo "==> Stage 8: Export instruct model ($(SIZE))"
	$(PYTHON) export/export.py --size $(SIZE) --variant instruct

export-chat:
	@echo "==> Stage 8: Export chat model ($(SIZE))"
	$(PYTHON) export/export.py --size $(SIZE) --variant chat

export-code:
	@echo "==> Stage 8: Export code model ($(SIZE))"
	$(PYTHON) export/export.py --size $(SIZE) --variant code

# ── Stage 10: Serve ───────────────────────────────────────────────────────────

serve:
	@echo "==> Stage 10: Serve chat model ($(SIZE))"
	MODEL=tohio/slm-$(SIZE)-chat ./serve/serve.sh


serve-local:
	@echo "==> Stage 10: Serve local chat checkpoint ($(SIZE))"
	MODEL=results/runs/$(SIZE)/dpo_chat/final ./serve/serve.sh


# ── S3 utilities ──────────────────────────────────────────────────────────────

s3-upload:
	$(PYTHON) curator/scripts/upload_s3.py upload --src "$(DATA_DIR)/runs/$(SIZE)/curated" --dst "runs/$(SIZE)/curated"

s3-download:
	$(PYTHON) curator/scripts/upload_s3.py download --src "runs/$(SIZE)/curated" --dst "$(DATA_DIR)/runs/$(SIZE)/curated"

s3-list:
	$(PYTHON) curator/scripts/upload_s3.py list

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	@echo "==> Running instance setup..."
	bash infra/setup.sh

setup-data-dir:
	@echo "==> Running instance setup with custom data dir..."
	bash infra/setup.sh --data-dir $(DATA_DIR)

setup-gpu:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required for setup-gpu"; exit 1)
	@echo "==> Running GPU instance setup (DATA_DIR=$(DATA_DIR), RUN_ID=$(RUN_ID))..."
	bash infra/setup_gpu_instance.sh --data-dir $(DATA_DIR) --size $(SIZE) --run-id $(RUN_ID)
	$(MAKE) restore-size-tokenizer SIZE=$(SIZE) DATA_DIR=$(DATA_DIR)
install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install orjson fasttext-wheel

install-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Install uv first: https://docs.astral.sh/uv/"; \
		exit 1; \
	fi
	uv venv --python 3.12
	uv pip install --upgrade pip
	uv pip install -r requirements.txt
	uv pip install orjson fasttext-wheel


install-conda:
	@if ! command -v conda >/dev/null 2>&1; then \
		echo "conda is not installed."; \
		exit 1; \
	fi
	conda create -n slm python=3.12 -y
	conda run -n slm python -m pip install --upgrade pip
	conda run -n slm pip install -r requirements.txt
	conda run -n slm pip install orjson fasttext-wheel


install-kenlm:
	.venv/bin/pip install https://github.com/kpu/kenlm/archive/master.zip

install-orjson:
	.venv/bin/pip install orjson fasttext-wheel

install-gpu:
	@echo "==> Installing dependencies for GPU training instance..."
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install orjson fasttext-wheel

download-kenlm-model:
	@echo "==> Downloading KenLM English model (~4GB)..."
	mkdir -p $(DATA_DIR)/models
	wget -q --show-progress \
		https://dl.fbaipublicfiles.com/cc_net/lm/en.arpa.bin \
		-O $(DATA_DIR)/models/en.arpa.bin
	@echo "  Saved to $(DATA_DIR)/models/en.arpa.bin"

download-fasttext-model:
	@echo "==> Downloading fasttext language identification model (~1MB)..."
	mkdir -p $(DATA_DIR)/models
	wget -q --show-progress \
		https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
		-O $(DATA_DIR)/models/lid.176.ftz
	@echo "  Saved to $(DATA_DIR)/models/lid.176.ftz"

accelerate-config:
	accelerate config

accelerate-config-single:
	@echo "==> Configuring accelerate for single GPU..."
	mkdir -p ~/.cache/huggingface/accelerate
	cp accelerate_configs/single_gpu.yaml ~/.cache/huggingface/accelerate/default_config.yaml
	@echo "  Single GPU config active"

accelerate-config-multi:
	@echo "==> Configuring accelerate for multi-GPU ($(GPUS) GPUs)..."
	mkdir -p ~/.cache/huggingface/accelerate
	cat accelerate_configs/multi_gpu.yaml | sed 's/num_processes: 8/num_processes: $(GPUS)/' > ~/.cache/huggingface/accelerate/default_config.yaml
	@echo "  Multi-GPU config active ($(GPUS) processes)"

# ── Tests ─────────────────────────────────────────────────────────────────────

test-curator:
	@echo "==> Validating curate-mini outputs..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/data_pipeline/test_pipeline_curator.py --size=$(TEST_SIZE) -v --tb=short

test-validate:
	@echo "==> Validating validate outputs..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/data_pipeline/test_pipeline_validate.py --size=$(TEST_SIZE) -v --tb=short

test-tokenizer:
	@echo "==> Validating tokenizer and tokenized binary outputs..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/data_pipeline/test_pipeline_tokenizer.py --size=$(TEST_SIZE) -v --tb=short

test-data-pipeline: test-curator test-validate test-tokenizer
	@echo "==> Data pipeline tests complete"

# GPU pipeline tests respect TEST_SIZE. By default TEST_SIZE=mini so normal
# development stays mini-focused. Passing SIZE=125m / 350m / 1b on the make
# command line also sets TEST_SIZE to that size for full-artifact checks.
test-training:
	@echo "==> Validating pretrain outputs (TEST_SIZE=$(TEST_SIZE))..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/gpu_pipeline/test_pipeline_training.py --size=$(TEST_SIZE) -v --tb=short

test-sft-instruct:
	@echo "==> Validating instruct SFT outputs (TEST_SIZE=$(TEST_SIZE))..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/gpu_pipeline/test_pipeline_sft.py::TestChatSFTModel tests/gpu_pipeline/test_pipeline_sft.py::TestSFTData --size=$(TEST_SIZE) -v --tb=short

test-sft-chat: test-sft-instruct


test-sft-code:
	@echo "==> Validating code SFT outputs (TEST_SIZE=$(TEST_SIZE))..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/gpu_pipeline/test_pipeline_sft.py::TestCodeSFTModel --size=$(TEST_SIZE) -v --tb=short

test-dpo-chat:
	@echo "==> Validating chat DPO outputs (TEST_SIZE=$(TEST_SIZE))..."
	PIPELINE_TEST_SIZE=$(TEST_SIZE) .venv/bin/pytest tests/gpu_pipeline/test_pipeline_dpo.py --size=$(TEST_SIZE) -v --tb=short

test-dpo: test-dpo-chat


test-gpu-pipeline: test-training test-sft-instruct test-sft-code test-dpo-chat
	@echo "==> GPU pipeline tests complete"

test-model:
	@echo "==> Running model unit tests..."
	.venv/bin/pytest tests/model/ -v --tb=short

test-config-gen:
	@echo "==> Running config_gen unit tests..."
	.venv/bin/pytest tests/test_config_gen.py -v --tb=short

test-accel-gen:
	@echo "==> Running accel_gen unit tests..."
	.venv/bin/pytest tests/test_accel_gen.py -v --tb=short

test-unit: test-model test-config-gen test-accel-gen
	@echo "==> Unit tests complete"

# ── Sanity check ──────────────────────────────────────────────────────────────

sanity-train:
	@echo "==> Sanity training: 125m arch on FineWeb-Edu (~2.5B tokens)"
	$(PYTHON) scripts/sanity_train.py --arch 125m --target-tokens 2500000000

sanity-train-small:
	@echo "==> Sanity training: mini arch on FineWeb-Edu (~500M tokens)"
	$(PYTHON) scripts/sanity_train.py --arch mini --target-tokens 500000000

sanity-train-tiny:
	@echo "==> Sanity training: mini arch on FineWeb-Edu (~50M tokens)"
	$(PYTHON) scripts/sanity_train.py --arch mini --target-tokens 50000000

sanity-train-save:
	@echo "==> Sanity training (SANITY_SIZE=$(SANITY_SIZE), saves to results/sanity-*)"
ifeq ($(SANITY_SIZE),small)
	$(PYTHON) scripts/sanity_train.py --arch mini --target-tokens 500000000 --save
else ifeq ($(SANITY_SIZE),tiny)
	$(PYTHON) scripts/sanity_train.py --arch mini --target-tokens 50000000 --save
else
	$(PYTHON) scripts/sanity_train.py --arch 125m --target-tokens 2500000000 --save
endif

# ── Clean ─────────────────────────────────────────────────────────────────────

clean-data:
	rm -rf "$(DATA_DIR)/runs/$(SIZE)" "$(DATA_DIR)/dedup_scratch"

clean-results:
	rm -rf results/

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
	@echo "Usage: make <target> [SIZE=125m|350m|1b] [GPUS=N] [WORKERS=N] [DATA_DIR=path]"
	@echo "       make config-gen-* [GPU=h200|b200|...] [MODE=conservative|balanced|aggressive]"
	@echo ""
	@echo "For full target documentation see: docs/COMMANDS.md"
	@echo ""
	@echo "Config generation (run before pretrain/sft-instruct/dpo-chat/sft-code):"
	@echo "  config-gen-pretrain  Auto-generate pretrain/configs/gpt_$(SIZE).yaml"
	@echo "  config-gen-sft       Auto-generate sft_instruct_$(SIZE).yaml + sft_code_$(SIZE).yaml"
	@echo "  config-gen-dpo       Auto-generate alignment/configs/dpo_chat_$(SIZE).yaml"
	@echo "  config-gen           Convenience: runs all three above"
	@echo "  accel-gen-ddp        Auto-generate accelerate_configs/multi_gpu.yaml"
	@echo "  accel-gen-fsdp       Auto-generate accelerate_configs/fsdp.yaml (for 1b runs)"
	@echo ""
	@echo "One-time setup:"
	@echo "  setup                    Bootstrap a fresh CPU curation instance"
	@echo "  setup-gpu                Bootstrap a GPU training instance"
	@echo "  setup-data-dir           Bootstrap with custom data dir"
	@echo "  download-fasttext-model  Download fasttext language ID model (~1MB)"
	@echo "  download-kenlm-model     Download KenLM English model (~4GB)"
	@echo "  accelerate-config        Configure accelerate interactively"
	@echo "  install                  Install dependencies (pip)"
	@echo "  install-uv               Install dependencies (uv)"
	@echo "  install-conda            Install dependencies (conda)"
	@echo "  install-kenlm            Install KenLM Python bindings from source"
	@echo "  install-orjson           Install orjson and fasttext-wheel"
	@echo ""
	@echo "Tests (CPU — data pipeline):"
	@echo "  test-curator             Validate curate-mini outputs"
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
	@echo "Tests (unit — no pipeline outputs needed):"
	@echo "  test-model               Model architecture unit tests"
	@echo "  test-config-gen          Config generator unit tests"
	@echo "  test-accel-gen           Accelerate config generator unit tests"
	@echo "  test-unit                All unit tests above"
	@echo ""
	@echo "Sanity check (model + training code only):"
	@echo "  sanity-train             125m arch, 2.5B tokens"
	@echo "  sanity-train-small       mini arch, 500M tokens"
	@echo "  sanity-train-tiny        mini arch, 50M tokens"
	@echo "  sanity-train-save        same as sanity-train but saves the model"
	@echo ""
	@echo "Pipeline:"
	@echo "  curate             Stage 1  — download, curate, blend, upload"
	@echo "  curate-mini        Stage 1  — mini run for pipeline validation"
	@echo "  validate           Stage 2  — perplexity filter on train + val splits"
	@echo "  validate-upload    Stage 2  — upload validated data to S3"
	@echo "  tokenizer          Stage 3  — train BPE tokenizer"
	@echo "  tokenize           Stage 4a — tokenize train + val to binaries"
	@echo "  artifacts-upload   Upload artifacts to S3 using a RUN_ID"
	@echo "  artifacts-download Download artifacts from S3 using RUN_ID=<run_id>"
	@echo "  pretrain           Stage 4b — pretrain from scratch (auto-runs smoke-gen)"
	@echo "  pretrain-mini      Stage 4b — mini pretrain run (auto-runs smoke-gen)"
	@echo "  smoke-gen          Stage 4b — generate from results/runs/\$$(SIZE)/pretrain/final to spot-check"
	@echo "  reinit-embeds      Stage 4c — re-init chat special-token embeds before SFT"
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
	@echo "  eval-mini              Stage 7  — mini eval (pipeline validation)"
	@echo "  export             Stage 8  — push all variants to HuggingFace Hub"
	@echo "  serve              Stage 10 — launch vLLM server"
	@echo ""

.PHONY: restore-size-tokenizer
restore-size-tokenizer:
	@echo "==> Restoring size-specific tokenizer ($(SIZE)) into active tokenizer path..."
	@test -d "$(DATA_DIR)/runs/$(SIZE)/tokenizer" || (echo "Missing $(DATA_DIR)/runs/$(SIZE)/tokenizer — run setup-gpu or artifacts-download first" && exit 1)
	mkdir -p "$(DATA_DIR)/tokenizer"
	cp -a "$(DATA_DIR)/runs/$(SIZE)/tokenizer/." "$(DATA_DIR)/tokenizer/"
	@echo "  Active tokenizer restored from $(DATA_DIR)/runs/$(SIZE)/tokenizer"
