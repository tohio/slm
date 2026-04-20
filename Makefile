# SLM Pipeline Makefile
# ----------------------
# Usage:
#   make <target>                                        # defaults: SIZE=125m, GPUS=1
#   make <target> SIZE=350m                              # different model size
#   make <target> GPUS=4                                 # multi-GPU
#   make <target> WORKERS=16                             # parallel workers for filter, dedup, blend
#   make <target> CONFIG=pretrain/configs/gpt_125m.yaml  # explicit config override
#
# Full pipeline:
#   make all SIZE=125m GPUS=4
#
# See docs/COMMANDS.md for full target documentation.

SIZE    ?= 125m
GPUS    ?= 1
WORKERS ?=
DATE     ?= $(shell date +%Y-%m-%d)

# DATA_DIR — read from .env if not set in environment.
# This ensures make targets use the correct path without requiring
# the user to export DATA_DIR manually after each login.
DATA_DIR ?= $(shell grep -v '^\#' .env 2>/dev/null | grep '^DATA_DIR=' | head -1 | cut -d= -f2 | tr -d ' ')
DATA_DIR ?= data

# Use the venv python and accelerate so make targets work without activating the venv.
# Override with: make pretrain PYTHON=python3
PYTHON     ?= .venv/bin/python
_ACCELERATE = .venv/bin/accelerate

# Config defaults — overridable with CONFIG=path/to/config.yaml
PRETRAIN_CONFIG ?= pretrain/configs/gpt_$(SIZE).yaml
SFT_CHAT_CONFIG ?= finetune/configs/sft_chat_$(SIZE).yaml
SFT_CODE_CONFIG ?= finetune/configs/sft_code_$(SIZE).yaml
DPO_CONFIG      ?= alignment/configs/dpo_$(SIZE).yaml

# accelerate launch with GPU count
ACCELERATE = $(_ACCELERATE) launch --num_processes $(GPUS)

# Optional workers flag — passed to filter, dedup, and blend stages
ifdef WORKERS
  WORKERS_FLAG = --workers $(WORKERS)
else
  WORKERS_FLAG =
endif

.PHONY: all curate curate-mini curate-download curate-filter curate-dedup \
        curate-blend curate-upload validate validate-upload validate-datatrove \
        tokenizer tokenizer-test tokenize tokenize-upload tokenize-download tokenizer-upload tokenizer-download \
        pretrain pretrain-mini pretrain-resume prepare-sft sft sft-mini sft-resume sft-code sft-code-mini sft-code-resume \
        prepare-dpo dpo dpo-resume eval export serve serve-local \
        export export-base export-instruct export-chat \
        setup setup-data-dir setup-gpu install install-gpu install-uv install-conda install-kenlm install-orjson \
        download-kenlm-model download-fasttext-model accelerate-config accelerate-config-single accelerate-config-multi \
        s3-upload s3-download s3-list \
        test-curator test-validate test-tokenizer test-data-pipeline \
        test-training test-sft-chat test-sft-code test-dpo test-gpu-pipeline test-model \
        clean clean-data clean-results clean-logs help

# ── Full pipeline ──────────────────────────────────────────────────────────────

all: curate validate tokenizer tokenize pretrain prepare-sft sft sft-code prepare-dpo dpo
	@echo "Pipeline complete for slm-$(SIZE) on $(GPUS) GPU(s)"

# ── Stage 1: Data curation ────────────────────────────────────────────────────

curate:
	@echo "==> Stage 1: Curation (target=$(SIZE))"
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) $(WORKERS_FLAG)

curate-mini:
	@echo "==> Stage 1: Mini curation run (pipeline validation)"
	$(PYTHON) curator/scripts/curate.py --target mini --mini $(WORKERS_FLAG)

curate-download:
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage download

curate-filter:
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage filter $(WORKERS_FLAG)

curate-dedup:
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage dedup $(WORKERS_FLAG)

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

tokenize-upload:
	@echo "==> Stage 4a: Upload tokenized binaries to S3 (target=$(SIZE))"
	$(PYTHON) pretrain/data/upload_tokenized.py upload --target $(SIZE)

tokenize-download:
	@echo "==> Stage 4a: Download tokenized binaries from S3 (target=$(SIZE), date=$(DATE))"
	$(PYTHON) pretrain/data/upload_tokenized.py download --target $(SIZE) --date $(DATE)

tokenizer-upload:
	@echo "==> Uploading tokenizer to S3..."
	$(PYTHON) curator/scripts/upload_s3.py upload --src $(DATA_DIR)/tokenizer --dst tokenizer
	@echo "  Tokenizer uploaded"

tokenizer-download:
	@echo "==> Downloading tokenizer from S3..."
	mkdir -p $(DATA_DIR)/tokenizer
	$(PYTHON) curator/scripts/upload_s3.py download --src tokenizer --dst $(DATA_DIR)/tokenizer
	@echo "  Tokenizer downloaded to $(DATA_DIR)/tokenizer/"

pretrain:
	@echo "==> Stage 4b: Pretraining ($(SIZE), $(GPUS) GPU(s), config=$(PRETRAIN_CONFIG))"
	$(ACCELERATE) pretrain/train.py \
		--config $(PRETRAIN_CONFIG)

pretrain-resume:
	$(ACCELERATE) pretrain/train.py \
		--config $(PRETRAIN_CONFIG) \
		--resume

pretrain-mini:
	@echo "==> Stage 4b: Mini pretraining run (pipeline validation)"
	$(ACCELERATE) pretrain/train.py \
		--config pretrain/configs/gpt_mini.yaml

# ── Stage 5: SFT ──────────────────────────────────────────────────────────────

prepare-sft:
	@echo "==> Stage 5a: Prepare SFT data"
	$(PYTHON) finetune/data/prepare_sft.py --stage both

sft:
	@echo "==> Stage 5b: Chat SFT ($(SIZE), $(GPUS) GPU(s), config=$(SFT_CHAT_CONFIG))"
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_CHAT_CONFIG)

sft-resume:
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_CHAT_CONFIG) \
		--resume

sft-mini:
	@echo "==> Stage 5b: Mini chat SFT (pipeline validation)"
	$(ACCELERATE) finetune/train_sft.py \
		--config finetune/configs/sft_chat_mini.yaml

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

# ── Stage 6: DPO ──────────────────────────────────────────────────────────────

prepare-dpo:
	@echo "==> Stage 6a: Prepare DPO data"
	$(PYTHON) alignment/data/prepare_dpo.py

dpo:
	@echo "==> Stage 6b: DPO alignment ($(SIZE), $(GPUS) GPU(s), config=$(DPO_CONFIG))"
	$(ACCELERATE) alignment/train_dpo.py \
		--config $(DPO_CONFIG)

dpo-resume:
	$(ACCELERATE) alignment/train_dpo.py \
		--config $(DPO_CONFIG) \
		--resume

dpo-mini:
	@echo "==> Stage 6b: Mini DPO (pipeline validation)"
	$(ACCELERATE) alignment/train_dpo.py \
		--config alignment/configs/dpo_mini.yaml

# ── Stage 7: Evaluation ───────────────────────────────────────────────────────

eval:
	@echo "==> Stage 7: Evaluation ($(SIZE))"
	$(PYTHON) eval/eval.py --model results/slm-$(SIZE)-dpo/final

eval-mini:
	@echo "==> Stage 7: Mini evaluation (pipeline validation)"
	$(PYTHON) eval/eval.py --model results/slm-mini-dpo/final --tasks hellaswag --limit 50 --batch-size 4

# ── Stage 8: Export ───────────────────────────────────────────────────────────

export: export-base export-instruct export-chat
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

# ── Stage 10: Serve ───────────────────────────────────────────────────────────

serve:
	@echo "==> Stage 10: Serve ($(SIZE))"
	MODEL=tohio/slm-$(SIZE) ./serve/serve.sh

serve-local:
	@echo "==> Stage 10: Serve local checkpoint ($(SIZE))"
	MODEL=results/slm-$(SIZE)-dpo/final ./serve/serve.sh

# ── S3 utilities ──────────────────────────────────────────────────────────────

s3-upload:
	$(PYTHON) curator/scripts/upload_s3.py upload --src data/curated --dst curated

s3-download:
	$(PYTHON) curator/scripts/upload_s3.py download --src curated --dst data/curated

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
	@echo "==> Running GPU instance setup (DATA_DIR=$(DATA_DIR))..."
	bash infra/setup_gpu_instance.sh --data-dir $(DATA_DIR) --size $(SIZE) $(if $(DATE),--date $(DATE),)

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install orjson fasttext-wheel

install-uv:
	uv venv && uv pip install -r requirements.txt && uv pip install orjson fasttext-wheel

install-conda:
	conda create -n slm python=3.12 -y && \
	conda activate slm && \
	pip install -r requirements.txt && \
	pip install orjson fasttext-wheel

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
# Data pipeline tests — run on CPU curation instance after each stage.
# GPU pipeline tests — run on GPU instance after each training stage.

test-curator:
	@echo "==> Validating curate-mini outputs..."
	.venv/bin/pytest tests/data_pipeline/test_pipeline_curator.py -v --tb=short

test-validate:
	@echo "==> Validating validate outputs..."
	.venv/bin/pytest tests/data_pipeline/test_pipeline_validate.py -v --tb=short

test-tokenizer:
	@echo "==> Validating tokenizer outputs..."
	.venv/bin/pytest tests/data_pipeline/test_pipeline_tokenizer.py -v --tb=short

test-data-pipeline: test-curator test-validate test-tokenizer
	@echo "==> Data pipeline tests complete"

test-training:
	@echo "==> Validating pretrain-mini outputs..."
	.venv/bin/pytest tests/gpu_pipeline/test_pipeline_training.py -v --tb=short

test-sft-chat:
	@echo "==> Validating sft-mini outputs..."
	.venv/bin/pytest tests/gpu_pipeline/test_pipeline_sft.py::TestChatSFTModel tests/gpu_pipeline/test_pipeline_sft.py::TestSFTData -v --tb=short

test-sft-code:
	@echo "==> Validating sft-code-mini outputs..."
	.venv/bin/pytest tests/gpu_pipeline/test_pipeline_sft.py::TestCodeSFTModel -v --tb=short

test-dpo:
	@echo "==> Validating dpo-mini outputs..."
	.venv/bin/pytest tests/gpu_pipeline/test_pipeline_dpo.py -v --tb=short

test-gpu-pipeline: test-training test-sft-chat test-sft-code test-dpo
	@echo "==> GPU pipeline tests complete"

test-model:
	@echo "==> Running model unit tests..."
	.venv/bin/pytest tests/model/ -v --tb=short

# ── Clean ─────────────────────────────────────────────────────────────────────

clean-data:
	rm -rf data/raw data/filtered data/curated data/validated data/tokenized data/sft data/dpo data/dedup_scratch

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
	@echo ""
	@echo "For full target documentation see: docs/COMMANDS.md"
	@echo ""
	@echo "One-time setup:"
	@echo "  setup                    Bootstrap a fresh CPU curation instance"
	@echo "  setup-gpu                Bootstrap a GPU training instance (run after each preemptible restart)"
	@echo "  setup-data-dir           Bootstrap with custom data dir"
	@echo "  download-fasttext-model  Download fasttext language ID model (~1MB)"
	@echo "  download-kenlm-model     Download KenLM English model (~4GB)"
	@echo "  accelerate-config        Configure accelerate for multi-GPU"
	@echo "  install                  Install dependencies (pip)"
	@echo "  install-uv               Install dependencies (uv)"
	@echo "  install-conda            Install dependencies (conda)"
	@echo "  install-kenlm            Install KenLM Python bindings from source"
	@echo "  install-orjson           Install orjson and fasttext-wheel"
	@echo ""
	@echo "Tests (CPU — data pipeline):"
	@echo "  test-curator             Validate curate-mini outputs (run after make curate-mini)"
	@echo "  test-validate            Validate validate outputs (run after make validate)"
	@echo "  test-tokenizer           Validate tokenizer outputs (run after make tokenizer)"
	@echo "  test-data-pipeline       Run all data pipeline tests"
	@echo ""
	@echo "Tests (GPU — training pipeline):"
	@echo "  test-training            Validate pretrain-mini outputs (run after make pretrain-mini)"
	@echo "  test-sft-chat            Validate sft-mini outputs (run after make sft-mini)"
	@echo "  test-sft-code            Validate sft-code-mini outputs (run after make sft-code-mini)"
	@echo "  test-dpo                 Validate dpo-mini outputs (run after make dpo-mini)"
	@echo "  test-gpu-pipeline        Run all GPU pipeline tests"
	@echo ""
	@echo "Tests (unit — no pipeline outputs needed):"
	@echo "  test-model               Model architecture unit tests"
	@echo ""
	@echo "Pipeline:"
	@echo "  curate             Stage 1  — download, curate, blend to train.jsonl + val.jsonl, upload"
	@echo "  curate-mini        Stage 1  — mini run for pipeline validation (~30 min)"
	@echo "  validate           Stage 2  — perplexity filter on train + val splits"
	@echo "  validate-upload    Stage 2  — upload validated data to S3"
	@echo "  tokenizer          Stage 3  — train BPE tokenizer"
	@echo "  tokenize           Stage 4a — tokenize train + val to binaries"
	@echo "  tokenize-upload    Stage 4a — upload tokenized binaries to S3"
	@echo "  tokenizer-upload   Stage 3  — upload tokenizer to S3"
	@echo "  tokenizer-download Stage 3  — download tokenizer from S3"
	@echo "  tokenize-download  Stage 4a — download tokenized binaries from S3"
	@echo "  pretrain           Stage 4b — pretrain from scratch"
	@echo "  pretrain-mini      Stage 4b — mini pretrain run for pipeline validation"
	@echo "  sft-mini           Stage 5b — mini chat SFT for pipeline validation"
	@echo "  sft-code-mini      Stage 5c — mini code SFT for pipeline validation"
	@echo "  dpo-mini           Stage 6b — mini DPO for pipeline validation"
	@echo "  prepare-sft        Stage 5a — download SFT datasets"
	@echo "  sft                Stage 5b — chat supervised fine-tuning"
	@echo "  sft-code           Stage 5c — code supervised fine-tuning"
	@echo "  prepare-dpo        Stage 6a — download DPO datasets"
	@echo "  dpo                Stage 6b — DPO alignment"
	@echo "  eval               Stage 7  — benchmark evaluation"
	@echo "  eval-mini          Stage 7  — quick eval validation (hellaswag, 50 examples)"
	@echo "  export             Stage 8  — push all variants to HuggingFace Hub"
	@echo "  export-base        Stage 8  — push base model only"
	@echo "  export-instruct    Stage 8  — push instruct model only"
	@echo "  export-chat        Stage 8  — push chat model only"
	@echo "  serve              Stage 10 — launch vLLM server (Hub model)"
	@echo "  serve-local        Stage 10 — launch vLLM server (local checkpoint)"
	@echo ""
	@echo "Curation sub-stages:"
	@echo "  curate-download    Download raw data only"
	@echo "  curate-filter      Quality filter only"
	@echo "  curate-dedup       Deduplication only"
	@echo "  curate-blend       Blend to train.jsonl + val.jsonl only"
	@echo "  curate-upload      Upload curated data to S3 only"
	@echo ""
	@echo "Resume targets:"
	@echo "  pretrain-resume    Resume pretraining from last checkpoint"
	@echo "  sft-resume         Resume chat SFT from last checkpoint"
	@echo "  sft-code-resume    Resume code SFT from last checkpoint"
	@echo "  dpo-resume         Resume DPO from last checkpoint"
	@echo ""
	@echo "S3 utilities:"
	@echo "  s3-upload          Upload curated data to S3 (unversioned)"
	@echo "  s3-download        Download curated data from S3"
	@echo "  s3-list            List S3 contents"
	@echo ""
	@echo "Clean:"
	@echo "  clean              Remove cache files and logs"
	@echo "  clean-data         Remove all data directories"
	@echo "  clean-results      Remove all training results"
	@echo "  clean-logs         Remove logs directory"
	@echo ""
	@echo "Examples:"
	@echo "  make curate SIZE=125m WORKERS=62"
	@echo "  make pretrain SIZE=125m GPUS=4"
	@echo "  make pretrain CONFIG=pretrain/configs/gpt_1b.yaml GPUS=8"
	@echo "  make sft SIZE=125m GPUS=4"
	@echo "  make dpo SIZE=125m GPUS=2"
	@echo "  make download-kenlm-model DATA_DIR=/data/slm/data"
	@echo "  make download-fasttext-model DATA_DIR=/data/slm/data"
	@echo ""