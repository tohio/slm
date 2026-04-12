# SLM Pipeline Makefile
# ----------------------
# Usage:
#   make <target>                                        # defaults: SIZE=125m, GPUS=1
#   make <target> SIZE=350m                              # different model size
#   make <target> GPUS=4                                 # multi-GPU
#   make <target> WORKERS=16                             # parallel workers for dedup
#   make <target> CONFIG=pretrain/configs/gpt_125m.yaml  # explicit config override
#
# Full pipeline:
#   make all SIZE=125m GPUS=4
#
# See docs/COMMANDS.md for full target documentation.

SIZE    ?= 125m
GPUS    ?= 1
WORKERS ?=
DATA_DIR ?= data
DATE     ?= $(shell date +%Y-%m-%d)

# Use the venv python by default so make targets work without activating the venv.
# Override with: make pretrain PYTHON=python3
PYTHON ?= .venv/bin/python

# Config defaults — overridable with CONFIG=path/to/config.yaml
PRETRAIN_CONFIG ?= pretrain/configs/gpt_$(SIZE).yaml
SFT_CHAT_CONFIG ?= finetune/configs/sft_chat_$(SIZE).yaml
SFT_CODE_CONFIG ?= finetune/configs/sft_code_$(SIZE).yaml
DPO_CONFIG      ?= alignment/configs/dpo_$(SIZE).yaml

# accelerate launch with GPU count
ACCELERATE = accelerate launch --num_processes $(GPUS)

# Optional workers flag for dedup
ifdef WORKERS
  WORKERS_FLAG = --workers $(WORKERS)
else
  WORKERS_FLAG =
endif

.PHONY: all curate curate-mini curate-download curate-filter curate-dedup \
        curate-blend curate-upload validate validate-upload validate-datatrove \
        tokenizer tokenizer-test tokenize tokenize-upload tokenize-download \
        pretrain pretrain-mini pretrain-resume prepare-sft sft sft-mini sft-resume sft-code sft-code-mini sft-code-resume \
        prepare-dpo dpo dpo-resume eval export serve serve-local \
        export export-base export-instruct export-chat \
        setup setup-data-dir install install-uv install-conda install-kenlm \
        download-kenlm-model download-fasttext-model accelerate-config \
        s3-upload s3-download s3-list \
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
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage filter

curate-dedup:
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage dedup $(WORKERS_FLAG)

curate-blend:
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage blend

curate-upload:
	@echo "==> Stage 1: Upload curated data to S3 (target=$(SIZE))"
	$(PYTHON) curator/scripts/curate.py --target $(SIZE) --stage upload

# ── Stage 2: Validation ───────────────────────────────────────────────────────

validate:
	@echo "==> Stage 2: Validation"
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
	@echo "==> Stage 4a: Tokenize dataset"
	$(PYTHON) pretrain/data/tokenize_data.py --chunk-size 256 --verify

tokenize-upload:
	@echo "==> Stage 4a: Upload tokenized binary to S3 (target=$(SIZE))"
	$(PYTHON) pretrain/data/upload_tokenized.py upload --target $(SIZE)

tokenize-download:
	@echo "==> Stage 4a: Download tokenized binary from S3 (target=$(SIZE), date=$(DATE))"
	$(PYTHON) pretrain/data/upload_tokenized.py download --target $(SIZE) --date $(DATE)

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

install:
	pip install -r requirements.txt

install-uv:
	uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

install-conda:
	conda create -n slm python=3.12 -y && \
	conda activate slm && \
	pip install -r requirements.txt

install-kenlm:
	pip install https://github.com/kpu/kenlm/archive/master.zip

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
	@echo "  setup                    Bootstrap a fresh instance"
	@echo "  setup-data-dir           Bootstrap with custom data dir"
	@echo "  download-fasttext-model  Download fasttext language ID model (~1MB)"
	@echo "  download-kenlm-model     Download KenLM English model (~4GB)"
	@echo "  accelerate-config        Configure accelerate for multi-GPU"
	@echo "  install                  Install dependencies (pip)"
	@echo "  install-uv               Install dependencies (uv)"
	@echo "  install-conda            Install dependencies (conda)"
	@echo "  install-kenlm            Install KenLM Python bindings from source"
	@echo ""
	@echo "Pipeline:"
	@echo "  curate             Stage 1  — download, curate, and upload to S3"
	@echo "  curate-mini        Stage 1  — mini run for pipeline validation (~30 min)"
	@echo "  validate           Stage 2  — perplexity filter and validate"
	@echo "  validate-upload    Stage 2  — upload validated data to S3"
	@echo "  tokenizer          Stage 3  — train BPE tokenizer"
	@echo "  tokenize           Stage 4a — tokenize dataset to binary"
	@echo "  tokenize-upload    Stage 4a — upload tokenized binary to S3"
	@echo "  tokenize-download  Stage 4a — download tokenized binary from S3"
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
	@echo "  curate-blend       Blend to train.jsonl only"
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
	@echo "  make curate SIZE=125m WORKERS=16"
	@echo "  make pretrain SIZE=125m GPUS=4"
	@echo "  make pretrain CONFIG=pretrain/configs/gpt_1b.yaml GPUS=8"
	@echo "  make sft SIZE=125m GPUS=4"
	@echo "  make dpo SIZE=125m GPUS=2"
	@echo "  make download-kenlm-model DATA_DIR=/data/slm/data"
	@echo "  make download-fasttext-model DATA_DIR=/data/slm/data"
	@echo ""