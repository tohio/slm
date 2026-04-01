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

SIZE    ?= 125m
GPUS    ?= 1
WORKERS ?=

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
        curate-blend curate-upload validate tokenizer tokenize \
        pretrain prepare-sft sft sft-code \
        prepare-dpo dpo eval export serve serve-local \
        setup install install-uv install-conda install-kenlm \
        accelerate-config clean clean-data clean-results clean-logs help

# ── Full pipeline ──────────────────────────────────────────────────────────────

all: curate validate tokenizer tokenize pretrain prepare-sft sft sft-code prepare-dpo dpo
	@echo "Pipeline complete for slm-$(SIZE) on $(GPUS) GPU(s)"

# ── Stage 1: Data curation ────────────────────────────────────────────────────

curate:
	@echo "==> Stage 1: Curation (target=$(SIZE))"
	python curator/scripts/curate.py --target $(SIZE) $(WORKERS_FLAG)

curate-mini:
	@echo "==> Stage 1: Mini curation run (pipeline validation)"
	python curator/scripts/curate.py --target mini --mini $(WORKERS_FLAG)

curate-download:
	python curator/scripts/curate.py --target $(SIZE) --stage download

curate-filter:
	python curator/scripts/curate.py --target $(SIZE) --stage filter

curate-dedup:
	python curator/scripts/curate.py --target $(SIZE) --stage dedup $(WORKERS_FLAG)

curate-blend:
	python curator/scripts/curate.py --target $(SIZE) --stage blend

curate-upload:
	python curator/scripts/curate.py --target $(SIZE) --stage upload

# ── Stage 2: Validation ───────────────────────────────────────────────────────

validate:
	@echo "==> Stage 2: Validation"
	python validation/scripts/validate.py

validate-datatrove:
	python validation/scripts/validate.py --use-datatrove

# ── Stage 3: Tokenizer ────────────────────────────────────────────────────────

tokenizer:
	@echo "==> Stage 3: Tokenizer training"
	python tokenizer/train_tokenizer.py

tokenizer-test:
	python tokenizer/test_tokenizer.py

# ── Stage 4: Pretrain ─────────────────────────────────────────────────────────

tokenize:
	@echo "==> Stage 4a: Tokenize dataset"
	python pretrain/data/tokenize.py --workers 8 --verify

pretrain:
	@echo "==> Stage 4b: Pretraining ($(SIZE), $(GPUS) GPU(s), config=$(PRETRAIN_CONFIG))"
	$(ACCELERATE) pretrain/train.py \
		--config $(PRETRAIN_CONFIG)

pretrain-resume:
	$(ACCELERATE) pretrain/train.py \
		--config $(PRETRAIN_CONFIG) \
		--resume

# ── Stage 5: SFT ──────────────────────────────────────────────────────────────

prepare-sft:
	@echo "==> Stage 5a: Prepare SFT data"
	python finetune/data/prepare_sft.py --stage both

sft:
	@echo "==> Stage 5b: Chat SFT ($(SIZE), $(GPUS) GPU(s), config=$(SFT_CHAT_CONFIG))"
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_CHAT_CONFIG)

sft-resume:
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_CHAT_CONFIG) \
		--resume

sft-code:
	@echo "==> Stage 5c: Code SFT ($(SIZE), $(GPUS) GPU(s), config=$(SFT_CODE_CONFIG))"
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_CODE_CONFIG)

sft-code-resume:
	$(ACCELERATE) finetune/train_sft.py \
		--config $(SFT_CODE_CONFIG) \
		--resume

# ── Stage 6: DPO ──────────────────────────────────────────────────────────────

prepare-dpo:
	@echo "==> Stage 6a: Prepare DPO data"
	python alignment/data/prepare_dpo.py

dpo:
	@echo "==> Stage 6b: DPO alignment ($(SIZE), $(GPUS) GPU(s), config=$(DPO_CONFIG))"
	$(ACCELERATE) alignment/train_dpo.py \
		--config $(DPO_CONFIG)

dpo-resume:
	$(ACCELERATE) alignment/train_dpo.py \
		--config $(DPO_CONFIG) \
		--resume

# ── Stage 7: Evaluation ───────────────────────────────────────────────────────

eval:
	@echo "==> Stage 7: Evaluation ($(SIZE))"
	python eval/eval.py --model results/slm-$(SIZE)-dpo/final

# ── Stage 8: Export ───────────────────────────────────────────────────────────

export:
	@echo "==> Stage 8: Export to HuggingFace Hub ($(SIZE))"
	python export/export.py --model results/slm-$(SIZE)-dpo/final --size $(SIZE)

# ── Stage 10: Serve ──────────────────────────────────────────────────────────

serve:
	@echo "==> Stage 10: Serve ($(SIZE))"
	MODEL=tohio/slm-$(SIZE) ./serve/serve.sh

serve-local:
	@echo "==> Stage 10: Serve local checkpoint ($(SIZE))"
	MODEL=results/slm-$(SIZE)-dpo/final ./serve/serve.sh

# ── S3 utilities ──────────────────────────────────────────────────────────────

s3-upload:
	python curator/scripts/upload_s3.py upload --src data/curated --dst curated

s3-download:
	python curator/scripts/upload_s3.py download --src curated --dst data/curated

s3-list:
	python curator/scripts/upload_s3.py list

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
	@echo "Usage: make <target> [SIZE=125m|350m|1b] [GPUS=N] [WORKERS=N]"
	@echo ""
	@echo "Setup:"
	@echo "  setup              Bootstrap a fresh instance (run once)"
	@echo "  setup-data-dir     Bootstrap with custom data dir: make setup-data-dir DATA_DIR=/data/slm/data"
	@echo "  install            Install dependencies (pip)"
	@echo "  install-uv         Install dependencies (uv)"
	@echo "  install-conda      Install dependencies (conda)"
	@echo ""
	@echo "Pipeline stages:"
	@echo "  curate-mini     Stage 1  — mini run to validate pipeline (~30 min)"
	@echo "  curate          Stage 1  — download and curate data"
	@echo "  validate        Stage 2  — quality filter and validate"
	@echo "  tokenizer       Stage 3  — train BPE tokenizer"
	@echo "  tokenize        Stage 4a — tokenize dataset to binary"
	@echo "  pretrain        Stage 4b — pretrain from scratch"
	@echo "  prepare-sft     Stage 5a — download SFT datasets"
	@echo "  sft             Stage 5b — chat supervised fine-tuning"
	@echo "  sft-code        Stage 5c — code supervised fine-tuning"
	@echo "  prepare-dpo     Stage 6a — download DPO datasets"
	@echo "  dpo             Stage 6b — DPO alignment"
	@echo "  eval            Stage 7  — benchmark evaluation"
	@echo "  export          Stage 8  — push to HuggingFace Hub"
	@echo "  serve           Stage 10 — launch vLLM server (Hub model)"
	@echo "  serve-local     Stage 10 — launch vLLM server (local checkpoint)"
	@echo ""
	@echo "Curation sub-stages:"
	@echo "  curate-download    Download raw data only"
	@echo "  curate-filter      Quality filter only"
	@echo "  curate-dedup       Deduplication only"
	@echo "  curate-blend       Blend to train.jsonl only"
	@echo "  curate-upload      Upload to S3 only"
	@echo ""
	@echo "Utilities:"
	@echo "  install-kenlm      Install KenLM from source"
	@echo "  accelerate-config  Configure accelerate for multi-GPU"
	@echo "  s3-upload          Upload curated data to S3"
	@echo "  s3-download        Download curated data from S3"
	@echo "  s3-list            List S3 contents"
	@echo "  clean              Remove cache files"
	@echo "  clean-data         Remove all data directories"
	@echo "  clean-results      Remove all training results"
	@echo ""
	@echo "Examples:"
	@echo "  make curate-mini                                             # validate pipeline"
	@echo "  make curate SIZE=125m WORKERS=16                            # full 125M curation"
	@echo "  make all SIZE=125m GPUS=2                                   # full pipeline"
	@echo "  make pretrain SIZE=125m GPUS=4                              # 125M on 4x A100"
	@echo "  make pretrain SIZE=350m GPUS=6                              # 350M on 6x H100"
	@echo "  make pretrain CONFIG=pretrain/configs/gpt_1b.yaml GPUS=8   # 1B explicit config"
	@echo "  make sft SIZE=125m GPUS=4"
	@echo "  make dpo SIZE=125m GPUS=2"
	@echo ""