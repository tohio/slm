# SLM Pipeline Makefile
# ----------------------
# Usage:
#   make <target>              # run a stage
#   make <target> SIZE=350m    # run for a specific model size (125m, 350m, 1b)
#
# Full pipeline:
#   make all SIZE=125m

SIZE ?= 125m

.PHONY: all curate validate tokenizer tokenize \
        pretrain prepare-sft sft sft-code \
        prepare-dpo dpo eval export \
        clean help

# ── Full pipeline ──────────────────────────────────────────────────────────────

all: curate validate tokenizer tokenize pretrain prepare-sft sft sft-code prepare-dpo dpo
	@echo "Pipeline complete for slm-$(SIZE)"

# ── Stage 1: Data curation ────────────────────────────────────────────────────

curate:
	@echo "==> Stage 1: Curation (target=$(SIZE))"
	python curator/scripts/curate.py --target $(SIZE)

curate-download:
	python curator/scripts/curate.py --target $(SIZE) --stage download

curate-filter:
	python curator/scripts/curate.py --target $(SIZE) --stage filter

curate-dedup:
	python curator/scripts/curate.py --target $(SIZE) --stage dedup

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
	@echo "==> Stage 4b: Pretraining ($(SIZE))"
	accelerate launch pretrain/train.py \
		--config pretrain/configs/gpt_$(SIZE).yaml

pretrain-resume:
	accelerate launch pretrain/train.py \
		--config pretrain/configs/gpt_$(SIZE).yaml \
		--resume

# ── Stage 5: SFT ──────────────────────────────────────────────────────────────

prepare-sft:
	@echo "==> Stage 5a: Prepare SFT data"
	python finetune/data/prepare_sft.py --stage both

sft:
	@echo "==> Stage 5b: Chat SFT ($(SIZE))"
	accelerate launch finetune/train_sft.py \
		--config finetune/configs/sft_chat_$(SIZE).yaml

sft-resume:
	accelerate launch finetune/train_sft.py \
		--config finetune/configs/sft_chat_$(SIZE).yaml \
		--resume

sft-code:
	@echo "==> Stage 5c: Code SFT ($(SIZE))"
	accelerate launch finetune/train_sft.py \
		--config finetune/configs/sft_code_$(SIZE).yaml

sft-code-resume:
	accelerate launch finetune/train_sft.py \
		--config finetune/configs/sft_code_$(SIZE).yaml \
		--resume

# ── Stage 6: DPO ──────────────────────────────────────────────────────────────

prepare-dpo:
	@echo "==> Stage 6a: Prepare DPO data"
	python alignment/data/prepare_dpo.py

dpo:
	@echo "==> Stage 6b: DPO alignment ($(SIZE))"
	accelerate launch alignment/train_dpo.py \
		--config alignment/configs/dpo_$(SIZE).yaml

dpo-resume:
	accelerate launch alignment/train_dpo.py \
		--config alignment/configs/dpo_$(SIZE).yaml \
		--resume

# ── Stage 7: Evaluation ───────────────────────────────────────────────────────

eval:
	@echo "==> Stage 7: Evaluation ($(SIZE))"
	python eval/eval.py --model results/slm-$(SIZE)-dpo/final

# ── Stage 8: Export ───────────────────────────────────────────────────────────

export:
	@echo "==> Stage 8: Export to HuggingFace Hub ($(SIZE))"
	python export/export.py --model results/slm-$(SIZE)-dpo/final --size $(SIZE)

# ── S3 utilities ──────────────────────────────────────────────────────────────

s3-upload:
	python curator/scripts/upload_s3.py upload --src data/curated --dst curated

s3-download:
	python curator/scripts/upload_s3.py download --src curated --dst data/curated

s3-list:
	python curator/scripts/upload_s3.py list

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-uv:
	uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

install-conda:
	conda create -n slm python=3.10 -y && \
	conda activate slm && \
	pip install -r requirements.txt

install-kenlm:
	pip install https://github.com/kpu/kenlm/archive/master.zip

accelerate-config:
	accelerate config

# ── Clean ─────────────────────────────────────────────────────────────────────

clean-data:
	rm -rf data/raw data/filtered data/curated data/validated data/tokenized data/sft data/dpo

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
	@echo "Usage: make <target> [SIZE=125m|350m|1b]"
	@echo ""
	@echo "Pipeline stages:"
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
	@echo ""
	@echo "Utilities:"
	@echo "  install         Install dependencies (pip)"
	@echo "  install-uv      Install dependencies (uv)"
	@echo "  install-conda   Install dependencies (conda)"
	@echo "  install-kenlm   Install KenLM from source"
	@echo "  accelerate-config  Configure accelerate for multi-GPU"
	@echo "  s3-upload       Upload curated data to S3"
	@echo "  s3-download     Download curated data from S3"
	@echo "  s3-list         List S3 contents"
	@echo "  clean           Remove cache files"
	@echo "  clean-data      Remove all data directories"
	@echo "  clean-results   Remove all training results"
	@echo ""
	@echo "Examples:"
	@echo "  make all SIZE=125m"
	@echo "  make pretrain SIZE=350m"
	@echo "  make sft SIZE=1b"
	@echo "  make dpo SIZE=125m"
	@echo ""