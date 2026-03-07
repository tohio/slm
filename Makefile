# ─────────────────────────────────────────────────────────────────────────────
# SLM Project Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   make help
#   make all
#   make pretrain GPUS=4 CONFIG=pretrain/configs/gpt_350m.yaml
#   make curate N_WARC_FILES=50
#   make upload-data S3_BUCKET=my-bucket
# ─────────────────────────────────────────────────────────────────────────────

GPUS          ?= $(shell python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
CONFIG        ?= pretrain/configs/gpt_125m.yaml
S3_BUCKET     ?=
S3_PREFIX     ?= slm/data
N_WARC_FILES  ?= 20
DATA_DIR      ?= /data
RESULTS_DIR   ?= /results
PYTHON        ?= python3

PRETRAIN_CKPT = $(shell find $(RESULTS_DIR)/slm_gpt_125m/checkpoints -name "*.nemo" 2>/dev/null | sort | tail -1)
SFT_CHAT_CKPT = $(shell find $(RESULTS_DIR)/slm_sft_chat/checkpoints -name "*.nemo" 2>/dev/null | sort | tail -1)
SFT_CODE_CKPT = $(shell find $(RESULTS_DIR)/slm_sft_code/checkpoints -name "*.nemo" 2>/dev/null | sort | tail -1)
DPO_CKPT      = $(shell find $(RESULTS_DIR)/slm_dpo/checkpoints      -name "*.nemo" 2>/dev/null | sort | tail -1)

.DEFAULT_GOAL := help
.PHONY: help all setup setup-instance \
        download-data curate tokenizer upload-data \
        prepare-sft-data prepare-dpo-data \
        pretrain sft dpo \
        eval-pretrain eval-sft eval-dpo \
        convert-hf clean clean-all \
        _check-pretrain-ckpt _check-sft-ckpt _check-dpo-ckpt

help:
	@echo ""
	@echo "  SLM — Small Language Model Pipeline"
	@echo "  ════════════════════════════════════"
	@echo ""
	@echo "  SETUP"
	@echo "    make setup                  Install dependencies"
	@echo "    make setup-instance         GPU instance first-time setup"
	@echo ""
	@echo "  DATA  (run on spot/local)"
	@echo "    make download-data          Download Common Crawl WARCs"
	@echo "    make curate                 Run full Curator pipeline"
	@echo "    make tokenizer              Train custom BPE tokenizer"
	@echo "    make upload-data            Upload dataset to S3"
	@echo ""
	@echo "  TRAINING  (run on GPU instance)"
	@echo "    make prepare-sft-data       Download & format SFT datasets"
	@echo "    make prepare-dpo-data       Download & format DPO preference data"
	@echo "    make pretrain               Pre-train from scratch"
	@echo "    make sft                    Supervised fine-tuning (chat → code)"
	@echo "    make dpo                    DPO alignment"
	@echo "    make all                    Full pipeline end to end"
	@echo ""
	@echo "  EVALUATION"
	@echo "    make eval-pretrain"
	@echo "    make eval-sft"
	@echo "    make eval-dpo"
	@echo ""
	@echo "  EXPORT"
	@echo "    make convert-hf             Export final model to HuggingFace format"
	@echo ""
	@echo "  OVERRIDES"
	@echo "    make pretrain GPUS=4 CONFIG=pretrain/configs/gpt_350m.yaml"
	@echo "    make curate N_WARC_FILES=50"
	@echo "    make upload-data S3_BUCKET=my-bucket"
	@echo ""

all: curate tokenizer upload-data prepare-sft-data prepare-dpo-data pretrain sft dpo eval-dpo
	@echo "✓ Full pipeline complete. Final model: $(DPO_CKPT)"

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "NOTE: Install PyTorch first:"
	@echo "  pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121"
	pip install -r requirements.txt

setup-instance:
	@if [ -z "$(S3_BUCKET)" ]; then echo "ERROR: S3_BUCKET required"; exit 1; fi
	bash infra/setup_gpu_instance.sh --bucket $(S3_BUCKET) --prefix $(S3_PREFIX)

# ── Data ──────────────────────────────────────────────────────────────────────
download-data:
	bash curator/scripts/download_cc.sh --n-files $(N_WARC_FILES) --output-dir $(DATA_DIR)/raw/common_crawl

curate:
	$(PYTHON) curator/pipelines/pipeline.py --config curator/configs/curator.yaml

curate-resume:
	@read -p "Resume from stage: " stage; \
	$(PYTHON) curator/pipelines/pipeline.py --config curator/configs/curator.yaml --start-stage $$stage

tokenizer:
	$(PYTHON) tokenizer/train_tokenizer.py \
		--config tokenizer/configs/tokenizer.yaml \
		--input-dir $(DATA_DIR)/curated/stages/pii \
		--output-dir $(DATA_DIR)/tokenizer

upload-data:
	@if [ -z "$(S3_BUCKET)" ]; then echo "ERROR: S3_BUCKET required"; exit 1; fi
	bash curator/scripts/upload_s3.sh --bucket $(S3_BUCKET) --prefix $(S3_PREFIX)

prepare-sft-data:
	$(PYTHON) finetune/data/prepare_sft.py --stage both

prepare-dpo-data:
	$(PYTHON) alignment/data/prepare_dpo.py --output-dir $(DATA_DIR)/dpo

# ── Training ──────────────────────────────────────────────────────────────────
pretrain:
	bash pretrain/scripts/train.sh --config $(CONFIG) --gpus $(GPUS)

sft: _check-pretrain-ckpt
	bash finetune/scripts/train_sft.sh --gpus $(GPUS) --pretrain-ckpt $(PRETRAIN_CKPT)

dpo: _check-sft-ckpt
	bash alignment/scripts/train_dpo.sh --gpus $(GPUS) --sft-ckpt $(SFT_CODE_CKPT)

# ── Evaluation ────────────────────────────────────────────────────────────────
eval-pretrain: _check-pretrain-ckpt
	$(PYTHON) eval/run_eval.py --stage pretrain --checkpoint $(PRETRAIN_CKPT) --val-data $(DATA_DIR)/pretrain

eval-sft: _check-sft-ckpt
	$(PYTHON) eval/run_eval.py --stage sft --checkpoint $(SFT_CODE_CKPT) --val-data $(DATA_DIR)/pretrain

eval-dpo: _check-dpo-ckpt _check-sft-ckpt
	$(PYTHON) eval/run_eval.py --stage dpo \
		--checkpoint $(DPO_CKPT) \
		--ref-checkpoint $(SFT_CODE_CKPT) \
		--val-data $(DATA_DIR)/pretrain

# ── Export ────────────────────────────────────────────────────────────────────
convert-hf: _check-dpo-ckpt
	bash pretrain/scripts/convert_ckpt.sh \
		--direction nemo_to_hf \
		--input $(DPO_CKPT) \
		--output $(RESULTS_DIR)/slm_final_hf

# ── Guards ────────────────────────────────────────────────────────────────────
_check-pretrain-ckpt:
	@if [ -z "$(PRETRAIN_CKPT)" ]; then \
		echo "ERROR: No pretrain checkpoint found. Run: make pretrain"; exit 1; fi

_check-sft-ckpt:
	@if [ -z "$(SFT_CODE_CKPT)" ]; then \
		echo "ERROR: No SFT checkpoint found. Run: make sft"; exit 1; fi

_check-dpo-ckpt:
	@if [ -z "$(DPO_CKPT)" ]; then \
		echo "ERROR: No DPO checkpoint found. Run: make dpo"; exit 1; fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find $(DATA_DIR)/curated/stages -name ".complete" -delete 2>/dev/null || true
	@echo "✓ Stage markers cleared (checkpoints and data preserved)"

clean-all:
	@read -p "Delete ALL results and curated data? [y/N] " c; \
	if [ "$$c" = "y" ]; then rm -rf $(RESULTS_DIR)/* $(DATA_DIR)/curated/*; fi
