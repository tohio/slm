# ─────────────────────────────────────────────────────────────────────────────
# SLM Project Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   make help
#   make docker-build              Build NeMo Docker image
#   make docker-curate             Run data curation in container
#   make docker-shell-cpu          Interactive CPU container (data curation)
#   make docker-shell-gpu          Interactive GPU container (training)
#   make all
#   make pretrain GPUS=4 CONFIG=pretrain/configs/gpt_350m.yaml
# ─────────────────────────────────────────────────────────────────────────────

-include .env
export

GPUS          ?= $(shell python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
CONFIG        ?= pretrain/configs/gpt_125m.yaml
S3_BUCKET     ?=
S3_PREFIX     ?= slm/data
N_WARC_FILES  ?= 20
DATA_DIR      ?= /data
RESULTS_DIR   ?= /results
LOGS_DIR      ?= /logs
PYTHON        ?= python3
DOCKER_IMAGE  ?= slm:latest

PRETRAIN_CKPT = $(shell find $(RESULTS_DIR)/slm_gpt_125m/checkpoints -name "*.nemo" 2>/dev/null | sort | tail -1)
SFT_CHAT_CKPT = $(shell find $(RESULTS_DIR)/slm_sft_chat/checkpoints -name "*.nemo" 2>/dev/null | sort | tail -1)
SFT_CODE_CKPT = $(shell find $(RESULTS_DIR)/slm_sft_code/checkpoints -name "*.nemo" 2>/dev/null | sort | tail -1)
DPO_CKPT      = $(shell find $(RESULTS_DIR)/slm_dpo/checkpoints      -name "*.nemo" 2>/dev/null | sort | tail -1)

.DEFAULT_GOAL := help
.PHONY: help all setup setup-instance init-dirs \
        docker-build docker-curate docker-shell-cpu docker-shell-gpu \
        download-data download-models \
        curate curate-resume tokenizer tokenize upload-data \
        train-quality-classifier curate-full \
        prepare-sft-data prepare-dpo-data \
        pretrain sft dpo \
        eval-pretrain eval-sft eval-dpo \
        convert-hf clean clean-all \
        _check-pretrain-ckpt _check-sft-ckpt _check-dpo-ckpt \
        _check-s3-bucket _check-data-dirs

help:
	@echo ""
	@echo "  SLM — Small Language Model Pipeline"
	@echo "  ════════════════════════════════════"
	@echo ""
	@echo "  FIRST TIME SETUP"
	@echo "    make init-dirs              Create host directories (/data, /results, /logs)"
	@echo "    make docker-build           Build NeMo Docker image"
	@echo "    make download-models        Download fasttext models (lid.176.bin)"
	@echo ""
	@echo "  DOCKER  (recommended)"
	@echo "    make docker-shell-cpu       Start interactive CPU container (data curation)"
	@echo "    make docker-shell-gpu       Start interactive GPU container (training)"
	@echo "    make docker-curate          Run full data curation in container"
	@echo ""
	@echo "  SETUP"
	@echo "    make setup                  Install dependencies (local only)"
	@echo "    make setup-instance         GPU instance first-time setup"
	@echo ""
	@echo "  DATA  (run in container or directly)"
	@echo "    make download-data          Download Common Crawl WARCs"
	@echo "    make download-models        Download fasttext language ID model"
	@echo "    make curate                 Run full Curator pipeline"
	@echo "    make curate-resume          Resume curator from a specific stage"
	@echo "    make tokenizer              Train custom BPE tokenizer (runs in Docker)"
	@echo "    make tokenize               Convert JSONL → .bin/.idx mmap (runs in Docker)"
	@echo "    make upload-data            Upload dataset to S3"
	@echo ""
	@echo "  TRAINING  (all run in Docker)"
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
	@echo "    make upload-data                  # reads S3_BUCKET from .env"
	@echo "    make train-quality-classifier   Train fastText quality classifier"
	@echo "    make curate-full                Full two-pass curation (automated)"
	@echo "    make docker-build DOCKER_IMAGE=slm:latest"
	@echo ""

all: curate tokenizer tokenize upload-data prepare-sft-data prepare-dpo-data pretrain sft dpo eval-dpo
	@echo "✓ Full pipeline complete. Final model: $(DPO_CKPT)"

# ── First-time host setup ─────────────────────────────────────────────────────
init-dirs:
	@echo "Creating host directories..."
	sudo mkdir -p $(DATA_DIR) $(RESULTS_DIR) $(LOGS_DIR)
	sudo chown -R $(shell whoami):$(shell whoami) $(DATA_DIR) $(RESULTS_DIR) $(LOGS_DIR)
	mkdir -p $(DATA_DIR)/models $(DATA_DIR)/raw $(DATA_DIR)/curated $(DATA_DIR)/logs
	@echo "✓ Directories created:"
	@echo "    $(DATA_DIR)       — raw data, curated output, models"
	@echo "    $(RESULTS_DIR)    — training checkpoints"
	@echo "    $(LOGS_DIR)       — curator and training logs"

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "NOTE: Install PyTorch first:"
	@echo "  pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121"
	pip install -r requirements.txt

setup-instance: _check-s3-bucket
	bash infra/setup_gpu_instance.sh --bucket $(S3_BUCKET) --prefix $(S3_PREFIX)

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	@echo "Building NeMo Docker image: $(DOCKER_IMAGE)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "✓ Docker image built: $(DOCKER_IMAGE)"

docker-shell-cpu: _check-data-dirs
	@echo "Starting CPU container for data curation..."
	docker run -it --rm \
		--shm-size=8g \
		-p 8787:8787 \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(LOGS_DIR):$(LOGS_DIR) \
		$(DOCKER_IMAGE) /bin/bash

docker-shell-gpu: _check-data-dirs
	@echo "Starting GPU container for training..."
	docker run --gpus all -it --rm \
		--shm-size=8g \
		-p 8787:8787 \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(RESULTS_DIR):$(RESULTS_DIR) \
		-v $(LOGS_DIR):$(LOGS_DIR) \
		$(DOCKER_IMAGE) /bin/bash

docker-curate: _check-data-dirs
	@echo "Running data curation in Docker..."
	docker run -it --rm \
		--shm-size=8g \
		-p 8787:8787 \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(LOGS_DIR):$(LOGS_DIR) \
		$(DOCKER_IMAGE) bash -c "cd /workspace/slm && make curate"

# ── Quality Classifier ───────────────────────────────────────────────────────
train-quality-classifier:
	@echo "Training quality classifier on pass 1 output..."
	bash curator/scripts/train_quality_classifier.sh \
		--input-dir $(DATA_DIR)/curated/stages/pii \
		--heuristic-dir $(DATA_DIR)/curated/stages/heuristic_filter \
		--output-model $(DATA_DIR)/models/quality_classifier.bin \
		--n-samples 50000

# ── Full two-pass curation ────────────────────────────────────────────────────
# Runs both passes end-to-end without manual intervention.
# Each step is checkpointed — re-running skips already completed steps.
# Pass 1: extract → filter → dedup → pii + tokenizer training
# Pass 2: quality_filter + tokenize (auto-detected by pipeline.py)
curate-full:
	@echo "=== curate-full: two-pass curation pipeline ==="
	@if [ -f $(DATA_DIR)/curated/stages/pii/.complete ]; then \
		echo "[SKIP] curate — pass 1 already complete"; \
	else \
		echo "[RUN] curate — pass 1..."; \
		$(MAKE) curate; \
	fi
	@if [ -f $(DATA_DIR)/tokenizer/.complete ]; then \
		echo "[SKIP] tokenizer — already trained"; \
	else \
		echo "[RUN] tokenizer..."; \
		$(MAKE) tokenizer; \
	fi
	@if [ -f $(DATA_DIR)/models/.complete ]; then \
		echo "[SKIP] train-quality-classifier — already trained"; \
	else \
		echo "[RUN] train-quality-classifier..."; \
		$(MAKE) train-quality-classifier; \
	fi
	@if [ -f $(DATA_DIR)/curated/stages/tokenize/.complete ]; then \
		echo "[SKIP] pass 2 — already complete"; \
	else \
		echo "[RUN] pass 2 — quality_filter + tokenize..."; \
		$(MAKE) tokenize; \
	fi
	@echo "✓ curate-full complete"

# ── Data ──────────────────────────────────────────────────────────────────────
download-data: _check-data-dirs
	bash curator/scripts/download_cc.sh \
		--n-files $(N_WARC_FILES) \
		--output-dir $(DATA_DIR)/raw/common_crawl

download-models: _check-data-dirs
	@echo "Downloading fasttext language ID model..."
	mkdir -p $(DATA_DIR)/models
	wget -q --show-progress \
		-O $(DATA_DIR)/models/lid.176.bin \
		https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
	@echo "✓ $(DATA_DIR)/models/lid.176.bin"
	@echo ""
	@echo "NOTE: quality_classifier.bin must be trained after first curation pass."
	@echo "      Run: make train-quality-classifier"

curate: _check-data-dirs
	$(PYTHON) curator/pipelines/pipeline.py \
		--config curator/configs/curator.yaml

curate-resume: _check-data-dirs
	@read -p "Resume from stage: " stage; \
	$(PYTHON) curator/pipelines/pipeline.py \
		--config curator/configs/curator.yaml \
		--start-stage $$stage

# Train custom BPE tokenizer — runs inside Docker (sentencepiece not on host)
tokenizer: _check-data-dirs
	@echo "Training BPE tokenizer in Docker..."
	docker run --rm \
		--shm-size=8g \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		$(DOCKER_IMAGE) bash -c "cd /workspace/slm && \
			python3 tokenizer/train_tokenizer.py \
				--config tokenizer/configs/tokenizer.yaml \
				--input-dir $(DATA_DIR)/curated/stages/pii \
				--output-dir $(DATA_DIR)/tokenizer"

# Convert curated JSONL → NeMo mmap .bin/.idx — runs inside Docker
tokenize: _check-data-dirs
	@echo "Tokenizing JSONL → mmap (.bin/.idx) in Docker..."
	docker run --rm \
		--shm-size=8g \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		$(DOCKER_IMAGE) bash -c "cd /workspace/slm && \
			python3 curator/pipelines/pipeline.py \
				--config curator/configs/curator.yaml \
				--start-stage tokenize"

upload-data: _check-s3-bucket
	bash curator/scripts/upload_s3.sh \
		--bucket $(S3_BUCKET) \
		--prefix $(S3_PREFIX)

prepare-sft-data:
	$(PYTHON) finetune/data/prepare_sft.py --stage both

prepare-dpo-data:
	$(PYTHON) alignment/data/prepare_dpo.py \
		--output-dir $(DATA_DIR)/dpo

# ── Training — all run inside Docker ─────────────────────────────────────────
pretrain:
	@echo "Launching pre-training in Docker..."
	docker run --gpus all --rm \
		--shm-size=8g \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(RESULTS_DIR):$(RESULTS_DIR) \
		-v $(LOGS_DIR):$(LOGS_DIR) \
		$(DOCKER_IMAGE) bash -c "cd /workspace/slm && \
			bash pretrain/scripts/train.sh \
				--config $(CONFIG) \
				--gpus $(GPUS)"

sft: _check-pretrain-ckpt
	@echo "Launching SFT in Docker..."
	docker run --gpus all --rm \
		--shm-size=8g \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(RESULTS_DIR):$(RESULTS_DIR) \
		-v $(LOGS_DIR):$(LOGS_DIR) \
		$(DOCKER_IMAGE) bash -c "cd /workspace/slm && \
			bash finetune/scripts/train_sft.sh \
				--gpus $(GPUS) \
				--pretrain-ckpt $(PRETRAIN_CKPT)"

dpo: _check-sft-ckpt
	@echo "Launching DPO in Docker..."
	docker run --gpus all --rm \
		--shm-size=8g \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(RESULTS_DIR):$(RESULTS_DIR) \
		-v $(LOGS_DIR):$(LOGS_DIR) \
		$(DOCKER_IMAGE) bash -c "cd /workspace/slm && \
			bash alignment/scripts/train_dpo.sh \
				--gpus $(GPUS) \
				--sft-ckpt $(SFT_CODE_CKPT)"

# ── Evaluation ────────────────────────────────────────────────────────────────
eval-pretrain: _check-pretrain-ckpt
	$(PYTHON) eval/run_eval.py \
		--stage pretrain \
		--checkpoint $(PRETRAIN_CKPT) \
		--val-data $(DATA_DIR)/pretrain

eval-sft: _check-sft-ckpt
	$(PYTHON) eval/run_eval.py \
		--stage sft \
		--checkpoint $(SFT_CODE_CKPT) \
		--val-data $(DATA_DIR)/pretrain

eval-dpo: _check-dpo-ckpt _check-sft-ckpt
	$(PYTHON) eval/run_eval.py \
		--stage dpo \
		--checkpoint $(DPO_CKPT) \
		--ref-checkpoint $(SFT_CODE_CKPT) \
		--val-data $(DATA_DIR)/pretrain

# ── Inference ─────────────────────────────────────────────────────────────────
inference: _check-dpo-ckpt
	docker run --gpus all -it --rm \
		--shm-size=8g \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(RESULTS_DIR):$(RESULTS_DIR) \
		$(DOCKER_IMAGE) \
		python /workspace/slm/inference.py \
		--checkpoint $(DPO_CKPT)

PROMPT ?= "Explain recursion to a 10-year-old."
inference-compare: _check-dpo-ckpt _check-sft-ckpt
	docker run --gpus all -it --rm \
		--shm-size=8g \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(RESULTS_DIR):$(RESULTS_DIR) \
		$(DOCKER_IMAGE) \
		python /workspace/slm/inference.py \
		--checkpoint $(DPO_CKPT) \
		--compare $(SFT_CODE_CKPT) \
		--prompt $(PROMPT)

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

_check-s3-bucket:
	@if [ -z "$(S3_BUCKET)" ]; then \
		echo "ERROR: S3_BUCKET not set."; \
		echo "  Option 1 (recommended): add S3_BUCKET=your-bucket to .env"; \
		echo "  Option 2: pass on command line: make $(@:_check-%=%) S3_BUCKET=your-bucket"; \
		exit 1; fi

_check-data-dirs:
	@if [ ! -d "$(DATA_DIR)" ] || [ ! -d "$(RESULTS_DIR)" ] || [ ! -d "$(LOGS_DIR)" ]; then \
		echo "ERROR: Host directories not found. Run: make init-dirs"; exit 1; fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	sudo find $(DATA_DIR)/curated/stages -name ".complete" -delete 2>/dev/null || true
	sudo find $(DATA_DIR)/tokenizer -name ".complete" -delete 2>/dev/null || true
	sudo find $(DATA_DIR)/models -name ".complete" -delete 2>/dev/null || true
	@echo "✓ Stage markers cleared (checkpoints and data preserved)"

clean-all:
	@read -p "Delete ALL results and curated data? [y/N] " c; \
	if [ "$$c" = "y" ]; then rm -rf $(RESULTS_DIR)/* $(DATA_DIR)/curated/*; fi