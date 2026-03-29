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
        curate curate-resume tokenizer upload-data \
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
	@echo "    make tokenizer              Train custom BPE tokenizer"
	@echo "    make upload-data            Upload dataset to S3"
	@echo ""
	@echo "  TRAINING  (run in GPU container)"
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
	@echo "    make docker-build DOCKER_IMAGE=slm:latest"
	@echo ""

all: curate tokenizer upload-data prepare-sft-data prepare-dpo-data pretrain sft dpo eval-dpo
	@echo "✓ Full pipeline complete. Final model: $(DPO_CKPT)"

# ── First-time host setup ─────────────────────────────────────────────────────
# Creates the host directories that Docker bind-mounts at runtime.
# Must be run once on any new instance before docker-shell-* or docker-curate.
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
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(LOGS_DIR):$(LOGS_DIR) \
		$(DOCKER_IMAGE) /bin/bash

docker-shell-gpu: _check-data-dirs
	@echo "Starting GPU container for training..."
	docker run --gpus all -it --rm \
		--shm-size=8g \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(RESULTS_DIR):$(RESULTS_DIR) \
		-v $(LOGS_DIR):$(LOGS_DIR) \
		$(DOCKER_IMAGE) /bin/bash

docker-curate: _check-data-dirs
	@echo "Running data curation in Docker..."
	docker run -it --rm \
		--shm-size=8g \
		-v $$(pwd):/workspace/slm \
		-v $(DATA_DIR):$(DATA_DIR) \
		-v $(LOGS_DIR):$(LOGS_DIR) \
		$(DOCKER_IMAGE) bash -c "cd /workspace/slm && make curate tokenizer"

# ── Data ──────────────────────────────────────────────────────────────────────
download-data: _check-data-dirs
	bash curator/scripts/download_cc.sh \
		--n-files $(N_WARC_FILES) \
		--output-dir $(DATA_DIR)/raw/common_crawl

# Downloads models required by curator.yaml:
#   lid.176.bin          : fasttext language identification model (Meta, public)
#   quality_classifier.bin is trained from your own data — see docs
download-models: _check-data-dirs
	@echo "Downloading fasttext language ID model..."
	mkdir -p $(DATA_DIR)/models
	wget -q --show-progress \
		-O $(DATA_DIR)/models/lid.176.bin \
		https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
	@echo "✓ $(DATA_DIR)/models/lid.176.bin"
	@echo ""
	@echo "NOTE: quality_classifier.bin must be trained after first curation pass."
	@echo "      Set quality_filter.enabled: false in curator/configs/curator.yaml"
	@echo "      until you have trained a classifier on your curated data."

curate: _check-data-dirs
	$(PYTHON) curator/pipelines/pipeline.py \
		--config curator/configs/curator.yaml

curate-resume: _check-data-dirs
	@read -p "Resume from stage: " stage; \
	$(PYTHON) curator/pipelines/pipeline.py \
		--config curator/configs/curator.yaml \
		--start-stage $$stage

tokenizer: _check-data-dirs
	$(PYTHON) tokenizer/train_tokenizer.py \
		--config tokenizer/configs/tokenizer.yaml \
		--input-dir $(DATA_DIR)/curated/stages/pii \
		--output-dir $(DATA_DIR)/tokenizer

upload-data: _check-s3-bucket
	bash curator/scripts/upload_s3.sh \
		--bucket $(S3_BUCKET) \
		--prefix $(S3_PREFIX)

prepare-sft-data:
	$(PYTHON) finetune/data/prepare_sft.py --stage both

prepare-dpo-data:
	$(PYTHON) alignment/data/prepare_dpo.py \
		--output-dir $(DATA_DIR)/dpo

# ── Training ──────────────────────────────────────────────────────────────────
pretrain:
	bash pretrain/scripts/train.sh --config $(CONFIG) --gpus $(GPUS)

sft: _check-pretrain-ckpt
	bash finetune/scripts/train_sft.sh \
		--gpus $(GPUS) \
		--pretrain-ckpt $(PRETRAIN_CKPT)

dpo: _check-sft-ckpt
	bash alignment/scripts/train_dpo.sh \
		--gpus $(GPUS) \
		--sft-ckpt $(SFT_CODE_CKPT)

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
		echo "ERROR: S3_BUCKET required. Pass: make $(@:_check-%=%) S3_BUCKET=my-bucket"; exit 1; fi

_check-data-dirs:
	@if [ ! -d "$(DATA_DIR)" ] || [ ! -d "$(RESULTS_DIR)" ] || [ ! -d "$(LOGS_DIR)" ]; then \
		echo "ERROR: Host directories not found. Run: make init-dirs"; exit 1; fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	# Stage files are written by root inside Docker — sudo required
	sudo find $(DATA_DIR)/curated/stages -name ".complete" -delete 2>/dev/null || true
	@echo "✓ Stage markers cleared (checkpoints and data preserved)"

clean-all:
	@read -p "Delete ALL results and curated data? [y/N] " c; \
	if [ "$$c" = "y" ]; then rm -rf $(RESULTS_DIR)/* $(DATA_DIR)/curated/*; fi