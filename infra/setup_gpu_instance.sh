#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# infra/setup_gpu_instance.sh
# One-time setup for GPU cloud instance before any training runs.
#
# Run this immediately after spinning up a new GPU instance.
# Installs NeMo, NeMo Aligner, pulls dataset from S3, sets up directories.
#
# Usage:
#   bash setup_gpu_instance.sh --bucket my-slm-bucket [--prefix slm/data]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

S3_BUCKET=""
S3_PREFIX="slm/data"
DATA_DIR="/data"
RESULTS_DIR="/results"
NEMO_VERSION="2.0.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket)  S3_BUCKET="$2";  shift 2 ;;
        --prefix)  S3_PREFIX="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== SLM GPU Instance Setup ==="
log "NVIDIA driver check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Directories ───────────────────────────────────────────────────────────────
log "Creating directory structure..."
mkdir -p \
    "$DATA_DIR/pretrain" \
    "$DATA_DIR/tokenizer" \
    "$DATA_DIR/sft/chat" \
    "$DATA_DIR/sft/code" \
    "$DATA_DIR/dpo" \
    "$RESULTS_DIR" \
    /logs

# ── System deps ───────────────────────────────────────────────────────────────
log "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    git \
    wget \
    curl \
    unzip \
    awscli \
    htop \
    tmux \
    nvtop 2>/dev/null || true   # nvtop may not be available everywhere

# ── Python environment ────────────────────────────────────────────────────────
log "Installing Python packages..."

pip install --upgrade pip --quiet

# NeMo full install (includes Megatron-Core)
pip install \
    "nemo_toolkit[all]==${NEMO_VERSION}" \
    --quiet

# NeMo Aligner (SFT + DPO + PPO)
pip install \
    nemo-aligner \
    --quiet

# Additional utilities
pip install \
    sentencepiece \
    wandb \
    matplotlib \
    pandas \
    --quiet

log "Package versions:"
python -c "import nemo; print(f'  NeMo:         {nemo.__version__}')"
python -c "import torch; print(f'  PyTorch:      {torch.__version__}')"
python -c "import torch; print(f'  CUDA:         {torch.version.cuda}')"
python -c "import torch; print(f'  GPU count:    {torch.cuda.device_count()}')"

# ── Pull dataset from S3 ──────────────────────────────────────────────────────
if [[ -n "$S3_BUCKET" ]]; then
    S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}"
    log "Pulling dataset from $S3_BASE..."

    aws s3 sync "${S3_BASE}/tokenized/" "$DATA_DIR/pretrain/" \
        --no-progress \
        --exclude "*" \
        --include "*.bin" \
        --include "*.idx"

    aws s3 sync "${S3_BASE}/tokenizer/" "$DATA_DIR/tokenizer/" \
        --no-progress

    log "Dataset sync complete"
    log "  Pretrain files: $(find $DATA_DIR/pretrain -name '*.bin' | wc -l) .bin files"
    log "  Tokenizer:      $(ls $DATA_DIR/tokenizer/)"
else
    log "WARNING: No --bucket specified. Skipping S3 data pull."
    log "  Manually sync your dataset before training."
fi

# ── Verify GPU access ─────────────────────────────────────────────────────────
log "Verifying PyTorch CUDA access..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} | {props.total_memory // 1024**3}GB VRAM')
"

log ""
log "=== Setup complete ==="
log "  Data directory:    $DATA_DIR"
log "  Results directory: $RESULTS_DIR"
log ""
log "Next steps:"
log "  1. Verify dataset: ls $DATA_DIR/pretrain/"
log "  2. Start training: bash pretrain/scripts/train.sh --config pretrain/configs/gpt_125m.yaml"
