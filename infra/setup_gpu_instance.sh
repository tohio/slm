#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# infra/setup_gpu_instance.sh
# One-time setup for GPU cloud instance before any training runs.
#
# Run this immediately after spinning up a new GPU instance.
# Pulls dataset from S3 and sets up directories.
#
# NOTE: Python dependencies are handled by the Docker image (slm:latest).
#       This script only sets up the host environment and pulls data.
#       Run training via: make docker-shell-gpu → make pretrain
#
# Usage:
#   bash setup_gpu_instance.sh --bucket my-slm-bucket [--prefix slm/data]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

S3_BUCKET=""
S3_PREFIX="slm/data"
DATA_DIR="/data"
RESULTS_DIR="/results"
LOGS_DIR="/logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket)  S3_BUCKET="$2";  shift 2 ;;
        --prefix)  S3_PREFIX="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== SLM GPU Instance Setup ==="

# ── GPU check ─────────────────────────────────────────────────────────────────
log "NVIDIA driver check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Directories ───────────────────────────────────────────────────────────────
log "Creating directory structure..."
sudo mkdir -p \
    "$DATA_DIR/curated/tokenized" \
    "$DATA_DIR/tokenizer" \
    "$DATA_DIR/models" \
    "$DATA_DIR/sft/chat" \
    "$DATA_DIR/sft/code" \
    "$DATA_DIR/dpo" \
    "$RESULTS_DIR" \
    "$LOGS_DIR"
sudo chown -R "$(whoami):$(whoami)" "$DATA_DIR" "$RESULTS_DIR" "$LOGS_DIR"

# ── System deps ───────────────────────────────────────────────────────────────
log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    wget \
    curl \
    unzip \
    htop \
    tmux \
    nvtop 2>/dev/null || true   # nvtop may not be available everywhere

# ── AWS CLI v2 ────────────────────────────────────────────────────────────────
# Install v2 standalone binary — avoids botocore conflict with boto3
# Skip if already installed
if ! command -v aws &>/dev/null || [[ $(aws --version 2>&1) == *"aws-cli/1"* ]]; then
    log "Installing AWS CLI v2..."
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/awscliv2.zip /tmp/aws
fi
log "AWS CLI: $(aws --version)"

# ── Docker ────────────────────────────────────────────────────────────────────
# Ensure Docker is installed for running the NeMo container
if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$(whoami)"
    log "Docker installed. You may need to log out and back in for group changes."
fi

# ── Pull dataset from S3 ──────────────────────────────────────────────────────
if [[ -n "$S3_BUCKET" ]]; then
    S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}"
    log "Pulling dataset from $S3_BASE..."

    # Pull tokenized mmap files for pretraining
    aws s3 sync "${S3_BASE}/curated/tokenized/" "$DATA_DIR/curated/tokenized/" \
        --no-progress \
        --exclude "*" \
        --include "*.bin" \
        --include "*.idx"

    # Pull tokenizer model
    aws s3 sync "${S3_BASE}/tokenizer/" "$DATA_DIR/tokenizer/" \
        --no-progress

    log "Dataset sync complete"
    log "  Tokenized files: $(find $DATA_DIR/curated/tokenized -name '*.bin' | wc -l) .bin files"
    log "  Tokenizer:       $(ls $DATA_DIR/tokenizer/ 2>/dev/null || echo 'empty')"
else
    log "WARNING: No --bucket specified. Skipping S3 data pull."
    log "  Run manually: aws s3 sync s3://your-bucket/${S3_PREFIX}/curated/tokenized/ $DATA_DIR/curated/tokenized/"
fi

# ── Build Docker image ────────────────────────────────────────────────────────
log "Building Docker image..."
if [[ -f "Dockerfile" ]]; then
    docker build -t slm:latest .
    log "Docker image built: slm:latest"
else
    log "WARNING: Dockerfile not found in current directory."
    log "  cd to repo root and run: make docker-build"
fi

log ""
log "=== Setup complete ==="
log "  Data directory:    $DATA_DIR"
log "  Results directory: $RESULTS_DIR"
log ""
log "Next steps:"
log "  1. Verify dataset:  ls $DATA_DIR/curated/tokenized/"
log "  2. Start GPU shell: make docker-shell-gpu"
log "  3. Run training:    make pretrain"