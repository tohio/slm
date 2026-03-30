#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# infra/setup_gpu_instance.sh
# One-time setup for GPU cloud instance before any training runs.
#
# Run this immediately after spinning up a new GPU instance.
# Pulls all curation artifacts from S3 and sets up directories.
#
# Pulls by default:
#   - Tokenized mmap files      (curated/tokenized/*.bin, *.idx)
#   - Curated JSONL files       (curated/pii/*.jsonl)
#   - Tokenizer model           (tokenizer/)
#   - Quality classifier model  (models/quality_classifier.bin)
#
# NOTE: Python dependencies are handled by the Docker image (slm:latest).
#       This script only sets up the host environment and pulls data.
#       Run training via: make docker-shell-gpu → make pretrain
#
# Usage:
#   bash setup_gpu_instance.sh --bucket my-slm-bucket [--prefix slm/data]
#   bash setup_gpu_instance.sh --bucket my-slm-bucket --skip-jsonl --skip-classifier
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Load .env if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a && source "$ENV_FILE" && set +a
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
S3_BUCKET=""
S3_PREFIX="slm/data"
DATA_DIR="/data"
RESULTS_DIR="/results"
LOGS_DIR="/logs"
SKIP_JSONL=false
SKIP_BIN=false
SKIP_TOKENIZER=false
SKIP_CLASSIFIER=false

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket)          S3_BUCKET="$2";  shift 2 ;;
        --prefix)          S3_PREFIX="$2";  shift 2 ;;
        --skip-jsonl)      SKIP_JSONL=true;      shift ;;
        --skip-bin)        SKIP_BIN=true;        shift ;;
        --skip-tokenizer)  SKIP_TOKENIZER=true;  shift ;;
        --skip-classifier) SKIP_CLASSIFIER=true; shift ;;
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
    "$DATA_DIR/curated/stages/pii" \
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
    nvtop 2>/dev/null || true

# ── AWS CLI v2 ────────────────────────────────────────────────────────────────
if ! command -v aws &>/dev/null || [[ $(aws --version 2>&1) == *"aws-cli/1"* ]]; then
    log "Installing AWS CLI v2..."
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/awscliv2.zip /tmp/aws
fi
log "AWS CLI: $(aws --version)"

# ── Docker ────────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$(whoami)"
    log "Docker installed. You may need to log out and back in for group changes."
fi

# ── NGC Login ─────────────────────────────────────────────────────────────────
# Required to pull nvcr.io/nvidia/nemo:25.02 base image.
# NGC_API_KEY must be set in .env or as an environment variable.
if [[ -n "${NGC_API_KEY:-}" ]]; then
    log "Logging into NGC container registry..."
    echo "$NGC_API_KEY" | docker login nvcr.io \
        --username '$oauthtoken' \
        --password-stdin
    log "NGC login successful"
else
    log "WARNING: NGC_API_KEY not set — skipping NGC login"
    log "  Add NGC_API_KEY=your-key to .env before running make docker-build"
    log "  Without this, docker build will fail pulling nvcr.io/nvidia/nemo:25.02"
fi

# ── Pull dataset from S3 ──────────────────────────────────────────────────────
if [[ -z "$S3_BUCKET" ]]; then
    log "WARNING: No --bucket specified. Skipping S3 data pull."
    log "  Run manually: make setup-instance S3_BUCKET=your-bucket"
else
    S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}"
    log "Pulling dataset from $S3_BASE..."
    log ""
    log "  Skip flags:"
    log "    --skip-bin:         $SKIP_BIN"
    log "    --skip-jsonl:       $SKIP_JSONL"
    log "    --skip-tokenizer:   $SKIP_TOKENIZER"
    log "    --skip-classifier:  $SKIP_CLASSIFIER"
    log ""

    # Pull tokenized mmap files
    if [[ "$SKIP_BIN" == "true" ]]; then
        log "[SKIP] tokenized mmap files (--skip-bin)"
    else
        log "Pulling tokenized mmap files (.bin/.idx)..."
        aws s3 sync "${S3_BASE}/curated/tokenized/" "$DATA_DIR/curated/tokenized/" \
            --no-progress \
            --exclude "*" \
            --include "*.bin" \
            --include "*.idx"
        BIN_COUNT=$(find "$DATA_DIR/curated/tokenized" -name "*.bin" | wc -l)
        log "  ✓ $BIN_COUNT .bin files"
    fi

    # Pull curated JSONL files
    if [[ "$SKIP_JSONL" == "true" ]]; then
        log "[SKIP] curated JSONL files (--skip-jsonl)"
    else
        log "Pulling curated JSONL files..."
        aws s3 sync "${S3_BASE}/curated/pii/" "$DATA_DIR/curated/stages/pii/" \
            --no-progress \
            --exclude "*" \
            --include "*.jsonl"
        JSONL_COUNT=$(find "$DATA_DIR/curated/stages/pii" -name "*.jsonl" | wc -l)
        log "  ✓ $JSONL_COUNT JSONL files"
    fi

    # Pull tokenizer
    if [[ "$SKIP_TOKENIZER" == "true" ]]; then
        log "[SKIP] tokenizer (--skip-tokenizer)"
    else
        log "Pulling tokenizer..."
        aws s3 sync "${S3_BASE}/tokenizer/" "$DATA_DIR/tokenizer/" \
            --no-progress
        log "  ✓ $(ls $DATA_DIR/tokenizer/ 2>/dev/null || echo 'empty')"
    fi

    # Pull quality classifier
    if [[ "$SKIP_CLASSIFIER" == "true" ]]; then
        log "[SKIP] quality classifier (--skip-classifier)"
    else
        log "Pulling quality classifier..."
        aws s3 cp "${S3_BASE}/models/quality_classifier.bin" \
            "$DATA_DIR/models/quality_classifier.bin" \
            --no-progress 2>/dev/null \
            && log "  ✓ quality_classifier.bin" \
            || log "  WARNING: quality_classifier.bin not found in S3 — skipping"
    fi

    log ""
    log "Dataset sync complete"
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