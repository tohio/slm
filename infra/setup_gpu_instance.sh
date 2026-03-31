#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# infra/setup_gpu_instance.sh
# One-time setup for a GPU cloud instance before any training runs.
#
# Run immediately after spinning up a new instance. Installs system
# dependencies, Python environment, and pulls curated data from S3.
#
# Training runs directly on the host — no Docker required.
#
# Usage:
#   bash infra/setup_gpu_instance.sh
#   bash infra/setup_gpu_instance.sh --skip-data
#   bash infra/setup_gpu_instance.sh --data-dir /data
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a && source "$ENV_FILE" && set +a
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
PYTHON="${PYTHON:-python3.10}"
SKIP_DATA=false
SKIP_PYTHON=false

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data)    SKIP_DATA=true;   shift ;;
        --skip-python)  SKIP_PYTHON=true; shift ;;
        --data-dir)     DATA_DIR="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== SLM GPU Instance Setup ==="
log "Repo:     $REPO_DIR"
log "Data:     $DATA_DIR"
log "Results:  $RESULTS_DIR"

# ── GPU check ─────────────────────────────────────────────────────────────────
log "GPU check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Directories ───────────────────────────────────────────────────────────────
log "Creating directory structure..."
mkdir -p \
    "$DATA_DIR/raw/wikipedia" \
    "$DATA_DIR/raw/code" \
    "$DATA_DIR/raw/common_crawl" \
    "$DATA_DIR/filtered" \
    "$DATA_DIR/curated" \
    "$DATA_DIR/validated" \
    "$DATA_DIR/tokenized" \
    "$DATA_DIR/tokenizer" \
    "$DATA_DIR/sft/chat" \
    "$DATA_DIR/sft/code" \
    "$DATA_DIR/dpo" \
    "$DATA_DIR/models" \
    "$RESULTS_DIR" \
    "$REPO_DIR/logs"
log "  ✓ Directories created"

# ── System dependencies ───────────────────────────────────────────────────────
log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    wget \
    curl \
    unzip \
    htop \
    tmux \
    nvtop \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip 2>/dev/null || true
log "  ✓ System dependencies installed"

# ── AWS CLI ───────────────────────────────────────────────────────────────────
if ! command -v aws &>/dev/null || [[ $(aws --version 2>&1) == *"aws-cli/1"* ]]; then
    log "Installing AWS CLI v2..."
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/awscliv2.zip /tmp/aws
fi
log "  ✓ AWS CLI: $(aws --version 2>&1 | head -1)"

# ── Python environment ────────────────────────────────────────────────────────
if [[ "$SKIP_PYTHON" == "true" ]]; then
    log "[SKIP] Python environment (--skip-python)"
else
    log "Setting up Python environment..."
    cd "$REPO_DIR"

    if [[ ! -d ".venv" ]]; then
        $PYTHON -m venv .venv
        log "  ✓ Virtual environment created"
    fi

    source .venv/bin/activate

    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    log "  ✓ Python dependencies installed"

    # KenLM — required for validation perplexity filtering
    if ! python -c "import kenlm" 2>/dev/null; then
        log "Installing KenLM from source..."
        pip install https://github.com/kpu/kenlm/archive/master.zip --quiet
        log "  ✓ KenLM installed"
    fi

    # Configure accelerate for multi-GPU if not already configured
    if [[ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]]; then
        log "Configuring accelerate..."
        accelerate config default
        log "  ✓ accelerate configured"
        log "  Edit ~/.cache/huggingface/accelerate/default_config.yaml for multi-GPU"
    fi
fi

# ── Pull data from S3 ─────────────────────────────────────────────────────────
if [[ "$SKIP_DATA" == "true" ]]; then
    log "[SKIP] S3 data pull (--skip-data)"
elif [[ -z "${S3_BUCKET:-}" ]]; then
    log "WARNING: S3_BUCKET not set in .env — skipping data pull"
    log "  Set S3_BUCKET in .env and re-run, or pull manually: make s3-download"
else
    S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX:-slm/data}"
    log "Pulling curated data from $S3_BASE..."

    log "  Pulling tokenized dataset..."
    aws s3 sync "${S3_BASE}/tokenized/" "$DATA_DIR/tokenized/" \
        --no-progress \
        --exclude "*" \
        --include "*.bin" \
        --include "*.json"
    BIN_COUNT=$(find "$DATA_DIR/tokenized" -name "*.bin" 2>/dev/null | wc -l)
    log "  ✓ $BIN_COUNT .bin files"

    log "  Pulling tokenizer..."
    aws s3 sync "${S3_BASE}/tokenizer/" "$DATA_DIR/tokenizer/" --no-progress
    log "  ✓ Tokenizer"

    log "  Pulling curated JSONL..."
    aws s3 sync "${S3_BASE}/curated/" "$DATA_DIR/curated/" \
        --no-progress \
        --exclude "*" \
        --include "*.jsonl" \
        --include "*.json"
    JSONL_COUNT=$(find "$DATA_DIR/curated" -name "*.jsonl" 2>/dev/null | wc -l)
    log "  ✓ $JSONL_COUNT JSONL files"

    log "  Pulling validated JSONL..."
    aws s3 sync "${S3_BASE}/validated/" "$DATA_DIR/validated/" \
        --no-progress \
        --exclude "*" \
        --include "*.jsonl"
    log "  ✓ Validated data"

    log "Data pull complete"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
log ""
log "=== Setup complete ==="
log ""
log "Activate environment:"
log "  source .venv/bin/activate"
log ""
log "GPU monitoring:"
log "  watch -n 2 nvidia-smi"
log "  nvtop"
log ""
log "Next steps:"
log "  make curate   SIZE=125m"
log "  make pretrain SIZE=125m GPUS=4"