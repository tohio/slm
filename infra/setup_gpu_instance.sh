#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# infra/setup_gpu_instance.sh
# Setup script for a GPU training instance.
#
# Safe to re-run after a preemptible VM restart — idempotent throughout.
# Handles directory creation, ownership, .env patching, and ~/.bashrc
# so DATA_DIR is consistent across all tools and make targets.
#
# Usage:
#   bash infra/setup_gpu_instance.sh
#   bash infra/setup_gpu_instance.sh --data-dir /mnt/persistent
#   bash infra/setup_gpu_instance.sh --data-dir /mnt/persistent --skip-data
#
# Or via make:
#   make setup-gpu DATA_DIR=/mnt/persistent
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_DIR/.env"

# Source existing .env if present — lets DATA_DIR default from .env
if [[ -f "$ENV_FILE" ]]; then
    set -a && source "$ENV_FILE" && set +a
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
SKIP_DATA=false
SKIP_PYTHON=false
TOKENIZE_DATE=""    # empty = auto-detect latest from S3
SIZE="125m"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data)    SKIP_DATA=true;       shift ;;
        --skip-python)  SKIP_PYTHON=true;     shift ;;
        --data-dir)     DATA_DIR="$2";        shift 2 ;;
        --data-dir=*)   DATA_DIR="${1#*=}";   shift ;;
        --date)         TOKENIZE_DATE="$2";   shift 2 ;;
        --date=*)       TOKENIZE_DATE="${1#*=}"; shift ;;
        --size)         SIZE="$2";            shift 2 ;;
        --size=*)       SIZE="${1#*=}";       shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

HF_CACHE_DIR="$(dirname "$DATA_DIR")/hf_cache"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== SLM GPU Instance Setup ==="
log "Repo:     $REPO_DIR"
log "Data:     $DATA_DIR"
log "HF cache: $HF_CACHE_DIR"
log "Results:  $RESULTS_DIR"

# ── GPU check ─────────────────────────────────────────────────────────────────
log "GPU check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Directories ───────────────────────────────────────────────────────────────
log "Creating directory structure..."

# If DATA_DIR parent doesn't exist or is owned by root, fix ownership
DATA_PARENT="$(dirname "$DATA_DIR")"
if [[ ! -w "$DATA_PARENT" ]]; then
    log "  $DATA_PARENT not writable — fixing ownership with sudo..."
    sudo mkdir -p "$DATA_PARENT"
    sudo chown -R "$(whoami):$(whoami)" "$DATA_PARENT"
fi

mkdir -p \
    "$DATA_DIR/tokenized" \
    "$DATA_DIR/tokenizer" \
    "$DATA_DIR/sft/chat" \
    "$DATA_DIR/sft/code" \
    "$DATA_DIR/dpo" \
    "$DATA_DIR/models" \
    "$RESULTS_DIR" \
    "$HF_CACHE_DIR" \
    "$REPO_DIR/logs"

log "  ✓ Directories created"

# ── System dependencies ───────────────────────────────────────────────────────
log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git wget curl unzip htop tmux nvtop build-essential \
    python3.12 python3.12-dev python3.12-venv python3-pip 2>/dev/null || true
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
        python3.12 -m venv .venv
        log "  ✓ Virtual environment created"
    fi

    .venv/bin/pip install --upgrade pip --quiet
    .venv/bin/pip install -r requirements.txt --quiet
    log "  ✓ Python dependencies installed"
fi

# ── Configure .env ────────────────────────────────────────────────────────────
# Patch DATA_DIR, HF_HOME, HF_DATASETS_CACHE in .env — same approach as
# setup.sh so all three path variables are always consistent with DATA_DIR.
log "Configuring .env..."

if [[ ! -f "$ENV_FILE" ]]; then
    cp "$REPO_DIR/.env.sample" "$ENV_FILE"
    log "  Created .env from .env.sample — fill in credentials before training"
fi

_set_env() {
    local key="$1" val="$2"
    if grep -q "^${key}=" "$ENV_FILE"; then
        sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
    else
        echo "${key}=${val}" >> "$ENV_FILE"
    fi
    log "  ${key}=${val}"
}

_set_env "DATA_DIR"           "$DATA_DIR"
_set_env "HF_HOME"            "$HF_CACHE_DIR"
_set_env "HF_DATASETS_CACHE"  "$HF_CACHE_DIR"

# ── Configure ~/.bashrc ───────────────────────────────────────────────────────
# Write path exports to ~/.bashrc so they persist across sessions and
# survive preemptible VM restarts (the disk persists, bashrc is re-sourced).
log "Configuring ~/.bashrc..."

BASHRC_MARKER="# SLM GPU environment"
BASHRC_BLOCK="
${BASHRC_MARKER} (managed by infra/setup_gpu_instance.sh)
export DATA_DIR=${DATA_DIR}
export HF_HOME=${HF_CACHE_DIR}
export HF_DATASETS_CACHE=${HF_CACHE_DIR}
export RESULTS_DIR=${RESULTS_DIR}
"

if grep -q "$BASHRC_MARKER" ~/.bashrc; then
    # Already present — update in place by removing old block and re-adding
    sed -i "/${BASHRC_MARKER}/,/^$/d" ~/.bashrc
    log "  Updated existing SLM block in ~/.bashrc"
fi

echo "$BASHRC_BLOCK" >> ~/.bashrc
log "  ✓ ~/.bashrc updated"

# Export for current session
export DATA_DIR="$DATA_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR"
export RESULTS_DIR="$RESULTS_DIR"

# ── Configure accelerate ──────────────────────────────────────────────────────
log "Configuring accelerate (single GPU — run make accelerate-config-multi for full training)..."
mkdir -p ~/.cache/huggingface/accelerate
cp "$REPO_DIR/accelerate_configs/single_gpu.yaml" \
   ~/.cache/huggingface/accelerate/default_config.yaml
log "  ✓ accelerate configured for single GPU"

# ── Pull data from S3 ─────────────────────────────────────────────────────────
if [[ "$SKIP_DATA" == "true" ]]; then
    log "[SKIP] S3 data pull (--skip-data)"
elif [[ -z "${S3_BUCKET:-}" ]]; then
    log "WARNING: S3_BUCKET not set in .env — skipping data pull"
    log "  Run manually: make tokenize-download SIZE=125m DATE=YYYY-MM-DD"
else
    S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX:-slm/data}"
    log "Pulling tokenized binary from $S3_BASE..."
    log "  (Use make tokenize-download SIZE=125m DATE=YYYY-MM-DD for versioned download)"

    aws s3 sync "${S3_BASE}/tokenized/" "$DATA_DIR/tokenized/" \
        --no-progress \
        --exclude "*" \
        --include "*.bin" \
        --include "*.json"

    BIN_SIZE=$(du -sh "$DATA_DIR/tokenized" 2>/dev/null | cut -f1 || echo "unknown")
    log "  ✓ Tokenized binary pulled ($BIN_SIZE)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
log ""
log "=== Setup complete ==="
log ""
log "Next steps:"
log "  source ~/.bashrc"
log "  make pretrain-mini GPUS=1                  # validate training loop"
log "  make accelerate-config-multi GPUS=8        # configure for full run"
log "  make pretrain SIZE=125m GPUS=8             # full pretraining"
log ""
log "GPU monitoring:"
log "  watch -n 2 nvidia-smi"
log "  nvtop"