#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# infra/setup_train.sh
# Setup script for a GPU training instance.
#
# Safe to re-run after a preemptible VM restart — idempotent throughout.
# Handles directory creation, ownership, .env patching, and ~/.bashrc
# so DATA_DIR is consistent across all tools and make targets.
#
# Usage:
#   bash infra/setup_train.sh
#   bash infra/setup_train.sh --data-dir /mnt/persistent
#   bash infra/setup_train.sh --data-dir /mnt/persistent --skip-data
#   bash infra/setup_train.sh --data-dir /mnt/persistent --size 125m --run-id 125m-20260412-a8f3c9
#
# Or via make:
#   make setup-train DATA_DIR=/mnt/persistent SIZE=125m RUN_ID=125m-20260412-a8f3c9
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
INSTALLER="${INSTALLER:-pip}"
VENV_DIR="$REPO_DIR/.venv"
SKIP_PYTHON=false
RUN_ID="${RUN_ID:-}"
SIZE="${SIZE:-125m}"
DATASET_SIZE="${DATASET_SIZE:-}"
ARTIFACT_BACKEND="${ARTIFACT_BACKEND:-s3}"
RESTORE_STAGES="${ARTIFACT_STAGES:-tokenized,tokenizer,metadata}"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --installer) INSTALLER="${2:?--installer requires a value}"; shift 2 ;;
        --help|-h) echo "make setup-train [INSTALLER=pip|uv|conda] [DATA_DIR=path] [DATASET_SIZE=size DATASET_RUN_ID=id]"; exit 0 ;;
        --skip-data)    SKIP_DATA=true;          shift ;;
        --skip-python)  SKIP_PYTHON=true;        shift ;;
        --data-dir)     DATA_DIR="$2";           shift 2 ;;
        --data-dir=*)   DATA_DIR="${1#*=}";      shift ;;
        --run-id)       RUN_ID="$2";             shift 2 ;;
        --run-id=*)     RUN_ID="${1#*=}";        shift ;;
        --dataset-size) DATASET_SIZE="$2"; shift 2 ;;
        --backend)      ARTIFACT_BACKEND="$2"; shift 2 ;;
        --stages)       RESTORE_STAGES="${2:?--stages requires a value}"; shift 2 ;;
        --size)         SIZE="$2";               shift 2 ;;
        --size=*)       SIZE="${1#*=}";          shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

source "$REPO_DIR/infra/setup_environment.sh"
check_installer
cd "$REPO_DIR"
DATA_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$DATA_DIR")"
HF_CACHE_DIR="$(dirname "$DATA_DIR")/hf_cache"
DATASET_SIZE="${DATASET_SIZE:-$SIZE}"
RUN_DATA_DIR="$DATA_DIR/runs/$DATASET_SIZE"
MODEL_DATA_DIR="$DATA_DIR/runs/$SIZE"
case "$ARTIFACT_BACKEND" in s3|hf) ;; *) echo "Backend must be s3 or hf"; exit 1;; esac

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== SLM GPU Instance Setup ==="
log "Repo:     $REPO_DIR"
log "Data:     $DATA_DIR"
log "HF cache: $HF_CACHE_DIR"
log "Results:  $RESULTS_DIR"
log "Run data: $RUN_DATA_DIR"
if [[ -n "$RUN_ID" ]]; then
    log "Run ID:   $RUN_ID"
fi

# ── GPU check ─────────────────────────────────────────────────────────────────
log "GPU check:"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: setup-train requires an NVIDIA GPU host with a working driver (nvidia-smi)."
    log "CPU-executed training tests use an already installed training environment; this is not a CPU-only installer."
    exit 1
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '\r')"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '\r')"
MIN_DRIVER_VERSION="580.65.06"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
log "Detected GPU: $GPU_NAME"
if [[ "$(printf '%s\n' "$MIN_DRIVER_VERSION" "$DRIVER_VERSION" | sort -V | head -1)" != "$MIN_DRIVER_VERSION" ]]; then
    log "ERROR: CUDA 13.0 requires NVIDIA driver >= $MIN_DRIVER_VERSION; found $DRIVER_VERSION"
    exit 1
fi

# ── Directories ───────────────────────────────────────────────────────────────
log "Creating directory structure..."

# Never chown DATA_PARENT (e.g. /mnt) or traverse existing directory trees.
# Only create missing project roots. Existing unwritable locations require the
# operator to grant access explicitly; setup must not take ownership of shared data.
ensure_project_directory() {
    local path="$1"
    if [[ -e "$path" ]]; then
        if [[ ! -d "$path" || ! -w "$path" || ! -x "$path" ]]; then
            log "ERROR: Project path is not a writable/searchable directory: $path"
            log "Grant access to this specific path, or select a different project location."
            exit 1
        fi
    elif ! mkdir -p -- "$path" 2>/dev/null; then
        log "  Creating project directory with sudo: $path"
        sudo mkdir -p -- "$path"
        # No -R: parent directories, existing contents and sibling mounts are untouched.
        sudo chown "$(id -u):$(id -g)" -- "$path"
    fi
}
ensure_project_directory "$DATA_DIR"
ensure_project_directory "$HF_CACHE_DIR"
ensure_project_directory "$RESULTS_DIR"

mkdir -p \
    "$RUN_DATA_DIR/tokenized" \
    "$MODEL_DATA_DIR/sft_instruct" \
    "$MODEL_DATA_DIR/sft_code" \
    "$MODEL_DATA_DIR/dpo_chat" \
    "$RUN_DATA_DIR/tokenizer" \
    "$DATA_DIR/models" \
    "$RESULTS_DIR" \
    "$HF_CACHE_DIR" \
    "$REPO_DIR/logs"

log "  ✓ Directories created"

# ── System dependencies ───────────────────────────────────────────────────────
log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common
if ! command -v python3.12 &>/dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -qq
fi
sudo apt-get install -y -qq \
    git wget curl unzip htop tmux nvtop build-essential \
    python3.12 python3.12-dev python3.12-venv python3-pip
log "  ✓ System dependencies installed"

# ── AWS CLI ───────────────────────────────────────────────────────────────────
if [[ "$ARTIFACT_BACKEND" == "s3" ]] && { ! command -v aws &>/dev/null || [[ $(aws --version 2>&1) == *"aws-cli/1"* ]]; }; then
    log "Installing AWS CLI v2..."
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/awscliv2.zip /tmp/aws
fi
if [[ "$ARTIFACT_BACKEND" == "s3" ]]; then log "  ✓ AWS CLI: $(aws --version 2>&1 | head -1)"; fi

# ── Python environment ────────────────────────────────────────────────────────
if [[ "$SKIP_PYTHON" == "true" ]]; then
    log "[SKIP] Python environment (--skip-python)"
else
    log "Setting up Python environment..."
    cd "$REPO_DIR"

    install_environment "$REPO_DIR/requirements-training.txt"
    run_environment "$VENV_DIR/bin/python" infra/verify_environment.py --require-cuda

    # FA3 imports torch during its build. Use the installed CUDA 13.0 stack
    # and compile Ampere + Hopper kernels, including backward, from the pin.
    run_environment "$VENV_DIR/bin/python" - <<'PY'
from pathlib import Path
import re
import subprocess
from torch.utils.cpp_extension import CUDA_HOME

nvcc = Path(CUDA_HOME) / "bin" / "nvcc" if CUDA_HOME else None
if nvcc is None or not nvcc.is_file():
    raise SystemExit("FA3 needs the CUDA 13.0 development toolkit (nvcc + headers). Install it and set CUDA_HOME before rerunning setup-train; see infra/README.md.")
version = subprocess.check_output([str(nvcc), "--version"], text=True)
if not re.search(r"release 13\.0[, ]", version):
    raise SystemExit(f"FA3 must build against the CUDA 13.0 toolkit matching PyTorch; found:\n{version}")
print(f"FA3 compiler: {nvcc}")
PY
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_DISABLE_SM80=FALSE \
    FLASH_ATTENTION_DISABLE_BACKWARD=FALSE \
    MAX_JOBS="${MAX_JOBS:-4}" \
        install_packages --no-build-isolation --no-deps -r "$REPO_DIR/requirements-flash-attention.txt"
    run_environment "$VENV_DIR/bin/python" -c \
        'import flash_attn_interface; print("FlashAttention-3 import:", flash_attn_interface.__file__)'
    run_environment "$VENV_DIR/bin/python" -m pip check

    log "  ✓ Python dependencies installed"
fi

# ── Configure .env ────────────────────────────────────────────────────────────
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
log "Configuring ~/.bashrc..."

BASHRC_MARKER="# SLM GPU environment"
BASHRC_BLOCK="
${BASHRC_MARKER} (managed by infra/setup_train.sh)
export DATA_DIR=${DATA_DIR}
export HF_HOME=${HF_CACHE_DIR}
export HF_DATASETS_CACHE=${HF_CACHE_DIR}
export RESULTS_DIR=${RESULTS_DIR}
$(activation_command)
"

if grep -q "$BASHRC_MARKER" ~/.bashrc; then
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

# ── Pull selected run-scoped artifacts ─────────────────────────────────────────
if [[ "$SKIP_DATA" == "true" ]]; then
    log "[SKIP] Artifact pull (--skip-data)"
elif [[ -z "$RUN_ID" ]]; then
    log "No dataset RUN_ID selected; environment setup only. Restore artifacts explicitly when ready."
else
    cd "$REPO_DIR"

    log "Restoring $ARTIFACT_BACKEND artifacts (model=$SIZE, dataset=$DATASET_SIZE, run_id=$RUN_ID)..."
    run_environment "$VENV_DIR/bin/python" curator/scripts/upload_s3.py artifacts-download \
        --size "$DATASET_SIZE" \
        --run-id "$RUN_ID" --backend "$ARTIFACT_BACKEND" \
        --stages "$RESTORE_STAGES"

    TRAIN_BIN="$RUN_DATA_DIR/tokenized/train.bin"
    if [[ -f "$TRAIN_BIN" ]]; then
        BIN_SIZE=$(du -sh "$TRAIN_BIN" | cut -f1)
        log "  ✓ train.bin: $BIN_SIZE"
    else
        log "  ERROR: train.bin not found at $TRAIN_BIN after artifact restore"
        exit 1
    fi

    TOKENIZER_FILE="$RUN_DATA_DIR/tokenizer/tokenizer.json"
    TOKENIZER_CONFIG="$RUN_DATA_DIR/tokenizer/tokenizer_config.json"

    if [[ -f "$TOKENIZER_FILE" ]]; then
        log "  ✓ tokenizer.json present"
    else
        log "  ERROR: tokenizer.json not found after artifact restore"
        exit 1
    fi

    if [[ -f "$TOKENIZER_CONFIG" ]]; then
        log "  ✓ tokenizer_config.json present (chat_template included)"
    else
        log "  ERROR: tokenizer_config.json not found after artifact restore"
        log "  This file contains the chat_template and must be restored with the tokenizer."
        exit 1
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
log ""
log "=== Setup complete ==="
log ""
log "Next steps (N is the number of GPUs you choose; no GPU family/count is required):"
log "  source ~/.bashrc"
log "  vi .env                                    # HF_TOKEN, required W&B key/project, selected storage credentials"
log "  make config-gen SIZE=$SIZE GPUS=N"
log "  make pretrain SIZE=$SIZE DATASET_SIZE=$DATASET_SIZE DATASET_RUN_ID=$RUN_ID GPUS=N            # full pretraining"
log ""
log "GPU monitoring:"
log "  watch -n 2 nvidia-smi"
log "  nvtop"
