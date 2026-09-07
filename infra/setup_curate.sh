#!/usr/bin/env bash
# infra/setup_curate.sh
# ---------------
# Bootstrap script for a fresh Ubuntu 22.04 instance.
# Run once after cloning the repo to set up the environment.
#
# Usage:
#   bash infra/setup_curate.sh [--data-dir /data/slm/data]
#
# The script:
#   1. Installs system dependencies (Python 3.12, gcc, build tools)
#   2. Creates a Python virtual environment
#   3. Installs Python dependencies
#   4. Installs KenLM Python bindings (required for validation)
#   5. Downloads the spaCy English model
#   6. Creates the required data directory structure
#   7. Configures .env with correct paths
#   8. Validates the environment
#
# Assumptions:
#   - Ubuntu 22.04
#   - Running from the repo root (/data/slm or wherever you cloned it)
#   - EBS volume (or local disk) mounted at the parent of DATA_DIR
#   - Internet access to PyPI and HuggingFace

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
INSTALLER="${INSTALLER:-pip}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --installer) INSTALLER="${2:?--installer requires a value}"; shift 2 ;;
        --data-dir) DATA_DIR="${2:?--data-dir requires a value}"; shift 2 ;;
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        --help|-h) echo "make setup-curate [INSTALLER=pip|uv|conda] [DATA_DIR=path]"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
source "$REPO_DIR/infra/setup_environment.sh"
check_installer
cd "$REPO_DIR"
# Store an absolute data path so setup and later invocations agree.
DATA_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$DATA_DIR")"

HF_CACHE_DIR="$(dirname "$DATA_DIR")/hf_cache"
VENV_DIR="${REPO_DIR}/.venv"

echo ""
echo "========================================"
echo " SLM Instance Setup"
echo "========================================"
echo " Repo:      $REPO_DIR"
echo " Data dir:  $DATA_DIR"
echo " HF cache:  $HF_CACHE_DIR"
echo " Venv:      $VENV_DIR"
echo "========================================"
echo ""

# ── 1. System dependencies ────────────────────────────────────────────────────

echo "==> Installing system dependencies..."

# Add deadsnakes PPA before attempting to install python3.12.
# Ubuntu 22.04's default apt repos may not have python3.12 — the PPA
# must be added first regardless of whether python3.12 is already present.
echo "  Adding deadsnakes PPA for Python 3.12..."
sudo apt-get install -y software-properties-common -qq
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -qq

sudo apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    gcc \
    g++ \
    build-essential \
    make \
    cmake \
    libboost-all-dev \
    git \
    curl \
    tmux \
    htop \
    nvme-cli

echo "  Python: $(python3.12 --version)"
echo "  GCC:    $(gcc --version | head -1)"

# ── 2. Virtual environment ────────────────────────────────────────────────────

echo ""
echo "==> Installing curation environment ($INSTALLER) at $VENV_DIR..."
install_environment "${REPO_DIR}/requirements-curation.txt"

# ── 4. KenLM Python bindings ──────────────────────────────────────────────────
# KenLM is not on PyPI — must be built from source.
# Required for the perplexity filter in the validation stage.

echo ""
echo "==> Installing KenLM Python bindings..."
install_packages https://github.com/kpu/kenlm/archive/master.zip
echo "  KenLM installed"

# ── 5. spaCy English model ────────────────────────────────────────────────────

echo ""
echo "==> Downloading spaCy English model..."
run_environment "$VENV_DIR/bin/python" -m spacy download en_core_web_sm

# ── 6. Data directory structure ───────────────────────────────────────────────
# Curation artifacts are run-scoped under data/runs/<size>/. Source classes
# create their own per-source subdirectories at first use, for example:
#
#   data/runs/125m/raw/wikipedia/
#   data/runs/125m/filtered/wikipedia/
#   data/runs/125m/curated/
#
# This layout lets artifact restore workflows repopulate a prior run, delete
# one source directory, and rerun curation while existing on-disk sources are
# skipped. The source-name list lives in config/data_mix.py and should not be
# duplicated here.

echo ""
echo "==> Creating data directory structure at $DATA_DIR..."

CURATION_SIZES=("smoke" "mini" "125m" "350m" "1b")

mkdir -p \
    "${DATA_DIR}/raw" \
    "${DATA_DIR}/filtered" \
    "${DATA_DIR}/curated" \
    "${DATA_DIR}/dedup_scratch" \
    "${DATA_DIR}/validated" \
    "${DATA_DIR}/tokenized" \
    "${DATA_DIR}/runs" \
    "${DATA_DIR}/models" \
    "${HF_CACHE_DIR}"

for size in "${CURATION_SIZES[@]}"; do
    mkdir -p \
        "${DATA_DIR}/runs/${size}/raw" \
        "${DATA_DIR}/runs/${size}/filtered" \
        "${DATA_DIR}/runs/${size}/dedup_scratch" \
        "${DATA_DIR}/runs/${size}/curated" \
        "${DATA_DIR}/runs/${size}/validated" \
        "${DATA_DIR}/runs/${size}/tokenized" \
        "${DATA_DIR}/runs/${size}/tokenizer"
done

echo "  Created: $DATA_DIR"
echo "  Created: $HF_CACHE_DIR"
echo "  Created run-scoped curation dirs for: ${CURATION_SIZES[*]}"

# ── 7. Configure .env ─────────────────────────────────────────────────────────

echo ""
echo "==> Configuring .env..."

ENV_FILE="${REPO_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
    cp "${REPO_DIR}/.env.sample" "$ENV_FILE"
    echo "  Created .env from .env.sample"
fi

# Set DATA_DIR — replace existing value or append
if grep -q "^DATA_DIR=" "$ENV_FILE"; then
    sed -i "s|^DATA_DIR=.*|DATA_DIR=${DATA_DIR}|" "$ENV_FILE"
else
    echo "DATA_DIR=${DATA_DIR}" >> "$ENV_FILE"
fi

# Set HF cache dirs — replace existing or append
if grep -q "^HF_HOME=" "$ENV_FILE"; then
    sed -i "s|^HF_HOME=.*|HF_HOME=${HF_CACHE_DIR}|" "$ENV_FILE"
else
    echo "HF_HOME=${HF_CACHE_DIR}" >> "$ENV_FILE"
fi

if grep -q "^HF_DATASETS_CACHE=" "$ENV_FILE"; then
    sed -i "s|^HF_DATASETS_CACHE=.*|HF_DATASETS_CACHE=${HF_CACHE_DIR}|" "$ENV_FILE"
else
    echo "HF_DATASETS_CACHE=${HF_CACHE_DIR}" >> "$ENV_FILE"
fi

echo "  DATA_DIR=${DATA_DIR}"
echo "  HF_HOME=${HF_CACHE_DIR}"
echo "  HF_DATASETS_CACHE=${HF_CACHE_DIR}"

# ── 8. Shell profile ──────────────────────────────────────────────────────────

echo ""
echo "==> Adding environment variables to ~/.bashrc..."

BASHRC_BLOCK="
# SLM environment (added by infra/setup_curate.sh)
export HF_HOME=${HF_CACHE_DIR}
export HF_DATASETS_CACHE=${HF_CACHE_DIR}
export DATA_DIR=${DATA_DIR}
"

if grep -q "^# SLM environment" ~/.bashrc; then
    sed -i "/^# SLM environment/,/^$/d" ~/.bashrc
fi
echo "$BASHRC_BLOCK" >> ~/.bashrc
echo "  Updated ~/.bashrc"

# Export for current session
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}"
export DATA_DIR="${DATA_DIR}"

# ── 9. Validate ───────────────────────────────────────────────────────────────

echo ""
echo "==> Validating environment..."

ERRORS=0

# Verify the curation stack. The GPU training stack is installed and checked
# separately by setup_train.sh.
run_environment "$VENV_DIR/bin/python" "${REPO_DIR}/infra/verify_environment.py" --profile curation \
    || ERRORS=$((ERRORS + 1))

# Check unpinned curation packages that are outside the version contract.
run_environment "$VENV_DIR/bin/python" -c "
import boto3, datatrove, dotenv, fasttext, kenlm, orjson, requests, spacy
import trafilatura, tqdm, warcio
print('  Curation packages importable')
" || ERRORS=$((ERRORS + 1))

# Check spaCy model
run_environment "$VENV_DIR/bin/python" -c "import spacy; spacy.load('en_core_web_sm'); print('  spaCy en_core_web_sm OK')" \
    || { echo "  MISSING spaCy model — rerun make setup-curate INSTALLER=$INSTALLER"; ERRORS=$((ERRORS + 1)); }

## Check data directories
for dir in "${DATA_DIR}/raw" "${DATA_DIR}/filtered" "${DATA_DIR}/curated" \
           "${DATA_DIR}/dedup_scratch" "${DATA_DIR}/validated" \
           "${DATA_DIR}/tokenized" "${DATA_DIR}/runs" \
           "${DATA_DIR}/models" "${HF_CACHE_DIR}"; do
    if [ -d "$dir" ]; then
        echo "  OK: $dir"
    else
        echo "  MISSING: $dir"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check run-scoped curation directories
for size in "${CURATION_SIZES[@]}"; do
    for dir in \
        "${DATA_DIR}/runs/${size}/raw" \
        "${DATA_DIR}/runs/${size}/filtered" \
        "${DATA_DIR}/runs/${size}/dedup_scratch" \
        "${DATA_DIR}/runs/${size}/curated" \
        "${DATA_DIR}/runs/${size}/validated" \
        "${DATA_DIR}/runs/${size}/tokenized" \
        "${DATA_DIR}/runs/${size}/tokenizer"; do
        if [ -d "$dir" ]; then
            echo "  OK: $dir"
        else
            echo "  MISSING: $dir"
            ERRORS=$((ERRORS + 1))
        fi
    done
done

# Check fasttext model — warn only (downloaded separately)
if [ -f "${DATA_DIR}/models/lid.176.ftz" ]; then
    echo "  OK: fasttext language model found"
else
    echo "  WARNING: fasttext model not found — run: make download-fasttext-model DATA_DIR=${DATA_DIR}"
    echo "           Required before running any curation target"
fi

# Check matched CCNet model pair — warn only (downloaded separately)
if [ -f "${DATA_DIR}/models/en.arpa.bin" ] && [ -f "${DATA_DIR}/models/en.sp.model" ]; then
    echo "  OK: CCNet KenLM and SentencePiece models found"
else
    echo "  WARNING: CCNet model pair incomplete — run: make download-kenlm-model DATA_DIR=${DATA_DIR}"
    echo "           Required before running any curation target"
fi

# Check .env required variables — warnings only, not hard errors.
# Credentials must be populated before running the pipeline but are
# not required for setup itself to succeed.
echo ""
echo "==> Checking .env variables..."
# Storage settings depend on the later selected stages/destination. A
# Dataset-only HF push must not require a Bucket (or any model repository).
REQUIRED_VARS=("HF_TOKEN")
MISSING_CREDS=0
for var in "${REQUIRED_VARS[@]}"; do
    value=$(grep "^${var}=" "$ENV_FILE" | cut -d'=' -f2 || true)
    if [ -z "$value" ]; then
        echo "  WARNING: ${var} is not set in .env — required before running pipeline"
        MISSING_CREDS=$((MISSING_CREDS + 1))
    else
        echo "  OK: ${var} is set"
    fi
done

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "========================================"
    echo " Setup complete — no errors"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review ${ENV_FILE}; configure only the storage destination you will use"
    echo "  2. source ~/.bashrc  (or open a new shell)"
    echo "  3. $(activation_command)"
    echo "  4. make download-fasttext-model DATA_DIR=${DATA_DIR}"
    echo "  5. make download-kenlm-model    DATA_DIR=${DATA_DIR}"
    echo "  6. Validate the curation pipeline with smoke (prerequisite checks run internally):"
    echo "       make curate-smoke DATA_DIR=${DATA_DIR}"
    echo "       make validate SIZE=smoke DATA_DIR=${DATA_DIR}"
    echo ""
    echo "     If smoke passes, continue with mini or a production size:"
    echo "       make curate-mini DATA_DIR=${DATA_DIR}                 # functional mini-scale curation"
    echo "       make curate SIZE=125m WORKERS=<n> DATA_DIR=${DATA_DIR}"
    echo "       make curate SIZE=350m WORKERS=<n> DATA_DIR=${DATA_DIR}"
    echo "       make curate SIZE=1b   WORKERS=<n> DATA_DIR=${DATA_DIR}"
    echo ""
    echo "     Runtime varies with CPU, network, storage, cache state, and Common Crawl throughput."
    echo ""
else
    echo "========================================"
    echo " Setup completed with $ERRORS error(s)"
    echo " Fix the errors above before proceeding"
    echo "========================================"
    exit 1
fi
