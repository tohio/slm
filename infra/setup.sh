#!/usr/bin/env bash
# infra/setup.sh
# ---------------
# Bootstrap script for a fresh Ubuntu 22.04 instance.
# Run once after cloning the repo to set up the environment.
#
# Usage:
#   bash infra/setup.sh [--data-dir /data/slm/data]
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
DEFAULT_DATA_DIR="${REPO_DIR}/data"
DATA_DIR="${1:-$DEFAULT_DATA_DIR}"

# Parse --data-dir flag
for arg in "$@"; do
    case $arg in
        --data-dir=*) DATA_DIR="${arg#*=}" ;;
        --data-dir)   shift; DATA_DIR="$1" ;;
    esac
done

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
echo "==> Creating virtual environment at $VENV_DIR..."
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 3. Python dependencies ────────────────────────────────────────────────────

echo ""
echo "==> Installing Python dependencies..."
pip install -r "${REPO_DIR}/requirements.txt"

# ── 4. KenLM Python bindings ──────────────────────────────────────────────────
# KenLM is not on PyPI — must be built from source.
# Required for the perplexity filter in the validation stage.

echo ""
echo "==> Installing KenLM Python bindings..."
pip install https://github.com/kpu/kenlm/archive/master.zip
echo "  KenLM installed"

# ── 5. spaCy English model ────────────────────────────────────────────────────

echo ""
echo "==> Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# ── 6. Data directory structure ───────────────────────────────────────────────

echo ""
echo "==> Creating data directory structure at $DATA_DIR..."
mkdir -p \
    "${DATA_DIR}/raw/wikipedia" \
    "${DATA_DIR}/raw/code" \
    "${DATA_DIR}/raw/common_crawl" \
    "${DATA_DIR}/filtered" \
    "${DATA_DIR}/curated" \
    "${DATA_DIR}/dedup_scratch" \
    "${DATA_DIR}/validated" \
    "${DATA_DIR}/tokenized" \
    "${DATA_DIR}/models" \
    "${HF_CACHE_DIR}"

echo "  Created: $DATA_DIR"
echo "  Created: $HF_CACHE_DIR"

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
# SLM environment (added by infra/setup.sh)
export HF_HOME=${HF_CACHE_DIR}
export HF_DATASETS_CACHE=${HF_CACHE_DIR}
export DATA_DIR=${DATA_DIR}
"

if ! grep -q "SLM environment" ~/.bashrc; then
    echo "$BASHRC_BLOCK" >> ~/.bashrc
    echo "  Added to ~/.bashrc"
else
    echo "  Already present in ~/.bashrc — skipping"
fi

# Export for current session
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}"
export DATA_DIR="${DATA_DIR}"

# ── 9. Validate ───────────────────────────────────────────────────────────────

echo ""
echo "==> Validating environment..."

ERRORS=0

# Check Python packages
python -c "
import sys
packages = [
    'torch', 'transformers', 'datasets', 'tokenizers',
    'accelerate', 'trl', 'trafilatura', 'langdetect',
    'warcio', 'datatrove', 'orjson', 'spacy',
    'boto3', 'dotenv', 'tqdm', 'requests', 'kenlm',
]
missing = []
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'  MISSING packages: {missing}')
    sys.exit(1)
else:
    print(f'  All {len(packages)} packages importable')
" || ERRORS=$((ERRORS + 1))

# Check spaCy model
python -c "import spacy; spacy.load('en_core_web_sm'); print('  spaCy en_core_web_sm OK')" \
    || { echo "  MISSING spaCy model — run: python -m spacy download en_core_web_sm"; ERRORS=$((ERRORS + 1)); }

# Check data directories
for dir in "${DATA_DIR}/raw" "${DATA_DIR}/filtered" "${DATA_DIR}/curated" \
           "${DATA_DIR}/validated" "${DATA_DIR}/models" "${HF_CACHE_DIR}"; do
    if [ -d "$dir" ]; then
        echo "  OK: $dir"
    else
        echo "  MISSING: $dir"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check KenLM model — warn only, not a hard error (downloaded separately)
if [ -f "${DATA_DIR}/models/en.arpa.bin" ]; then
    echo "  OK: KenLM model found"
else
    echo "  WARNING: KenLM model not found — run: make download-kenlm-model DATA_DIR=${DATA_DIR}"
    echo "           Required before running: make validate"
fi

# Check .env required variables — warnings only, not hard errors.
# Credentials must be populated before running the pipeline but are
# not required for setup itself to succeed.
echo ""
echo "==> Checking .env variables..."
REQUIRED_VARS=("S3_BUCKET" "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "WANDB_API_KEY" "HF_TOKEN")
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
    if [ "$MISSING_CREDS" -gt 0 ]; then
        echo "  1. Fill in missing credentials in ${ENV_FILE}"
        echo "  2. source ~/.bashrc  (or open a new shell)"
        echo "  3. source ${VENV_DIR}/bin/activate"
        echo "  4. make download-kenlm-model DATA_DIR=${DATA_DIR}"
        echo "  5. make curate-mini"
    else
        echo "  1. source ~/.bashrc  (or open a new shell)"
        echo "  2. source ${VENV_DIR}/bin/activate"
        echo "  3. make download-kenlm-model DATA_DIR=${DATA_DIR}"
        echo "  4. make curate-mini"
    fi
    echo ""
else
    echo "========================================"
    echo " Setup completed with $ERRORS error(s)"
    echo " Fix the errors above before proceeding"
    echo "========================================"
    exit 1
fi