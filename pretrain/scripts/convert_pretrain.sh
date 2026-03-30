#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# pretrain/scripts/convert_pretrain.sh
# Convert NeMo 2.x distributed checkpoint → mcore_gpt.nemo for NeMo-Aligner.
#
# NeMo 2.x saves distributed checkpoints that NeMo-Aligner cannot load directly.
# This script packages the checkpoint into the .nemo tarball format expected by
# GPTSFTModel.restore_from() and GPTDPOModel.restore_from().
#
# Usage:
#   bash convert_pretrain.sh
#   bash convert_pretrain.sh --input /results/slm_gpt_125m --output /results/slm_gpt_125m/mcore_gpt.nemo
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_RESULTS_DIR="${RESULTS_DIR:-/results}"
INPUT_DIR="${BASE_RESULTS_DIR}/slm_gpt_125m"
OUTPUT_PATH="${BASE_RESULTS_DIR}/slm_gpt_125m/mcore_gpt.nemo"
TOKENIZER="/data/tokenizer/slm_tokenizer.model"
LOG_DIR="${BASE_RESULTS_DIR}/pretrain_logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)     INPUT_DIR="$2";    shift 2 ;;
        --output)    OUTPUT_PATH="$2";  shift 2 ;;
        --tokenizer) TOKENIZER="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/convert_pretrain_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== NeMo 2.x → mcore_gpt.nemo Conversion ==="
log "Input:     $INPUT_DIR"
log "Output:    $OUTPUT_PATH"
log "Tokenizer: $TOKENIZER"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    echo "  Run: make pretrain"
    exit 1
fi

if [[ ! -f "$TOKENIZER" ]]; then
    echo "ERROR: Tokenizer not found: $TOKENIZER"
    echo "  Run: make setup-instance"
    exit 1
fi

# ── Convert ───────────────────────────────────────────────────────────────────
log "Running conversion script..."

python3 "$REPO_ROOT/pretrain/convert_pretrain.py" \
    --input     "$INPUT_DIR" \
    --output    "$OUTPUT_PATH" \
    --tokenizer "$TOKENIZER" \
    2>&1 | tee -a "$LOG_FILE"

if [[ -f "$OUTPUT_PATH" ]]; then
    SIZE=$(du -sh "$OUTPUT_PATH" | cut -f1)
    log "✓ Conversion complete: $OUTPUT_PATH ($SIZE)"
    log "  Ready for: make sft"
else
    echo "ERROR: Output file not created: $OUTPUT_PATH"
    exit 1
fi