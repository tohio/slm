#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# pretrain/scripts/convert_pretrain.sh
# Convert NeMo 2.x distributed checkpoint → mcore_gpt.nemo for NeMo-Aligner.
#
# NeMo 2.x pretraining saves a distributed checkpoint directory.
# NeMo-Aligner SFT/DPO requires a .nemo tarball (mcore_gpt format).
# This script bridges the two using NeMo's export API.
#
# Usage:
#   bash convert_pretrain.sh
#   bash convert_pretrain.sh --input /results/slm_gpt_125m --output /results/slm_gpt_125m/mcore_gpt.nemo
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT_DIR="${RESULTS_DIR:-/results}/slm_gpt_125m"
OUTPUT_PATH="${RESULTS_DIR:-/results}/slm_gpt_125m/mcore_gpt.nemo"
LOG_DIR="${RESULTS_DIR:-/results}/pretrain_logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)  INPUT_DIR="$2";   shift 2 ;;
        --output) OUTPUT_PATH="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/convert_pretrain_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== NeMo 2.x → mcore_gpt.nemo Conversion ==="
log "Input:  $INPUT_DIR"
log "Output: $OUTPUT_PATH"

# ── Validate input ────────────────────────────────────────────────────────────
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input checkpoint directory not found: $INPUT_DIR"
    echo "  Run: make pretrain"
    exit 1
fi

# ── Convert ───────────────────────────────────────────────────────────────────
log "Converting checkpoint..."

python3 - <<PYEOF
from pathlib import Path
from nemo.collections import llm

input_path = Path("$INPUT_DIR")
output_path = Path("$OUTPUT_PATH")
output_path.parent.mkdir(parents=True, exist_ok=True)

# NeMo 2.x saves checkpoints as:
#   <results_dir>/slm_gpt/<date>/checkpoints/<name>/context/ + weights/
# Find the checkpoint with the lowest val_loss (best model)
ckpt_dirs = sorted(input_path.glob("*/*/checkpoints/*"), key=lambda p: p.name)
# Filter to dirs that have a context subdirectory (valid NeMo 2.x checkpoints)
# and prefer -last checkpoints, otherwise pick lowest val_loss
valid = [d for d in ckpt_dirs if (d / "context").exists()]
if not valid:
    raise FileNotFoundError(f"No valid NeMo 2.x checkpoints found under {input_path}")

# Prefer -last checkpoint, fallback to latest by name
last = [d for d in valid if d.name.endswith("-last")]
ckpt_path = last[-1] if last else valid[-1]

print(f"Converting checkpoint: {ckpt_path}")
print(f"Output: {output_path}")

llm.export_ckpt(
    path=ckpt_path,
    target="nemo",
    output_path=output_path,
)

print(f"✓ Conversion complete: {output_path}")
print(f"  Size: {output_path.stat().st_size / 1024**3:.2f} GB")
PYEOF

log "✓ Conversion complete: $OUTPUT_PATH"
log "  Ready for NeMo-Aligner SFT/DPO"
log ""
log "Next step: make sft"