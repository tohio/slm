#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# alignment/scripts/train_dpo.sh
# Launch NeMo-Aligner Direct Preference Optimization (DPO).
#
# Requires: SFT .nemo checkpoint from make sft
#
# Usage:
#   bash train_dpo.sh
#   bash train_dpo.sh --gpus 2
#   bash train_dpo.sh --sft-ckpt /results/slm_sft_code/checkpoints/last.nemo
#   bash train_dpo.sh --beta 0.05
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
SFT_CKPT=""
BETA=""
WANDB=false
BASE_RESULTS_DIR="${RESULTS_DIR:-/results}"
LOG_DIR="${BASE_RESULTS_DIR}/dpo_logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)     GPUS="$2";     shift 2 ;;
        --sft-ckpt) SFT_CKPT="$2"; shift 2 ;;
        --beta)     BETA="$2";     shift 2 ;;
        --wandb)    WANDB=true;    shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/dpo_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== SLM DPO Alignment ==="

# ── Locate SFT checkpoint ─────────────────────────────────────────────────────
if [[ -z "$SFT_CKPT" ]]; then
    SFT_CKPT=$(find "${BASE_RESULTS_DIR}/slm_sft_code" -name "*.nemo" 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$SFT_CKPT" ]]; then
        echo "ERROR: No SFT checkpoint found in ${BASE_RESULTS_DIR}/slm_sft_code/"
        echo "  Run: make sft"
        exit 1
    fi
fi

log "SFT checkpoint: $SFT_CKPT"
log "GPUs:           $GPUS"
log "Log:            $LOG_FILE"

# ── Validate DPO dataset ──────────────────────────────────────────────────────
if [[ ! -f "/data/dpo/train.jsonl" ]]; then
    echo "ERROR: DPO data not found at /data/dpo/train.jsonl"
    echo "  Run: make prepare-dpo-data"
    exit 1
fi

DPO_TRAIN_COUNT=$(wc -l < /data/dpo/train.jsonl)
log "DPO train examples: $DPO_TRAIN_COUNT"

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0

# ── Build overrides ───────────────────────────────────────────────────────────
OVERRIDES=(
    trainer.devices="$GPUS"
    trainer.num_nodes=1
    model.restore_from_path="$SFT_CKPT"
)

[[ -n "$BETA" ]] && OVERRIDES+=(model.dpo.beta="$BETA") && log "DPO beta override: $BETA"
[[ "$WANDB" == "true" ]] && OVERRIDES+=(exp_manager.create_wandb_logger=true)

# ── Launch ────────────────────────────────────────────────────────────────────
log "Launching DPO training..."

python "$REPO_ROOT/alignment/train_dpo.py" \
    --config-path "$REPO_ROOT/alignment/configs" \
    --config-name "dpo" \
    "${OVERRIDES[@]}" \
    2>&1 | tee -a "$LOG_FILE"

log "DPO training complete"
log "Final checkpoint: ${BASE_RESULTS_DIR}/slm_dpo/"
log ""
log "Next steps:"
log "  make eval-dpo"
log "  make convert-hf"