#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# alignment/scripts/train_dpo.sh
# Launch NeMo Aligner Direct Preference Optimization (DPO).
#
# DPO trains the model directly on preference pairs (chosen vs rejected)
# without needing a separate reward model. The SFT checkpoint acts as
# the frozen reference policy during training.
#
# Memory note: DPO runs both the trainable policy AND the frozen reference
# model simultaneously → roughly 2x memory of SFT. On a single A6000 (48GB)
# with 125M params this is comfortable. At 1B you may need 2+ GPUs.
#
# Usage:
#   bash train_dpo.sh
#   bash train_dpo.sh --gpus 2
#   bash train_dpo.sh --sft-ckpt /results/slm_sft_code/checkpoints/best.nemo
#   bash train_dpo.sh --beta 0.2    # adjust KL penalty
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/alignment/configs/dpo.yaml"
GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
SFT_CKPT=""
BETA=""
WANDB=false

# Changed from /logs/dpo → /results/dpo_logs
# /logs is not reliably bind-mounted; /results always is
LOG_DIR="/results/dpo_logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)      GPUS="$2";     shift 2 ;;
        --sft-ckpt)  SFT_CKPT="$2"; shift 2 ;;
        --beta)      BETA="$2";     shift 2 ;;
        --wandb)     WANDB=true;    shift ;;
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
    SFT_CKPT=$(find "/results/slm_sft_code/checkpoints" -name "*.nemo" | sort | tail -1 2>/dev/null || true)
    if [[ -z "$SFT_CKPT" ]]; then
        echo "ERROR: No SFT checkpoint found in /results/slm_sft_code/"
        echo "  Run: bash finetune/scripts/train_sft.sh first"
        echo "  Or pass: --sft-ckpt <path>"
        exit 1
    fi
fi

log "SFT checkpoint: $SFT_CKPT"
log "GPUs:           $GPUS"
log "Config:         $CONFIG"
log "Log:            $LOG_FILE"

# ── Validate DPO dataset ──────────────────────────────────────────────────────
if [[ ! -f "/data/dpo/train.jsonl" ]]; then
    echo "ERROR: DPO training data not found at /data/dpo/train.jsonl"
    echo "  Run: python alignment/data/prepare_dpo.py"
    exit 1
fi

DPO_TRAIN_COUNT=$(wc -l < /data/dpo/train.jsonl)
log "DPO train examples: $DPO_TRAIN_COUNT"

# ── Build overrides ───────────────────────────────────────────────────────────
OVERRIDES=(
    "trainer.devices=$GPUS"
    "model.restore_from_path=$SFT_CKPT"
)

if [[ -n "$BETA" ]]; then
    OVERRIDES+=("model.dpo.beta=$BETA")
    log "DPO beta override: $BETA"
fi

if [[ "$WANDB" == "true" ]]; then
    OVERRIDES+=("exp_manager.create_wandb_logger=true")
fi

OVERRIDE_STR="${OVERRIDES[*]}"

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1

# ── Launch ────────────────────────────────────────────────────────────────────
log "Launching DPO training..."

if [[ "$GPUS" -eq 1 ]]; then
    python "$REPO_ROOT/alignment/train_dpo.py" \
        --config-path "$(dirname "$CONFIG")" \
        --config-name "$(basename "$CONFIG" .yaml)" \
        $OVERRIDE_STR \
        2>&1 | tee -a "$LOG_FILE"
else
    torchrun \
        --nproc_per_node="$GPUS" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29502 \
        "$REPO_ROOT/alignment/train_dpo.py" \
        --config-path "$(dirname "$CONFIG")" \
        --config-name "$(basename "$CONFIG" .yaml)" \
        $OVERRIDE_STR \
        2>&1 | tee -a "$LOG_FILE"
fi

log "DPO training complete"
log "Final checkpoint: /results/slm_dpo/checkpoints/"
log ""
log "Next steps:"
log "  Evaluate:  python eval/run_eval.py --model /results/slm_dpo/checkpoints/last.nemo"
log "  Export HF: bash pretrain/scripts/convert_ckpt.sh --direction nemo_to_hf \\"
log "               --input /results/slm_dpo/checkpoints/last.nemo \\"
log "               --output /results/slm_final_hf/"