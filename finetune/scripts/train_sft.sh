#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# finetune/scripts/train_sft.sh
# Launch NeMo-Aligner SFT.
#
# Runs sequentially:
#   Stage 1: SFT on general chat data  (OpenAssistant / Dolly)
#   Stage 2: SFT on coding data        (CodeSearchNet / The Stack subset)
#
# Requires: mcore_gpt.nemo produced by make convert-pretrain
#
# Usage:
#   bash train_sft.sh                          # run both stages
#   bash train_sft.sh --stage chat             # chat SFT only
#   bash train_sft.sh --stage code             # code SFT only
#   bash train_sft.sh --gpus 4
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STAGE="both"
GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
WANDB=false
PRETRAIN_CKPT=""
LOG_DIR="/results/sft_logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)         STAGE="$2";          shift 2 ;;
        --gpus)          GPUS="$2";           shift 2 ;;
        --pretrain-ckpt) PRETRAIN_CKPT="$2";  shift 2 ;;
        --wandb)         WANDB=true;          shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Locate mcore_gpt.nemo pretrain checkpoint ─────────────────────────────────
if [[ -z "$PRETRAIN_CKPT" ]]; then
    PRETRAIN_CKPT=$(find /results/slm_gpt_125m -name "mcore_gpt.nemo" 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$PRETRAIN_CKPT" ]]; then
        echo "ERROR: mcore_gpt.nemo not found in /results/slm_gpt_125m/"
        echo "  Run: make convert-pretrain"
        exit 1
    fi
fi
log "Pretrain checkpoint: $PRETRAIN_CKPT"

# ── Generic SFT launcher ──────────────────────────────────────────────────────
run_sft() {
    local stage_name="$1"
    local config_name="$2"
    local restore_from="$3"
    local log_file="$LOG_DIR/${stage_name}_${TIMESTAMP}.log"

    log "--- Starting SFT Stage: $stage_name ---"
    log "Config:       $config_name"
    log "Restore from: $restore_from"
    log "GPUs:         $GPUS"
    log "Log:          $log_file"

    if [[ ! -f "$restore_from" ]]; then
        echo "ERROR: Checkpoint not found: $restore_from"
        exit 1
    fi

    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONFAULTHANDLER=1
    export NVTE_MASKED_SOFTMAX_FUSION=0
    export NVTE_FLASH_ATTN=0
    export NVTE_FUSED_ATTN=0

    local wandb_args=""
    [[ "$WANDB" == "true" ]] && wandb_args="exp_manager.create_wandb_logger=true exp_manager.wandb_logger_kwargs.name=${stage_name}"

    python "$REPO_ROOT/finetune/train_sft.py" \
        --config-path "$REPO_ROOT/finetune/configs" \
        --config-name "$config_name" \
        trainer.devices="$GPUS" \
        trainer.num_nodes=1 \
        model.restore_from_path="$restore_from" \
        $wandb_args \
        2>&1 | tee "$log_file"

    log "SFT stage '$stage_name' complete"
}

# ── Stage 1: Chat SFT ──────────────────────────────────────────────────────────
if [[ "$STAGE" == "both" || "$STAGE" == "chat" ]]; then
    if [[ ! -f "/data/sft/chat/train.jsonl" ]]; then
        echo "ERROR: Chat SFT data not found at /data/sft/chat/train.jsonl"
        echo "  Run: make prepare-sft-data"
        exit 1
    fi
    run_sft "sft_chat" "sft_chat" "$PRETRAIN_CKPT"
fi

# ── Stage 2: Code SFT ──────────────────────────────────────────────────────────
if [[ "$STAGE" == "both" || "$STAGE" == "code" ]]; then
    CHAT_CKPT=$(find /results/slm_sft_chat -name "*.nemo" 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$CHAT_CKPT" ]]; then
        echo "ERROR: Chat SFT checkpoint not found in /results/slm_sft_chat/"
        echo "  Run chat SFT first: bash train_sft.sh --stage chat"
        exit 1
    fi
    log "Code SFT loading from chat checkpoint: $CHAT_CKPT"

    if [[ ! -f "/data/sft/code/train.jsonl" ]]; then
        echo "ERROR: Code SFT data not found at /data/sft/code/train.jsonl"
        echo "  Run: make prepare-sft-data"
        exit 1
    fi
    run_sft "sft_code" "sft_code" "$CHAT_CKPT"
fi

log "=== All SFT stages complete ==="
log "Checkpoints:"
log "  Chat: /results/slm_sft_chat/"
log "  Code: /results/slm_sft_code/"
log ""
log "Next step: make dpo"