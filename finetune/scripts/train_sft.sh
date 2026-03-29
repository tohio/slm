#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# finetune/scripts/train_sft.sh
# Launch NeMo Aligner Supervised Fine-Tuning (SFT).
#
# Runs sequentially:
#   Stage 1: SFT on general chat data  (OpenAssistant / Dolly)
#   Stage 2: SFT on coding data        (CodeSearchNet / The Stack subset)
#
# Each stage loads from the previous stage's best checkpoint.
# answer_only_loss=true means loss is computed only on assistant responses.
#
# Usage:
#   bash train_sft.sh                          # run both stages
#   bash train_sft.sh --stage chat             # chat SFT only
#   bash train_sft.sh --stage code             # code SFT only (needs chat ckpt)
#   bash train_sft.sh --stage chat --gpus 4
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAGE="both"
GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
WANDB=false
PRETRAIN_CKPT=""    # override pretrain checkpoint path

# Changed from /logs/sft → /results/sft_logs
# /logs is not reliably bind-mounted; /results always is
LOG_DIR="/results/sft_logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)          STAGE="$2";          shift 2 ;;
        --gpus)           GPUS="$2";           shift 2 ;;
        --pretrain-ckpt)  PRETRAIN_CKPT="$2";  shift 2 ;;
        --wandb)          WANDB=true;           shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Locate pre-training checkpoint ───────────────────────────────────────────
find_latest_ckpt() {
    local results_dir="$1"
    find "$results_dir" -name "*.nemo" | sort | tail -1
}

if [[ -z "$PRETRAIN_CKPT" ]]; then
    PRETRAIN_CKPT=$(find_latest_ckpt "/results/slm_gpt_125m/checkpoints")
    if [[ -z "$PRETRAIN_CKPT" ]]; then
        echo "ERROR: No pre-training checkpoint found in /results/slm_gpt_125m/"
        echo "  Run pretrain/scripts/train.sh first, or pass --pretrain-ckpt <path>"
        exit 1
    fi
fi

log "Pre-train checkpoint: $PRETRAIN_CKPT"

# ── Generic SFT launcher ──────────────────────────────────────────────────────
run_sft() {
    local stage_name="$1"
    local config="$2"
    local restore_from="$3"
    local log_file="$LOG_DIR/${stage_name}_${TIMESTAMP}.log"

    log "--- Starting SFT Stage: $stage_name ---"
    log "Config:       $config"
    log "Restore from: $restore_from"
    log "GPUs:         $GPUS"
    log "Log:          $log_file"

    # Validate inputs
    if [[ ! -f "$config" ]]; then
        echo "ERROR: Config not found: $config"; exit 1
    fi
    if [[ ! -f "$restore_from" ]]; then
        echo "ERROR: Checkpoint not found: $restore_from"; exit 1
    fi

    local overrides=(
        "trainer.devices=$GPUS"
        "model.restore_from_path=$restore_from"
    )
    if [[ "$WANDB" == "true" ]]; then
        overrides+=("exp_manager.create_wandb_logger=true")
        overrides+=("exp_manager.wandb_logger_kwargs.name=${stage_name}")
    fi

    local override_str="${overrides[*]}"

    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONFAULTHANDLER=1

    if [[ "$GPUS" -eq 1 ]]; then
        python "$REPO_ROOT/finetune/train_sft.py" \
            --config-path "$(dirname "$config")" \
            --config-name "$(basename "$config" .yaml)" \
            $override_str \
            2>&1 | tee "$log_file"
    else
        torchrun \
            --nproc_per_node="$GPUS" \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=localhost \
            --master_port=29501 \
            "$REPO_ROOT/finetune/train_sft.py" \
            --config-path "$(dirname "$config")" \
            --config-name "$(basename "$config" .yaml)" \
            $override_str \
            2>&1 | tee "$log_file"
    fi

    log "SFT stage '$stage_name' complete"
}

# ── Stage 1: Chat SFT ─────────────────────────────────────────────────────────
if [[ "$STAGE" == "both" || "$STAGE" == "chat" ]]; then
    CHAT_CONFIG="$REPO_ROOT/finetune/configs/sft_chat.yaml"

    if [[ ! -f "/data/sft/chat/train.jsonl" ]]; then
        echo "ERROR: Chat SFT training data not found at /data/sft/chat/train.jsonl"
        echo "  Run: python finetune/data/prepare_sft.py --stage chat"
        exit 1
    fi

    run_sft "sft_chat" "$CHAT_CONFIG" "$PRETRAIN_CKPT"
fi

# ── Stage 2: Code SFT ─────────────────────────────────────────────────────────
if [[ "$STAGE" == "both" || "$STAGE" == "code" ]]; then
    CODE_CONFIG="$REPO_ROOT/finetune/configs/sft_code.yaml"

    CHAT_CKPT=$(find_latest_ckpt "/results/slm_sft_chat/checkpoints")
    if [[ -z "$CHAT_CKPT" ]]; then
        echo "ERROR: Chat SFT checkpoint not found in /results/slm_sft_chat/"
        echo "  Run chat SFT first: bash train_sft.sh --stage chat"
        exit 1
    fi
    log "Code SFT loading from chat checkpoint: $CHAT_CKPT"

    if [[ ! -f "/data/sft/code/train.jsonl" ]]; then
        echo "ERROR: Code SFT training data not found at /data/sft/code/train.jsonl"
        echo "  Run: python finetune/data/prepare_sft.py --stage code"
        exit 1
    fi

    run_sft "sft_code" "$CODE_CONFIG" "$CHAT_CKPT"
fi

log "=== All SFT stages complete ==="
log "Checkpoints:"
log "  Chat: /results/slm_sft_chat/checkpoints/"
log "  Code: /results/slm_sft_code/checkpoints/"
log ""
log "Next step: bash alignment/scripts/train_dpo.sh"