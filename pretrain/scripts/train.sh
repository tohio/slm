#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# pretrain/scripts/train.sh
# Launch NeMo 1.x GPT pre-training via megatron_gpt_pretraining.py.
#
# Saves checkpoints in .nemo format directly.
# No conversion step needed — NeMo-Aligner SFT/DPO loads .nemo directly.
#
# Usage:
#   bash train.sh                          # default: 125m, 1 GPU
#   bash train.sh --size 350m --gpus 4
#   bash train.sh --size 1b --gpus 4
#   bash train.sh --wandb
#   bash train.sh --resume
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
SIZE="125m"
GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
WANDB=false
RESUME=false
BASE_RESULTS_DIR="${RESULTS_DIR:-/results}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)   SIZE="$2";   shift 2 ;;
        --gpus)   GPUS="$2";   shift 2 ;;
        --wandb)  WANDB=true;  shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LOG_DIR="${BASE_RESULTS_DIR}/pretrain_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/gpt_${SIZE}_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== SLM Pre-Training (NeMo 1.x) ==="
log "Size:   $SIZE"
log "GPUs:   $GPUS"
log "Log:    $LOG_FILE"

# ── Config path ───────────────────────────────────────────────────────────────
CONFIG_PATH="$REPO_ROOT/pretrain/configs/gpt_${SIZE}.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    exit 1
fi

# ── Validate prerequisites ────────────────────────────────────────────────────
BIN_COUNT=$(find /data/curated/tokenized -name "*.bin" 2>/dev/null | wc -l)
if [[ "$BIN_COUNT" -eq 0 ]]; then
    echo "ERROR: No .bin dataset files found in /data/curated/tokenized"
    echo "  Run: make tokenize"
    exit 1
fi
log "Dataset:   $BIN_COUNT .bin file(s)"

if [[ ! -f "/data/tokenizer/slm_tokenizer.model" ]]; then
    echo "ERROR: Tokenizer not found: /data/tokenizer/slm_tokenizer.model"
    echo "  Run: make tokenizer"
    exit 1
fi
log "Tokenizer: /data/tokenizer/slm_tokenizer.model"

# ── Results dir override ──────────────────────────────────────────────────────
RESULTS_OVERRIDE="exp_manager.explicit_log_dir=${BASE_RESULTS_DIR}/slm_gpt_${SIZE}"

# ── W&B ───────────────────────────────────────────────────────────────────────
WANDB_OVERRIDES=""
if [[ "$WANDB" == "true" ]]; then
    WANDB_OVERRIDES="exp_manager.create_wandb_logger=true exp_manager.wandb_logger_kwargs.name=gpt_${SIZE}"
fi

# ── Resume ────────────────────────────────────────────────────────────────────
RESUME_OVERRIDES=""
if [[ "$RESUME" == "true" ]]; then
    RESUME_OVERRIDES="exp_manager.resume_if_exists=true exp_manager.resume_ignore_no_checkpoint=true"
fi

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── Launch ────────────────────────────────────────────────────────────────────
NEMO_SCRIPT="/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py"

log "Launching training..."

if [[ "$GPUS" -eq 1 ]]; then
    python3 "$NEMO_SCRIPT" \
        --config-path "$REPO_ROOT/pretrain/configs" \
        --config-name "gpt_${SIZE}" \
        trainer.devices="$GPUS" \
        trainer.num_nodes=1 \
        $RESULTS_OVERRIDE \
        $WANDB_OVERRIDES \
        $RESUME_OVERRIDES \
        2>&1 | tee -a "$LOG_FILE"
else
    torchrun \
        --nproc_per_node="$GPUS" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        "$NEMO_SCRIPT" \
        --config-path "$REPO_ROOT/pretrain/configs" \
        --config-name "gpt_${SIZE}" \
        trainer.devices="$GPUS" \
        trainer.num_nodes=1 \
        $RESULTS_OVERRIDE \
        $WANDB_OVERRIDES \
        $RESUME_OVERRIDES \
        2>&1 | tee -a "$LOG_FILE"
fi

log "Training complete."
log "Checkpoint: ${BASE_RESULTS_DIR}/slm_gpt_${SIZE}/"
log "Next step: make sft"