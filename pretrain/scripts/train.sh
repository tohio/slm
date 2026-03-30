#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# pretrain/scripts/train.sh
# Launch NeMo 2.x GPT pre-training.
#
# Handles:
#   - Single and multi-GPU launch via torchrun
#   - Model size selection (125M / 350M / 1B) via --size flag
#   - Automatic resume from latest checkpoint
#   - Logging to file + optional W&B
#
# Usage:
#   bash train.sh                          # default: 125M, 1 GPU
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
LOG_DIR="/results/pretrain_logs"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)    SIZE="$2";   shift 2 ;;
        --gpus)    GPUS="$2";   shift 2 ;;
        --wandb)   WANDB=true;  shift ;;
        --resume)  RESUME=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/gpt_${SIZE}_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== SLM Pre-Training (NeMo 2.x) ==="
log "Size:      $SIZE"
log "GPUs:      $GPUS"
log "Log:       $LOG_FILE"

# ── Model size configs ────────────────────────────────────────────────────────
case "$SIZE" in
    125m)
        NUM_LAYERS=12
        HIDDEN_SIZE=768
        FFN_HIDDEN_SIZE=3072
        NUM_ATTENTION_HEADS=12
        MICRO_BATCH_SIZE=4
        GLOBAL_BATCH_SIZE=32
        TP=1
        PP=1
        LR=3e-4
        MIN_LR=3e-5
        RESULTS_DIR="/results/slm_gpt_125m"
        ;;
    350m)
        NUM_LAYERS=24
        HIDDEN_SIZE=1024
        FFN_HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=16
        MICRO_BATCH_SIZE=4
        GLOBAL_BATCH_SIZE=128
        TP=1
        PP=1
        LR=2e-4
        MIN_LR=2e-5
        RESULTS_DIR="/results/slm_gpt_350m"
        ;;
    1b)
        NUM_LAYERS=32
        HIDDEN_SIZE=2048
        FFN_HIDDEN_SIZE=8192
        NUM_ATTENTION_HEADS=16
        MICRO_BATCH_SIZE=2
        GLOBAL_BATCH_SIZE=128
        TP=2
        PP=1
        LR=1e-4
        MIN_LR=1e-5
        RESULTS_DIR="/results/slm_gpt_1b"
        ;;
    *)
        echo "ERROR: Unknown size '$SIZE'. Valid: 125m, 350m, 1b"
        exit 1
        ;;
esac

# ── Validate prerequisites ────────────────────────────────────────────────────
PRETRAIN_DIR="/data/curated/tokenized"
BIN_COUNT=$(find "$PRETRAIN_DIR" -name "*.bin" 2>/dev/null | wc -l)
if [[ "$BIN_COUNT" -eq 0 ]]; then
    echo "ERROR: No .bin dataset files found in $PRETRAIN_DIR"
    echo "  Run: make tokenize"
    exit 1
fi
log "Dataset:   $BIN_COUNT .bin file(s) in $PRETRAIN_DIR"

TOKENIZER="/data/tokenizer/slm_tokenizer.model"
if [[ ! -f "$TOKENIZER" ]]; then
    echo "ERROR: Tokenizer not found: $TOKENIZER"
    echo "  Run: make tokenizer"
    exit 1
fi
log "Tokenizer: $TOKENIZER"

# ── Build args ────────────────────────────────────────────────────────────────
TRAIN_ARGS=(
    --num-layers          "$NUM_LAYERS"
    --hidden-size         "$HIDDEN_SIZE"
    --ffn-hidden-size     "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --tensor-model-parallel-size   "$TP"
    --pipeline-model-parallel-size "$PP"
    --gpus                "$GPUS"
    --micro-batch-size    "$MICRO_BATCH_SIZE"
    --global-batch-size   "$GLOBAL_BATCH_SIZE"
    --lr                  "$LR"
    --min-lr              "$MIN_LR"
    --results-dir         "$RESULTS_DIR"
    --tokenizer-model     "$TOKENIZER"
    --data-paths
        0.7 "$PRETRAIN_DIR/text_document"
        0.3 "$PRETRAIN_DIR/text_document"
)

[[ "$WANDB" == "true" ]] && TRAIN_ARGS+=(--wandb)
[[ "$RESUME" == "true" ]] && TRAIN_ARGS+=(--resume)

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

if [[ "$GPUS" -gt 1 ]]; then
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=2
fi

# ── Launch ────────────────────────────────────────────────────────────────────
log "Launching training..."

if [[ "$GPUS" -eq 1 ]]; then
    python "$REPO_ROOT/pretrain/train.py" "${TRAIN_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
else
    torchrun \
        --nproc_per_node="$GPUS" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        "$REPO_ROOT/pretrain/train.py" "${TRAIN_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
fi

log "Training complete. Results in $RESULTS_DIR"