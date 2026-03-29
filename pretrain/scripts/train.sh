#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# pretrain/scripts/train.sh
# Launch NeMo GPT pre-training.
#
# Handles:
#   - Single and multi-GPU launch via torchrun
#   - Config selection (125M / 350M / 1B)
#   - Automatic resume from latest checkpoint
#   - Logging to file + optional W&B
#
# Usage:
#   bash train.sh                                    # default: 125M, 1 GPU
#   bash train.sh --config configs/gpt_350m.yaml     # 350M
#   bash train.sh --config configs/gpt_1b.yaml       # 1B
#   bash train.sh --config configs/gpt_125m.yaml --gpus 4
#   bash train.sh --config configs/gpt_125m.yaml --resume /results/slm_gpt_125m/checkpoints/last.ckpt
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/../configs/gpt_125m.yaml"
GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
RESUME=""
WANDB=false

# Changed from /logs/pretrain → /results/pretrain_logs
# /logs is not reliably bind-mounted; /results always is
LOG_DIR="/results/pretrain_logs"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIG="$2";  shift 2 ;;
        --gpus)    GPUS="$2";    shift 2 ;;
        --resume)  RESUME="$2";  shift 2 ;;
        --wandb)   WANDB=true;   shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CONFIG_NAME=$(basename "$CONFIG" .yaml)
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${CONFIG_NAME}_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== SLM Pre-Training ==="
log "Config:    $CONFIG"
log "GPUs:      $GPUS"
log "Log:       $LOG_FILE"

# ── Validate prerequisites ────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# Check dataset files exist
# Tokenized output is written to /data/curated/tokenized/ by make tokenizer
PRETRAIN_DIR="/data/curated/tokenized"
BIN_COUNT=$(find "$PRETRAIN_DIR" -name "*.bin" 2>/dev/null | wc -l)
if [[ "$BIN_COUNT" -eq 0 ]]; then
    echo "ERROR: No .bin dataset files found in $PRETRAIN_DIR"
    echo "  Run: make tokenizer"
    exit 1
fi
log "Dataset:   $BIN_COUNT .bin file(s) in $PRETRAIN_DIR"

# Check tokenizer
TOKENIZER="/data/tokenizer/slm_tokenizer.model"
if [[ ! -f "$TOKENIZER" ]]; then
    echo "ERROR: Tokenizer not found: $TOKENIZER"
    echo "  Run: make tokenizer"
    exit 1
fi
log "Tokenizer: $TOKENIZER"

# ── Build override args ───────────────────────────────────────────────────────
OVERRIDES=(
    "trainer.devices=$GPUS"
)

# Resume from checkpoint if specified or auto-detect latest
if [[ -n "$RESUME" ]]; then
    OVERRIDES+=("model.resume_from_checkpoint=$RESUME")
    log "Resuming from: $RESUME"
elif [[ -d "/results/${CONFIG_NAME}/checkpoints" ]]; then
    LATEST=$(find "/results/${CONFIG_NAME}/checkpoints" -name "*.nemo" -o -name "*.ckpt" | sort | tail -1)
    if [[ -n "$LATEST" ]]; then
        OVERRIDES+=("model.resume_from_checkpoint=$LATEST")
        log "Auto-resuming from: $LATEST"
    fi
fi

# Enable W&B if requested
if [[ "$WANDB" == "true" ]]; then
    OVERRIDES+=("exp_manager.create_wandb_logger=true")
    log "W&B logging enabled"
fi

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1       # required for Megatron tensor parallelism
export TOKENIZERS_PARALLELISM=false        # avoid HuggingFace tokenizer warnings
export PYTHONFAULTHANDLER=1                # better crash traces
export NCCL_DEBUG=WARN                     # suppress verbose NCCL logs (set INFO to debug)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # reduce memory fragmentation

# For multi-GPU: ensure NCCL uses NVLink if available
if [[ "$GPUS" -gt 1 ]]; then
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=2
fi

# ── Launch ────────────────────────────────────────────────────────────────────
log "Launching training..."
log "Overrides: ${OVERRIDES[*]}"

# Build override string for NeMo/Hydra
OVERRIDE_STR="${OVERRIDES[*]}"

if [[ "$GPUS" -eq 1 ]]; then
    # Single GPU — direct python launch
    python "$REPO_ROOT/pretrain/train.py" \
        --config-path "$(dirname "$CONFIG")" \
        --config-name "$(basename "$CONFIG" .yaml)" \
        $OVERRIDE_STR \
        2>&1 | tee -a "$LOG_FILE"
else
    # Multi-GPU — torchrun for DDP
    torchrun \
        --nproc_per_node="$GPUS" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        "$REPO_ROOT/pretrain/train.py" \
        --config-path "$(dirname "$CONFIG")" \
        --config-name "$(basename "$CONFIG" .yaml)" \
        $OVERRIDE_STR \
        2>&1 | tee -a "$LOG_FILE"
fi

log "Training complete. Results in /results/${CONFIG_NAME}/"