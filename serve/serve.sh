#!/usr/bin/env bash
# serve/serve.sh
# ---------------
# Launch a vLLM OpenAI-compatible server for SLM.
#
# Usage:
#   ./serve/serve.sh                                    # serve slm-125m on port 8000
#   ./serve/serve.sh --model tohio/slm-350m             # serve from Hub
#   ./serve/serve.sh --model results/slm-125m-dpo/final # serve local checkpoint
#   ./serve/serve.sh --port 8080                        # custom port
#   MODEL=tohio/slm-1b ./serve/serve.sh                 # via env var

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

MODEL="${MODEL:-tohio/slm-125m}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
# MAX_MODEL_LEN intentionally unset by default — vLLM introspects the max
# context length from the model's config.json (max_position_embeddings).
# Override only to explicitly cap shorter than the trained context:
#   MAX_MODEL_LEN=1024 ./serve/serve.sh
# Don't set it to 2048 "as a safe default" — that silently truncates the 1B
# variant's 4096 context back to 2048.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"

# ── Parse args ────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)   MODEL="$2";   shift 2 ;;
    --port)    PORT="$2";    shift 2 ;;
    --host)    HOST="$2";    shift 2 ;;
    --tp)      TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Derive served model name ───────────────────────────────────────────────────
#
# Locality is determined by whether the path resolves to an existing
# directory — NOT by whether it contains a slash or starts with "results/".
# A pattern-based check mis-classifies absolute paths like
# /data/slm/results/... (contains slashes, doesn't start with "results/")
# as Hub IDs.
#
# For Hub IDs (e.g. tohio/slm-125m): strip the owner prefix -> slm-125m.
# For local paths: use the parent directory name if the leaf is "final"
# (our checkpoint-dir convention), otherwise use the leaf.

if [[ -d "$MODEL" ]]; then
  # Local directory
  LEAF="${MODEL%/}"      # strip trailing slash if any
  LEAF="${LEAF##*/}"
  if [[ "$LEAF" == "final" ]]; then
    PARENT="${MODEL%/}"
    PARENT="${PARENT%/final}"
    SERVED_MODEL_NAME="${PARENT##*/}"
  else
    SERVED_MODEL_NAME="$LEAF"
  fi
else
  # Not a local directory — treat as Hub ID. Strip owner prefix if present.
  if [[ "$MODEL" == *"/"* ]]; then
    SERVED_MODEL_NAME="${MODEL##*/}"
  else
    SERVED_MODEL_NAME="$MODEL"
  fi
fi

# ── Tensor parallelism warning ─────────────────────────────────────────────────
# --tp > 1 uses vLLM's built-in tensor parallelism to split the model across
# multiple GPUs. This is independent of the training pipeline — no code changes
# are needed in the SLM codebase, but you must have N GPUs available and vLLM
# installed with the appropriate NCCL backend.
# For 125m and 350m, TP is unnecessary — both fit on a single GPU.
# For 1b with 4096 context, TP=2 may be needed on GPUs with less than 40GB VRAM.

if [[ "$TENSOR_PARALLEL_SIZE" -gt 1 ]]; then
  echo "NOTE: --tp $TENSOR_PARALLEL_SIZE requires $TENSOR_PARALLEL_SIZE GPUs available."
  echo "      vLLM handles model splitting — no changes needed in the SLM codebase."
fi

echo "=================================="
echo "  SLM vLLM Server"
echo "=================================="
echo "  Model:       $MODEL"
echo "  Served as:   $SERVED_MODEL_NAME"
echo "  Host:        $HOST:$PORT"
echo "  Max length:  ${MAX_MODEL_LEN:-<from model config>}"
echo "  GPU util:    $GPU_MEMORY_UTILIZATION"
echo "  TP size:     $TENSOR_PARALLEL_SIZE"
echo "  Dtype:       $DTYPE"
echo "=================================="

# Build the argv dynamically so --max-model-len is only passed when set —
# otherwise vLLM will use max_position_embeddings from the model config.
VLLM_ARGS=(
  --model                  "$MODEL"
  --host                   "$HOST"
  --port                   "$PORT"
  --dtype                  "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --tensor-parallel-size   "$TENSOR_PARALLEL_SIZE"
  --served-model-name      "$SERVED_MODEL_NAME"
  --trust-remote-code
)
if [[ -n "$MAX_MODEL_LEN" ]]; then
  VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"