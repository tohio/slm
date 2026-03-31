#!/usr/bin/env bash
# serve/serve.sh
# ---------------
# Launch a vLLM OpenAI-compatible server for SLM.
#
# Usage:
#   ./serve/serve.sh                          # serve slm-125m on port 8000
#   ./serve/serve.sh --model tohio/slm-350m   # serve from Hub
#   ./serve/serve.sh --port 8080              # custom port
#   MODEL=tohio/slm-1b ./serve/serve.sh       # via env var

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="${MODEL:-tohio/slm-125m}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --host)      HOST="$2"; shift 2 ;;
        --tp)        TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --max-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=================================="
echo "  SLM vLLM Server"
echo "=================================="
echo "  Model:       $MODEL"
echo "  Host:        $HOST:$PORT"
echo "  Max length:  $MAX_MODEL_LEN"
echo "  GPU util:    $GPU_MEMORY_UTILIZATION"
echo "  TP size:     $TENSOR_PARALLEL_SIZE"
echo "  Dtype:       $DTYPE"
echo "=================================="

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --served-model-name "${MODEL##*/}" \
    --trust-remote-code