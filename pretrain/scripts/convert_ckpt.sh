#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# pretrain/scripts/convert_ckpt.sh
# Convert checkpoints between NeMo and HuggingFace formats.
#
# Why conversion matters:
#   - NeMo .nemo checkpoints are needed for NeMo Aligner (SFT, DPO)
#   - HuggingFace format is needed for evaluation with lm-evaluation-harness,
#     or for serving with vLLM / TGI
#
# Directions:
#   nemo_to_hf  — after pre-training, export for evaluation/inference
#   hf_to_nemo  — import a public HF model to continue training in NeMo
#
# Usage:
#   bash convert_ckpt.sh --direction nemo_to_hf \
#                        --input /results/slm_gpt_125m/checkpoints/last.ckpt \
#                        --output /results/slm_gpt_125m_hf/
#
#   bash convert_ckpt.sh --direction hf_to_nemo \
#                        --input /models/llama-3.2-1b \
#                        --output /results/llama_nemo.nemo
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DIRECTION=""
INPUT=""
OUTPUT=""
TP_SIZE=1       # tensor parallel size used during training
PP_SIZE=1       # pipeline parallel size used during training
PRECISION="bf16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --direction)  DIRECTION="$2"; shift 2 ;;
        --input)      INPUT="$2";     shift 2 ;;
        --output)     OUTPUT="$2";    shift 2 ;;
        --tp)         TP_SIZE="$2";   shift 2 ;;
        --pp)         PP_SIZE="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$DIRECTION" || -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --direction [nemo_to_hf|hf_to_nemo] --input <path> --output <path>"
    exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$OUTPUT"

if [[ "$DIRECTION" == "nemo_to_hf" ]]; then
    log "Converting NeMo checkpoint → HuggingFace"
    log "  Input:  $INPUT"
    log "  Output: $OUTPUT"

    python - <<EOF
import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

# Load NeMo model
print("Loading NeMo checkpoint...")
model = MegatronGPTModel.restore_from(
    restore_path="$INPUT",
    map_location=torch.device("cpu"),
)
model.eval()

# Export to HuggingFace format
print("Exporting to HuggingFace format...")
model.save_to_hf("$OUTPUT")
print(f"HuggingFace model saved to $OUTPUT")
EOF

elif [[ "$DIRECTION" == "hf_to_nemo" ]]; then
    log "Converting HuggingFace model → NeMo"
    log "  Input:  $INPUT"
    log "  Output: $OUTPUT"

    # NeMo provides conversion scripts for common models
    python -m nemo.collections.nlp.models.language_modeling.megatron_gpt_model \
        convert_hf_to_nemo \
        --input_name_or_path "$INPUT" \
        --output_path "$OUTPUT" \
        --precision "$PRECISION" \
        --tensor_parallelism_size "$TP_SIZE" \
        --pipeline_parallelism_size "$PP_SIZE"

else
    echo "ERROR: Unknown direction '$DIRECTION'. Use 'nemo_to_hf' or 'hf_to_nemo'"
    exit 1
fi

log "Conversion complete → $OUTPUT"
