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
#                        --input /results/slm_gpt_125m/checkpoints/last.nemo \
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
        --precision)  PRECISION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$DIRECTION" || -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --direction [nemo_to_hf|hf_to_nemo] --input <path> --output <path>"
    exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$OUTPUT"

# ── NeMo → HuggingFace ───────────────────────────────────────────────────────
if [[ "$DIRECTION" == "nemo_to_hf" ]]; then
    log "Converting NeMo checkpoint → HuggingFace"
    log "  Input:  $INPUT"
    log "  Output: $OUTPUT"

    # NeMo 2.x uses a dedicated conversion script in megatron-core
    # The script handles weight remapping from Megatron layout to HF layout
    python -c "
import torch
import os
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from omegaconf import open_dict

print('Loading NeMo checkpoint from: $INPUT')
model = MegatronGPTModel.restore_from(
    restore_path='$INPUT',
    map_location=torch.device('cpu'),
)
model.eval()

# NeMo 2.x export via to_huggingface_checkpoint
print('Exporting to HuggingFace format...')
hf_config, hf_state_dict = model.to_huggingface_checkpoint()

# Save config
hf_config.save_pretrained('$OUTPUT')

# Save weights
torch.save(hf_state_dict, os.path.join('$OUTPUT', 'pytorch_model.bin'))

# Save tokenizer if available
tokenizer_path = model.cfg.tokenizer.get('model', None)
if tokenizer_path and os.path.exists(tokenizer_path):
    import shutil
    shutil.copy(tokenizer_path, '$OUTPUT')
    print(f'Tokenizer copied to $OUTPUT')

print('HuggingFace model saved to: $OUTPUT')
print('Files:', os.listdir('$OUTPUT'))
"

# ── HuggingFace → NeMo ───────────────────────────────────────────────────────
elif [[ "$DIRECTION" == "hf_to_nemo" ]]; then
    log "Converting HuggingFace model → NeMo"
    log "  Input:     $INPUT"
    log "  Output:    $OUTPUT"
    log "  TP size:   $TP_SIZE"
    log "  PP size:   $PP_SIZE"
    log "  Precision: $PRECISION"

    # NeMo 2.x conversion script location
    # For GPT-style models (GPT-2, LLaMA, Mistral etc.)
    CONVERT_SCRIPT=$(python -c "
import nemo, os
nemo_dir = os.path.dirname(nemo.__file__)
candidates = [
    os.path.join(nemo_dir, 'collections/nlp/models/language_modeling/megatron/gpt/convert_hf_to_nemo.py'),
    os.path.join(nemo_dir, 'collections/nlp/models/language_modeling/convert_hf_to_nemo.py'),
]
for c in candidates:
    if os.path.exists(c):
        print(c)
        break
else:
    print('NOT_FOUND')
")

    if [[ "$CONVERT_SCRIPT" == "NOT_FOUND" ]]; then
        log "WARNING: NeMo HF→NeMo conversion script not found at expected paths."
        log "  Falling back to manual Python conversion..."

        python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

print('Loading HuggingFace model from: $INPUT')
hf_model = AutoModelForCausalLM.from_pretrained('$INPUT', torch_dtype=torch.bfloat16)
hf_tokenizer = AutoTokenizer.from_pretrained('$INPUT')

print('NOTE: Manual HF→NeMo conversion requires architecture-specific weight remapping.')
print('For supported models (LLaMA, Mistral), use the NeMo conversion scripts:')
print('  https://github.com/NVIDIA/NeMo/tree/main/scripts/nlp_language_modeling')
print('For a custom GPT model trained from scratch in NeMo, use nemo_to_hf direction instead.')
"
    else
        log "Using NeMo conversion script: $CONVERT_SCRIPT"
        python "$CONVERT_SCRIPT" \
            --input_name_or_path "$INPUT" \
            --output_path "$OUTPUT" \
            --precision "$PRECISION" \
            --tensor_parallelism_size "$TP_SIZE" \
            --pipeline_parallelism_size "$PP_SIZE"
    fi

else
    echo "ERROR: Unknown direction '$DIRECTION'. Use 'nemo_to_hf' or 'hf_to_nemo'"
    exit 1
fi

log "Conversion complete → $OUTPUT"
log "Output contents:"
ls -lh "$OUTPUT" 2>/dev/null || true