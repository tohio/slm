#!/usr/bin/env python3
"""
Run lm-evaluation-harness with the local SLM architecture pre-registered.

lm-eval-harness calls AutoConfig.from_pretrained internally, which doesn't
know about model_type='slm'. This wrapper registers SLMConfig and
SLMForCausalLM with the Auto* registries before invoking lm_eval, so the
CLI works against SLM checkpoints without trust_remote_code.

Usage:
    python scripts/run_lm_eval.py \\
      --model hf \\
      --model_args pretrained=results/slm-125m-chat-code/final,dtype=bfloat16 \\
      --tasks humaneval \\
      --num_fewshot 0 \\
      --batch_size 1 \\
      --apply_chat_template \\
      --output_path results/eval/debug_humaneval \\
      --log_samples \\
      --limit 5
"""

import sys
from pathlib import Path

# Make repo root importable even when launched from another directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoConfig, AutoModelForCausalLM

from model.config import SLMConfig
from model.model import SLMForCausalLM

# Register custom SLM architecture before lm_eval calls AutoConfig/AutoModel.
AutoConfig.register("slm", SLMConfig)
AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

from lm_eval.__main__ import cli_evaluate


if __name__ == "__main__":
    cli_evaluate()