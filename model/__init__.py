"""
model/
------
SLM model package — custom decoder-only transformer architecture.

Public API:

    from model import SLMConfig, SLMForCausalLM, SLMModel
    from model import SLM_125M, SLM_350M, SLM_1B, CONFIGS

Register with HuggingFace AutoModel so the model can be loaded with:

    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    model = AutoModelForCausalLM.from_pretrained(
        "results/runs/125m/pretrain/final"
    )

Published Hub checkpoints use the native LlamaForCausalLM export contract and
do not require this registration.
"""

from .config import CONFIGS, SLM_125M, SLM_350M, SLM_1B, SLMConfig
from .model import SLMForCausalLM, SLMModel

__all__ = [
    "SLMConfig",
    "SLMModel",
    "SLMForCausalLM",
    "SLM_125M",
    "SLM_350M",
    "SLM_1B",
    "CONFIGS",
]
