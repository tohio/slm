"""
model/block.py
--------------
Transformer block — wires together RMSNorm, GQA, and SwiGLU.

Uses pre-normalization (Pre-LN): normalization is applied before
each sub-layer rather than after. Pre-LN significantly improves
training stability in deep networks by keeping gradient magnitudes
consistent across layers.

Block structure:
    x = x + Attention(RMSNorm(x))   ← attention residual
    x = x + MLP(RMSNorm(x))         ← MLP residual

This is the standard decoder block used by LLaMA, Mistral, Qwen,
and most modern transformer LLMs.

Reference:
    Pre-LN: Xiong et al. (2020) — https://arxiv.org/abs/2002.04745
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

from .attention import GroupedQueryAttention
from .config import SLMConfig
from .mlp import SwiGLUMLP
from .norm import RMSNorm


class SLMDecoderBlock(nn.Module):
    """
    Single transformer decoder block.

    Applies pre-norm before both attention and MLP sub-layers,
    with residual connections around each.

    Args:
        config (SLMConfig): Model configuration.
        layer_idx (int): Index of this layer in the stack.
            Passed to attention for KV cache management.

    Shape:
        Input:  (batch, seq_len, hidden_size)
        Output: (batch, seq_len, hidden_size)

    Example::

        config = SLMConfig()
        block = SLMDecoderBlock(config, layer_idx=0)
        x = torch.randn(2, 512, 768)
        # The full SLMModel supplies shared RoPE position embeddings.
    """

    def __init__(self, config: SLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional 4D causal/padding mask
            position_embeddings: Shared RoPE cosine and sine tensors
            past_key_values: Optional Transformers cache object
            use_cache: Whether the supplied cache is active

        Returns:
            hidden_states: (batch, seq_len, hidden_size). The cache, when
                present, is updated in place.
        """
        # ── Attention sub-layer ────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # ── MLP sub-layer ──────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
