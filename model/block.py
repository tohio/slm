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
        out, _ = block(x)  # shape: (2, 512, 768)
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
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional causal mask
            past_key_value: Optional KV cache from previous steps
            use_cache: Whether to return updated KV cache

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
            past_key_value: Updated KV cache if use_cache else None
        """
        # ── Attention sub-layer ────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # ── MLP sub-layer ──────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value