"""
model/attention.py
------------------
Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE).

GQA reduces the number of key/value heads relative to query heads,
cutting KV cache memory at inference time while maintaining quality.
At the extreme (1 KV head) this becomes Multi-Query Attention (MQA).

RoPE encodes position information by rotating query and key vectors
in pairs of dimensions using a set of fixed frequencies. Unlike learned
absolute embeddings, RoPE generalizes naturally to unseen sequence lengths
and preserves relative position information in the attention dot product.

References:
    GQA: Ainslie et al. (2023) — https://arxiv.org/abs/2305.13245
    RoPE: Su et al. (2021) — https://arxiv.org/abs/2104.09864
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SLMConfig


# ── RoPE ──────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes cos/sin rotation matrices for positions up to
    max_position_embeddings. At forward time, slices to the actual
    sequence length.

    Args:
        config (SLMConfig): Model configuration.

    The inverse frequencies follow the original RoPE formula:
        theta_i = 1 / (base ^ (2i / head_dim))  for i in [0, head_dim/2)
    """

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta

        # Inverse frequencies — shape: (head_dim / 2,)
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for all positions
        self._build_cache(self.max_position_embeddings)

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, dtype=torch.float32)
        # Outer product: (seq_len, head_dim/2)
        freqs = torch.outer(positions, self.inv_freq)
        # Concatenate to get (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin matrices for positions [0, seq_len).

        Returns:
            cos: (seq_len, head_dim)
            sin: (seq_len, head_dim)
        """
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of the last dimension to implement RoPE.

    Splits x into two halves along the last dimension, negates the
    second half and swaps: [-x2, x1]. Combined with cos/sin application
    this produces the RoPE rotation.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE rotation to query and key tensors.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, kv_heads, seq_len, head_dim)
        cos: Cosine matrix of shape (seq_len, head_dim)
        sin: Sine matrix of shape (seq_len, head_dim)

    Returns:
        Rotated (q, k) tensors of the same shape.
    """
    # Broadcast cos/sin over batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ── GQA ───────────────────────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE.

    Query heads are grouped — each group of (num_heads / num_kv_heads)
    query heads shares a single key/value head. This reduces KV cache
    size at inference by a factor of (num_heads / num_kv_heads).

    At num_kv_heads == num_heads: standard Multi-Head Attention (MHA)
    At num_kv_heads == 1: Multi-Query Attention (MQA)

    No projection biases — consistent with the no-bias architecture.
    Uses scaled dot-product attention (F.scaled_dot_product_attention)
    which dispatches to FlashAttention when available.

    Args:
        config (SLMConfig): Model configuration.
        layer_idx (int): Layer index, used for KV cache management.
    """

    def __init__(self, config: SLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_query_groups = config.num_query_groups
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        # Projections — no bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(config)

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
            attention_mask: Optional causal mask (batch, 1, seq_len, seq_len)
            past_key_value: Optional cached (key, value) from previous steps
            use_cache: Whether to return updated KV cache

        Returns:
            output: (batch, seq_len, hidden_size)
            past_key_value: Updated KV cache if use_cache else None
        """
        bsz, q_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        kv_seq_len = k.shape[2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]
        cos, sin = self.rotary_emb(kv_seq_len)
        # Slice cos/sin for current positions only (handles KV cache offset)
        offset = kv_seq_len - q_len
        q_cos = cos[offset:offset + q_len]
        q_sin = sin[offset:offset + q_len]
        k_cos = cos[:kv_seq_len] if past_key_value is None else cos[offset:offset + q_len]
        k_sin = sin[:kv_seq_len] if past_key_value is None else sin[offset:offset + q_len]
        q, k = apply_rotary_emb(q, k, q_cos, q_sin) if past_key_value is None else apply_rotary_emb(q, k, cos[offset:offset+q_len], sin[offset:offset+q_len])

        # Append to KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v) if use_cache else None

        # Expand KV heads to match Q heads for GQA
        # (batch, kv_heads, seq_len, head_dim) → (batch, num_heads, seq_len, head_dim)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_query_groups, dim=1)
            v = v.repeat_interleave(self.num_query_groups, dim=1)

        # Scaled dot-product attention
        # Dispatches to FlashAttention 2 when available (torch >= 2.0)
        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=attention_mask is None,  # use causal mask if no explicit mask
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value