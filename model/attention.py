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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

from .config import SLMConfig


# ── RoPE ──────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Key design decisions:
    - inv_freq is computed from config and never saved to / loaded from
      checkpoints (persistent=False). _load_from_state_dict drops it.
    - cos/sin are computed once per model forward and shared by every layer.
    - frequency computation stays in float32 before being cast back to the
      model dtype. This avoids reduced-precision RoPE position errors.
    """

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta

        # Stored only for device tracking — recomputed from config on load
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Drop saved RoPE buffers — always recompute from config.
        for key in ["inv_freq", "cos_cached", "sin_cached"]:
            state_dict.pop(prefix + key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return RoPE cos/sin tensors shaped ``(batch, seq_len, head_dim)``."""
        inv_freq = self.inv_freq.to(device=hidden_states.device, dtype=torch.float32)
        positions = position_ids.to(device=hidden_states.device, dtype=torch.float32)
        freqs = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
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
    Apply RoPE rotation. RotaryEmbedding computes the trigonometric values
    in float32, then returns them in the hidden-state dtype. Normalize them to
    the query dtype here in case query/key tensors use a different dtype.
    """
    cos = cos.to(dtype=q.dtype).unsqueeze(1)
    sin = sin.to(dtype=q.dtype).unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ── GQA ───────────────────────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE.

    At num_kv_heads == num_heads: standard Multi-Head Attention (MHA)
    At num_kv_heads == 1: Multi-Query Attention (MQA)
    """

    def __init__(self, config: SLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_query_groups = config.num_query_groups
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided")
        cos, sin = position_embeddings
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Cache objects update in place and preserve DynamicCache/StaticCache
        # semantics expected by Transformers generation and compilation.
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # create_causal_mask() prepares an offset-aware 4D mask whenever one
        # is required. A None mask means SDPA may use its causal fast path.
        is_causal = attention_mask is None and q_len > 1

        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            # Native SDPA GQA avoids materialising repeated K/V heads and lets
            # CUDA dispatch directly to an eligible fused implementation.
            enable_gqa=self.num_kv_heads != self.num_heads,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output
