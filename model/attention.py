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

from .config import SLMConfig


# ── RoPE ──────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Key design decisions:
    - inv_freq is computed from config and never saved to / loaded from
      checkpoints (persistent=False). _load_from_state_dict drops it.
    - cos/sin cache is NOT stored as a buffer at all. It is recomputed
      lazily on each forward call in float32, then cast to the query
      dtype at point of use. This avoids NaN from bfloat16 precision
      limits and survives model.bfloat16() / model.half() casts.
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

        # Cache stored as plain Python attributes (not buffers) so they are
        # never affected by model dtype casts (.bfloat16(), .half(), etc.)
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_len: int = 0

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Drop saved RoPE buffers — always recompute from config.
        for key in ["inv_freq", "cos_cached", "sin_cached"]:
            state_dict.pop(prefix + key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        """Recompute float32 cos/sin cache up to seq_len on device."""
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        # Store as plain attributes — immune to dtype casts
        self._cos_cache = emb.cos()  # float32
        self._sin_cache = emb.sin()  # float32
        self._cache_len = seq_len

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns float32 cos and sin matrices for positions [0, seq_len).
        Always float32 regardless of model dtype — cast at point of use.
        """
        dev = device or self.inv_freq.device
        if self._cos_cache is None or seq_len > self._cache_len or self._cos_cache.device != dev:
            self._build_cache(max(seq_len, self.max_position_embeddings), dev)
        return self._cos_cache[:seq_len], self._sin_cache[:seq_len]


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
    Apply RoPE rotation. cos/sin arrive as float32 and are cast to
    the query dtype here so the rotation matches the computation dtype.
    """
    cos = cos.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
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
        self.config = config
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

        self.rotary_emb = RotaryEmbedding(config)

    def _prepare_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        q: torch.Tensor,
        kv_len: int,
    ) -> Optional[torch.Tensor]:
        """
        Normalise attention_mask to 4D float additive format for SDPA.

        Handles:
            None  — no mask, caller uses is_causal=True
            2D (batch, kv_len) — padding mask (1=keep, 0=pad)
                During training: combined with causal mask → 4D additive
                During generation (q_len=1): only padding matters, no causal needed
            4D (batch, 1, q_len, kv_len) — full additive mask, dtype normalised
        """
        if attention_mask is None:
            return None

        bsz, q_len = q.shape[0], q.shape[2]

        if attention_mask.dim() == 2:
            if q_len == 1:
                # Generation step — single query token attending to all cached keys.
                # No causal mask needed (trivially causal). Just apply padding mask.
                pad = attention_mask[:, None, None, :].bool()  # (batch, 1, 1, kv_len)
                mask = torch.zeros(bsz, 1, 1, kv_len, dtype=q.dtype, device=q.device)
                return mask.masked_fill(~pad, float("-inf"))
            else:
                # Training / prefill — combine padding mask with causal mask
                causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device).tril()
                pad = attention_mask[:, None, None, :].bool()
                combined = causal.unsqueeze(0) & pad
                mask = torch.zeros(bsz, 1, q_len, kv_len, dtype=q.dtype, device=q.device)
                return mask.masked_fill(~combined, float("-inf"))

        if attention_mask.dtype == torch.bool:
            mask = torch.zeros_like(attention_mask, dtype=q.dtype)
            return mask.masked_fill(~attention_mask, float("-inf"))

        if attention_mask.dtype != q.dtype:
            return attention_mask.to(dtype=q.dtype)

        return attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE — pass device so cache is built on the right device
        kv_seq_len = k.shape[2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]
        cos, sin = self.rotary_emb(kv_seq_len, device=q.device)
        offset = kv_seq_len - q_len
        q, k = apply_rotary_emb(q, k, cos[offset:offset + q_len], sin[offset:offset + q_len])

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v) if use_cache else None

        # GQA: expand KV heads
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_query_groups, dim=1)
            v = v.repeat_interleave(self.num_query_groups, dim=1)

        # Attention mask
        # During training: use padding mask combined with causal mask.
        # During inference: always use is_causal=True and ignore the attention
        # mask — transformers v5 passes various mask formats during generation
        # that conflict with our custom mask handling. Causal masking is always
        # correct for decoder-only generation.
        if self.training and attention_mask is not None:
            attn_mask = self._prepare_mask(attention_mask, q, k.shape[2])
            is_causal = False
        else:
            attn_mask = None
            is_causal = True

        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value