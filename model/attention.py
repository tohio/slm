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
        past_len: int,
    ) -> Optional[torch.Tensor]:
        """
        Build a 4D additive float mask for SDPA that combines (as needed):
          - causal masking with an offset for populated KV cache
          - 2D padding masks (batch, kv_len) where 1=keep, 0=pad
          - pass-through for already-4D additive or boolean masks

        Args:
            attention_mask: one of:
                None — caller only needs causal (see past_len logic)
                2D (batch, kv_len) — padding mask, 1=keep, 0=pad
                4D (batch, 1, q_len, kv_len) — full additive mask
                    (dtype normalised to q.dtype, returned as-is if already float)
            q: query tensor (batch, n_heads, q_len, head_dim), used for
               batch/q_len/dtype/device.
            kv_len: total key/value length (past + current).
            past_len: number of cached key positions before this call.
                      0 during training or fresh forward passes.
                      >0 during generation with an existing cache.

        Returns:
            None if no masking is needed (caller should use is_causal=True
            with SDPA), otherwise a (batch, 1, q_len, kv_len) additive mask
            in q.dtype with -inf at masked positions.

        The offset-aware causal mask handles the case where q_len < kv_len
        (multi-token prefill on top of a populated cache). SDPA's built-in
        `is_causal=True` only works when q_len == kv_len.
        """
        bsz, _, q_len, _ = q.shape
        device = q.device
        dtype = q.dtype

        # Fast path: already a 4D additive mask — just normalise dtype.
        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                m = torch.zeros_like(attention_mask, dtype=dtype)
                return m.masked_fill(~attention_mask, float("-inf"))
            if attention_mask.dtype != dtype:
                return attention_mask.to(dtype=dtype)
            return attention_mask

        # Build causal component. Needed whenever q_len > 1.
        # With past_len > 0, row i of the causal mask allows attending to
        # columns 0 .. past_len + i (inclusive), i.e. tril with diagonal=past_len.
        causal: Optional[torch.Tensor] = None
        if q_len > 1:
            causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(
                diagonal=past_len
            )

        # Build padding component from 2D mask if provided.
        # Shape: (batch, 1, 1, kv_len) — broadcasts across q_len.
        #
        # HF `generate` passes a 2D mask of length (past_len + q_len) covering
        # the full conceptual sequence, but `kv_len` reflects what's actually
        # in the cache + current step. These can diverge in edge cases (e.g.
        # the cache gets reset between decode steps, or use_cache=False is
        # flipped mid-generation). Align to kv_len by taking the rightmost
        # kv_len entries — this preserves causal alignment because the most
        # recent mask values correspond to the most recent kv positions.
        pad: Optional[torch.Tensor] = None
        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.shape[1] != kv_len:
                attention_mask = attention_mask[:, -kv_len:]
            pad = attention_mask[:, None, None, :].bool()

        # No masking needed at all (q_len == 1 and no padding mask).
        if causal is None and pad is None:
            return None

        # Combine causal + padding. Either may be None.
        if causal is not None and pad is not None:
            combined = causal.unsqueeze(0).unsqueeze(0) & pad  # (batch, 1, q_len, kv_len)
        elif causal is not None:
            combined = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, q_len, kv_len)
        else:
            # pad only — applies across all q_len rows (used when q_len == 1
            # during batched generation with left-padded prompts).
            combined = pad.expand(bsz, 1, q_len, kv_len)

        mask = torch.zeros(bsz, 1, q_len, kv_len, dtype=dtype, device=device)
        return mask.masked_fill(~combined, float("-inf"))

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

        # RoPE — pass device so cache is built on the right device.
        # past_len is the number of positions already in the cache; the new
        # tokens start at that offset.
        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        kv_seq_len = k.shape[2] + past_len
        cos, sin = self.rotary_emb(kv_seq_len, device=q.device)
        q, k = apply_rotary_emb(
            q, k,
            cos[past_len:past_len + q_len],
            sin[past_len:past_len + q_len],
        )

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v) if use_cache else None

        # GQA: expand KV heads
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_query_groups, dim=1)
            v = v.repeat_interleave(self.num_query_groups, dim=1)

        # Mask handling.
        #
        # We build an explicit mask whenever ANY of the following is true:
        #   - a padding/additive mask was passed in (batched training, or
        #     batched generation with left-padded prompts)
        #   - we have an existing cache AND q_len > 1 (multi-token prefill
        #     on top of cached state — SDPA's is_causal=True assumes
        #     q_len == kv_len and would apply the mask at the wrong offset)
        #
        # Otherwise (single-sequence, no cache OR q_len == 1 with no padding
        # mask) we can rely on SDPA's is_causal=True fast path.
        kv_len = k.shape[2]
        needs_explicit_mask = (
            attention_mask is not None
            or (past_len > 0 and q_len > 1)
        )
        if needs_explicit_mask:
            attn_mask = self._prepare_mask(attention_mask, q, kv_len, past_len)
            is_causal = False
        else:
            attn_mask = None
            is_causal = q_len > 1  # q_len == 1 needs no causal mask at all

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