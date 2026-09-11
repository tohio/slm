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

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import Cache
from transformers.integrations import use_kernel_func_from_hub, use_kernelized_func
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.integrations.sdpa_attention import repeat_kv, use_gqa_in_sdpa

from .config import SLMConfig


# ── RoPE ──────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Key design decisions:
    - inv_freq is rebuilt from config in FP32 on CPU, then copied to the
      destination device. Device moves must not change its rounded values.
      It is never saved to / loaded from checkpoints (persistent=False).
    - cos/sin are computed once per model forward and shared by every layer.
    - frequency computation stays in float32 before being cast back to the
      model dtype. This avoids reduced-precision RoPE position errors.
    """

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, device=None) -> None:
        """Rebuild parameter-free state; never initialize a learned tensor.

        HF's low-memory loader may materialize non-persistent buffers with
        empty storage. They have no checkpoint entry to restore them from.
        This method is also used after dtype/device conversion so BF16 cannot
        permanently round the frequencies that must be computed in FP32.
        Build on CPU even for a CUDA destination: recomputing the power on a
        different device can perturb the constants and break optimizer parity
        with a native Llama initialized on CPU and then moved to CUDA.
        """
        if device is None:
            device = (
                self.inv_freq.device
                if hasattr(self, "inv_freq")
                else torch.get_default_device()
            )
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.int64, device="cpu")
                .to(dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq.to(device=device), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.reset_parameters()
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Drop saved RoPE buffers — always recompute from config.
        for key in ["inv_freq", "cos_cached", "sin_cached"]:
            state_dict.pop(prefix + key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.reset_parameters()

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


@use_kernel_func_from_hub("rotary_pos_emb")
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

@use_kernelized_func(apply_rotary_emb)
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
        self.is_causal = True

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
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        if self.config._attn_implementation not in (None, "sdpa", "flash_attention_3"):
            raise ValueError("SLM attention supports only SDPA and FlashAttention-3")

        # Combine projections while retaining the original learned Parameters.
        # GQA needs unequal Q/K/V widths, not three equal chunks.
        weight = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0)
        q, k, v = F.linear(hidden_states, weight).split(
            (
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ),
            dim=-1,
        )

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided")
        cos, sin = position_embeddings
        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.config._attn_implementation == "flash_attention_3":
            if self.attention_dropout != 0:
                raise ValueError("FlashAttention-3 requires attention_dropout=0")
            if not q.is_cuda or q.dtype != torch.bfloat16:
                raise ValueError(
                    "SLM FlashAttention-3 requires CUDA BF16 inputs; use BF16 "
                    "autocast or select SDPA for FP32/CPU diagnostics"
                )
            if attention_mask is not None and attention_mask.ndim != 2:
                raise ValueError("FlashAttention-3 requires a 2D padding mask; select SDPA for custom 4D masks")

        # Cache updates retain grouped heads. Validate FA3 inputs first so an
        # unsupported dtype/mask cannot partially advance a caller's cache.
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        if self.config._attn_implementation == "flash_attention_3":
            # Transformers handles padding; FA3 consumes grouped KV directly
            # and aligns causal attention to the end of the dynamic KV cache.
            attn_output, _ = flash_attention_forward(
                self, q, k, v, attention_mask,
                dropout=0.0,
                scaling=self.head_dim ** -0.5,
                is_causal=True,
                position_ids=position_ids,
            )
            return self.o_proj(attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim))

        # create_causal_mask() prepares an offset-aware 4D mask whenever one
        # is required. A None mask means SDPA may use its causal fast path.
        is_causal = attention_mask is None and q_len > 1

        # Match the pinned Transformers Llama SDPA eligibility policy. An
        # explicit mask (including one produced during compilation) can make
        # native GQA fall back to math attention. Expand K/V only in that case
        # or when the native-GQA shape/backend guard rejects the inputs, so
        # fused memory-efficient SDPA remains eligible. Do not drop the mask.
        # Cache updates above still store only the original KV heads.
        enable_gqa = False
        if self.num_query_groups > 1:
            enable_gqa = use_gqa_in_sdpa(attention_mask, k, v)
            if not enable_gqa:
                k = repeat_kv(k, self.num_query_groups)
                v = repeat_kv(v, self.num_query_groups)

        dropout_p = self.attention_dropout if self.training else 0.0
        # CUDA BF16/FP16 training must use a fused attention backend. Keep the
        # restriction at the SDPA call, inside the compiled forward, rather
        # than relying only on process-global flags outside torch.compile.
        # An unsupported fused shape/mask must raise, never retry with math.
        # CPU, FP32 diagnostics and eval-mode inference retain normal dispatch.
        require_fused = self.training and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
        attention_context = (
            sdpa_kernel([
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ])
            if require_fused else nullcontext()
        )
        with attention_context:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output
