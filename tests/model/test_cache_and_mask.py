"""
tests/model/test_cache_and_mask.py
----------------------------------
Focused tests for KV cache and attention mask correctness.

These tests target behaviours that test_model.py does NOT cover:

    - Multi-token prefill on top of a populated KV cache
      (catches cache/mask offset bugs when q_len < kv_len)
    - Token-by-token generation matching full forward
      (catches q_len==1 cache path bugs)
    - Native Transformers cache preservation
    - Batched inference respecting left-padding masks
      (catches attention_mask being ignored during eval)
    - Exact analytical parameter counts for all three tiers
      (test_model.py only checks the mini config)

Runs on CPU in under 10 seconds. No pipeline outputs required.

Run with:
    .venv/bin/pytest tests/model/test_cache_and_mask.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, StaticCache

from model.config import CONFIGS, SLMConfig
from model.model import SLMForCausalLM


def test_attention_uses_native_sdpa_gqa(monkeypatch):
    """GQA should not materialize repeated K/V heads before SDPA."""
    from model.attention import GroupedQueryAttention, RotaryEmbedding

    config = _tiny_config()
    attention = GroupedQueryAttention(config, layer_idx=0)
    original = F.scaled_dot_product_attention
    observed = {}

    def wrapped(query, key, value, **kwargs):
        observed["q_heads"] = query.shape[1]
        observed["kv_heads"] = key.shape[1]
        observed["enable_gqa"] = kwargs.get("enable_gqa")
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", wrapped)
    hidden = torch.randn(1, 4, config.hidden_size)
    position_ids = torch.arange(4).unsqueeze(0)
    position_embeddings = RotaryEmbedding(config)(hidden, position_ids)
    cache = DynamicCache(config=config)
    output = attention(
        hidden,
        position_embeddings=position_embeddings,
        past_key_values=cache,
        use_cache=True,
    )

    assert output.shape == hidden.shape
    assert observed == {
        "q_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "enable_gqa": True,
    }
    assert cache.layers[0].keys.shape[1] == config.num_key_value_heads


# ── Tiny config for multi-forward tests ───────────────────────────────────────
# Smaller than make_mini_config so cache/prefill tests that run 2-10 forwards
# stay cheap. 2 layers × 64 hidden exercises all the mask/cache logic.

def _tiny_config() -> SLMConfig:
    return SLMConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )


# ── Prefill-then-continue equivalence (Bug #1 regression) ─────────────────────

def test_prefill_then_continue_matches_full_forward():
    """
    A full forward over a prompt should produce the same logits as:
      1. Forward first N tokens with use_cache=True → cache
      2. Forward remaining tokens with past_key_values=cache

    This exercises the q_len > 1 cache path. If is_causal=True is passed
    to SDPA when q_len < kv_len, SDPA applies a square lower-triangular
    mask at the wrong offset — this test fails.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    model = SLMForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 20))

    with torch.no_grad():
        full = model(prompt).logits

        prefix_len = 15
        pre = model(prompt[:, :prefix_len], use_cache=True)
        cont = model(
            prompt[:, prefix_len:],
            past_key_values=pre.past_key_values,
            use_cache=True,
        ).logits

    torch.testing.assert_close(full[:, prefix_len:], cont, atol=1e-4, rtol=1e-4)


def test_token_by_token_generation_matches_full_forward():
    """
    Generating one token at a time on top of a growing cache should
    produce the same logits as a full forward pass. Exercises the
    q_len == 1 cache path (the common generation case).
    """
    torch.manual_seed(0)
    config = _tiny_config()
    model = SLMForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        full = model(prompt).logits

        out = model(prompt[:, :1], use_cache=True)
        pkv = out.past_key_values
        step_logits = [out.logits]
        for t in range(1, prompt.shape[1]):
            out = model(prompt[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            step_logits.append(out.logits)

    stepped = torch.cat(step_logits, dim=1)
    torch.testing.assert_close(full, stepped, atol=1e-4, rtol=1e-4)


# ── HuggingFace generate() cache integration ──────────────────────────────────

def test_empty_dynamic_cache_does_not_truncate_prefill():
    """
    Transformers creates an empty DynamicCache before the first generation
    forward. The full prompt must still reach the model on that prefill step.

    Treating a non-None but empty cache as populated truncates the prompt to its
    final token and makes cached generation diverge immediately.
    """
    config = _tiny_config()
    model = SLMForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    attention_mask = torch.ones_like(prompt)
    empty_cache = DynamicCache(config=config)

    model_inputs = model.prepare_inputs_for_generation(
        prompt,
        next_sequence_length=None,
        past_key_values=empty_cache,
        attention_mask=attention_mask,
        is_first_iteration=True,
        use_cache=True,
    )

    assert model_inputs["input_ids"].shape[1] == prompt.shape[1]


def test_generate_cached_matches_uncached():
    """
    Greedy generation with and without the KV cache must produce the same token
    sequence. This covers the full GenerationMixin integration rather than
    only direct forward calls with manually managed cache tuples.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    model = SLMForCausalLM(config).eval()
    prompt = torch.randint(4, config.vocab_size, (1, 10))
    attention_mask = torch.ones_like(prompt)
    generation_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": 8,
        "do_sample": False,
        "pad_token_id": config.pad_token_id,
        "eos_token_id": None,
    }

    with torch.no_grad():
        cached = model.generate(prompt, use_cache=True, **generation_kwargs)
        uncached = model.generate(prompt, use_cache=False, **generation_kwargs)

    assert torch.equal(cached, uncached)


def test_forward_preserves_native_cache_object():
    """The model must update and return the supplied Cache without conversion."""
    config = _tiny_config()
    model = SLMForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    cache = DynamicCache(config=config)

    with torch.no_grad():
        out = model(prompt, past_key_values=cache, use_cache=True)

    assert out.past_key_values is cache
    assert cache.get_seq_length() == prompt.shape[1]
    assert len(cache.layers) == config.num_hidden_layers


def test_static_cache_matches_full_forward():
    """Compileable StaticCache must preserve causal logits across decode steps."""
    torch.manual_seed(0)
    config = _tiny_config()
    model = SLMForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 11))
    cache = StaticCache(config=config, max_cache_len=16)

    with torch.no_grad():
        full = model(prompt, use_cache=False).logits
        model(
            prompt[:, :10],
            attention_mask=torch.ones(1, 10, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
        continued = model(
            prompt[:, 10:],
            attention_mask=torch.ones(1, 11, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        ).logits

    torch.testing.assert_close(full[:, 10:], continued, atol=1e-4, rtol=1e-4)


def test_short_legacy_cache_is_rejected():
    """A malformed legacy cache must not silently skip decoder layers."""
    config = _tiny_config()
    model = SLMForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 1))
    head_dim = config.head_dim
    one_layer_cache = [
        (
            torch.zeros(1, config.num_key_value_heads, 2, head_dim),
            torch.zeros(1, config.num_key_value_heads, 2, head_dim),
        )
    ]

    with pytest.raises(ValueError, match="exactly one entry per decoder layer"):
        model(
            prompt,
            past_key_values=one_layer_cache,
            use_cache=True,
        )


# ── Batched inference with padding (Bug #2 regression) ────────────────────────

def test_batched_forward_respects_padding_mask():
    """
    When a batch contains sequences of different lengths (left-padded),
    the padded positions must not affect the logits of the real tokens.

    Run a short prompt alone, then run it batched alongside a longer
    prompt (forcing the short one to be padded). The logits at the
    short prompt's real positions should be identical in both cases.

    Left padding is intentional: a causal mask already prevents real tokens
    from attending to right-padding in the future, so a right-padding test
    cannot detect an ignored padding mask.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    model = SLMForCausalLM(config).eval()

    # Avoid token id 0 in real prompts so it's distinguishable from pad
    short = torch.randint(1, config.vocab_size, (1, 5))
    long_ = torch.randint(1, config.vocab_size, (1, 12))
    pad_id = 0

    short_padded = torch.full((1, 12), pad_id, dtype=torch.long)
    short_padded[:, -5:] = short
    batch = torch.cat([short_padded, long_], dim=0)  # (2, 12)

    attn_mask = torch.ones(2, 12, dtype=torch.long)
    attn_mask[0, :-5] = 0

    with torch.no_grad():
        solo = model(short).logits                              # (1, 5, V)
        batched = model(batch, attention_mask=attn_mask).logits  # (2, 12, V)

    torch.testing.assert_close(batched[0:1, -5:], solo, atol=1e-4, rtol=1e-4)


# ── Parameter counts across all tiers ─────────────────────────────────────────

def _analytical_parameter_count(config: SLMConfig) -> int:
    hidden = config.hidden_size
    head_dim = config.head_dim
    kv_width = config.num_key_value_heads * head_dim

    embeddings = config.vocab_size * hidden
    attention = (
        hidden * hidden
        + hidden * kv_width
        + hidden * kv_width
        + hidden * hidden
    )
    mlp = 3 * hidden * config.intermediate_size
    block_norms = 2 * hidden
    final_norm = hidden
    untied_lm_head = 0 if config.tie_word_embeddings else embeddings

    return (
        embeddings
        + config.num_hidden_layers * (attention + mlp + block_norms)
        + final_norm
        + untied_lm_head
    )


@pytest.mark.parametrize(
    "name,layers,expected",
    [
        ("125m", 16, 125_264_640),
        ("350m", 27, 351_329_280),
        ("1b", 21, 1_012_488_192),
    ],
)
def test_production_architecture_and_parameter_count(
    name: str,
    layers: int,
    expected: int,
):
    """
    Production depth and unique parameter count are exact architecture
    contracts. Analytical counting avoids allocating the full 1B model on CPU.
    """
    config = CONFIGS[name]
    assert config.num_hidden_layers == layers
    assert config.tie_word_embeddings is True
    assert _analytical_parameter_count(config) == expected
