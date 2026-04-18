"""
tests/model/test_cache_and_mask.py
----------------------------------
Focused tests for KV cache and attention mask correctness.

These tests target behaviours that test_model.py does NOT cover:

    - Multi-token prefill on top of a populated KV cache
      (catches cache/mask offset bugs when q_len < kv_len)
    - Token-by-token generation matching full forward
      (catches q_len==1 cache path bugs)
    - Batched inference respecting padding masks
      (catches attention_mask being ignored during eval)
    - Parameter count validation for all three tiers
      (test_model.py only checks the mini config)

Runs on CPU in under 10 seconds. No pipeline outputs required.

Run with:
    .venv/bin/pytest tests/model/test_cache_and_mask.py -v
"""

from __future__ import annotations

import pytest
import torch

from model.config import CONFIGS, SLMConfig
from model.model import SLMForCausalLM


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


# ── Batched inference with padding (Bug #2 regression) ────────────────────────

def test_batched_forward_respects_padding_mask():
    """
    When a batch contains sequences of different lengths (right-padded),
    the padded positions must not affect the logits of the real tokens.

    Run a short prompt alone, then run it batched alongside a longer
    prompt (forcing the short one to be padded). The logits at the
    short prompt's real positions should be identical in both cases.

    If attention_mask is ignored during eval, the batched run attends
    over pad positions and this test fails.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    model = SLMForCausalLM(config).eval()

    # Avoid token id 0 in real prompts so it's distinguishable from pad
    short = torch.randint(1, config.vocab_size, (1, 5))
    long_ = torch.randint(1, config.vocab_size, (1, 12))
    pad_id = 0

    short_padded = torch.full((1, 12), pad_id, dtype=torch.long)
    short_padded[:, :5] = short
    batch = torch.cat([short_padded, long_], dim=0)  # (2, 12)

    attn_mask = torch.ones(2, 12, dtype=torch.long)
    attn_mask[0, 5:] = 0

    with torch.no_grad():
        solo = model(short).logits                              # (1, 5, V)
        batched = model(batch, attention_mask=attn_mask).logits  # (2, 12, V)

    torch.testing.assert_close(batched[0:1, :5], solo, atol=1e-4, rtol=1e-4)


# ── Parameter counts across all tiers ─────────────────────────────────────────

@pytest.mark.parametrize(
    "name, target",
    [
        ("125m", 125_000_000),
        ("350m", 350_000_000),
        ("1b",   1_000_000_000),
    ],
)
def test_parameter_count_within_tolerance(name: str, target: int):
    """
    Parameter count per tier should match the published target within 10%.
    A wider miss usually means intermediate_size, num_layers, or vocab_size
    drifted away from the intended configuration.

    test_model.py only validates the mini config (~25M); this covers the
    three production tiers.
    """
    config = CONFIGS[name]
    model = SLMForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    drift = abs(n_params - target) / target
    assert drift < 0.10, (
        f"{name}: {n_params:,} params vs target {target:,} (drift {drift:.1%})"
    )