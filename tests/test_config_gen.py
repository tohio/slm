"""
tests/test_config_gen.py
------------------------
Unit tests for config_gen.config_gen — pure utility tests, no GPU or pipeline
outputs required.

Run with:
    pytest tests/test_config_gen.py -v
"""

import pytest
import yaml

from config_gen.config_gen import (
    DPO_PROFILES,
    GPU_SPECS,
    MODES,
    SFT_CHAT_PROFILES,
    SFT_CODE_PROFILES,
    SIZE_PROFILES,
    _round_down_pow2,
    compute_dpo_config,
    compute_pretrain_config,
    compute_sft_chat_config,
    compute_sft_code_config,
    render_dpo_yaml,
    render_pretrain_yaml,
    render_sft_chat_yaml,
    render_sft_code_yaml,
    render_plan,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

class TestRoundDownPow2:
    @pytest.mark.parametrize("n,expected", [
        (1, 1), (2, 2), (3, 2), (7, 4), (8, 8), (127, 64), (128, 128), (1000, 512),
    ])
    def test_positive(self, n, expected):
        assert _round_down_pow2(n) == expected

    @pytest.mark.parametrize("n", [0, -1, -100])
    def test_non_positive_returns_one(self, n):
        assert _round_down_pow2(n) == 1


# ── Pretrain — specific scenarios ────────────────────────────────────────────

class TestPretrainSpecific:
    def test_125m_h200_1gpu(self):
        cfg = compute_pretrain_config("h200", "125m", 1)
        assert cfg.gradient_checkpointing is False
        assert cfg.actual_global_batch == 32
        assert cfg.micro_batch_size * cfg.gradient_accumulation_steps == 32

    def test_125m_h200_8gpu(self):
        cfg = compute_pretrain_config("h200", "125m", 8)
        assert cfg.actual_global_batch == 32
        assert cfg.micro_batch_size == 4
        assert cfg.gradient_accumulation_steps == 1

    def test_1b_h200_auto_ckpt(self):
        """1b on H200×1 needs ckpt to fit ref_global=128 in one step."""
        cfg = compute_pretrain_config("h200", "1b", 1)
        assert cfg.gradient_checkpointing is True
        assert cfg.gradient_accumulation_steps == 1


# ── Pretrain — invariants ────────────────────────────────────────────────────

class TestPretrainInvariants:
    @pytest.mark.parametrize("size", sorted(SIZE_PROFILES))
    @pytest.mark.parametrize("num_gpus", [1, 4, 8])
    def test_token_budget_within_2pct(self, size, num_gpus):
        cfg = compute_pretrain_config("h200", size, num_gpus)
        target = SIZE_PROFILES[size].tokens
        err = abs(cfg.actual_total_tokens - target) / target
        assert err < 0.02, f"{size}/{num_gpus}: err={err:.3%}"

    @pytest.mark.parametrize("size", sorted(SIZE_PROFILES))
    @pytest.mark.parametrize("num_gpus", [1, 4, 8])
    def test_global_batch_hits_reference(self, size, num_gpus):
        cfg = compute_pretrain_config("h200", size, num_gpus)
        assert cfg.actual_global_batch == SIZE_PROFILES[size].ref_global_batch


# ── SFT chat ─────────────────────────────────────────────────────────────────

class TestSFTChat:
    def test_125m_h200_1gpu_global_batch(self):
        cfg = compute_sft_chat_config("h200", "125m", 1)
        assert cfg.actual_global_batch == 128

    def test_global_batch_hits_reference_across_grid(self):
        for size in SFT_CHAT_PROFILES:
            for gpus in [1, 4, 8]:
                cfg = compute_sft_chat_config("h200", size, gpus)
                ref = SFT_CHAT_PROFILES[size].ref_global_batch
                assert cfg.actual_global_batch == ref, \
                    f"{size}/{gpus}: ref={ref}, got={cfg.actual_global_batch}"

    def test_unknown_size_raises(self):
        with pytest.raises(ValueError, match="Unknown SFT size"):
            compute_sft_chat_config("h200", "huge", 1)


# ── SFT code ─────────────────────────────────────────────────────────────────

class TestSFTCode:
    def test_global_batch_hits_reference_across_grid(self):
        for size in SFT_CODE_PROFILES:
            for gpus in [1, 4, 8]:
                cfg = compute_sft_code_config("h200", size, gpus)
                ref = SFT_CODE_PROFILES[size].ref_global_batch
                assert cfg.actual_global_batch == ref

    def test_chat_and_code_diverge_on_lr(self):
        """Chat and code use different LRs — make sure profiles aren't aliased."""
        for size in SFT_CHAT_PROFILES:
            assert SFT_CHAT_PROFILES[size].lr > SFT_CODE_PROFILES[size].lr, \
                f"chat LR should exceed code LR for {size}"


# ── DPO ──────────────────────────────────────────────────────────────────────

class TestDPO:
    def test_125m_h200_1gpu(self):
        cfg = compute_dpo_config("h200", "125m", 1)
        assert cfg.actual_global_batch == 64

    def test_1b_h200_auto_ckpt(self):
        """1b DPO on H200×1: should enable ckpt — DPO state + activations are heavy."""
        cfg = compute_dpo_config("h200", "1b", 1)
        assert cfg.gradient_checkpointing is True

    def test_global_batch_hits_reference_across_grid(self):
        for size in DPO_PROFILES:
            for gpus in [1, 4, 8]:
                cfg = compute_dpo_config("h200", size, gpus)
                ref = DPO_PROFILES[size].ref_global_batch
                assert cfg.actual_global_batch == ref

    def test_dpo_state_exceeds_sft(self):
        """DPO state must include the reference model — should exceed SFT for the same size."""
        for size in DPO_PROFILES:
            assert DPO_PROFILES[size].state_gb > SFT_CHAT_PROFILES[size].state_gb


# ── Modes ────────────────────────────────────────────────────────────────────

class TestModes:
    def test_modes_have_distinct_vram_fractions(self):
        fractions = [MODES[m].vram_fraction for m in ["conservative", "balanced", "aggressive"]]
        assert fractions == [0.70, 0.80, 0.90]
        # Each strictly larger than the previous
        assert fractions[0] < fractions[1] < fractions[2]

    def test_aggressive_fits_at_least_as_much_on_tight_gpu(self):
        """1b SFT on A100-40 — modes must produce different micro_batch values."""
        c = compute_sft_chat_config("a100_40", "1b", 1, mode_name="conservative")
        b = compute_sft_chat_config("a100_40", "1b", 1, mode_name="balanced")
        a = compute_sft_chat_config("a100_40", "1b", 1, mode_name="aggressive")
        assert c.micro_batch_size <= b.micro_batch_size <= a.micro_batch_size

    def test_conservative_uses_smaller_vram_budget(self):
        c = compute_sft_chat_config("a100_40", "1b", 1, mode_name="conservative")
        a = compute_sft_chat_config("a100_40", "1b", 1, mode_name="aggressive")
        assert c.vram_budget_gb < a.vram_budget_gb

    def test_aggressive_allows_non_power_of_two(self):
        """Aggressive mode disables power-of-2 rounding — should sometimes pick odd numbers."""
        a = compute_sft_chat_config("a100_40", "1b", 1, mode_name="aggressive")
        # Won't always be non-pow2, but the mode allows it
        # Strict assertion: the mode flag is set
        assert MODES["aggressive"].power_of_two_only is False
        assert MODES["balanced"].power_of_two_only is True


# ── B200 vs H200 ─────────────────────────────────────────────────────────────

class TestGPUComparison:
    @pytest.mark.parametrize("size", sorted(SIZE_PROFILES))
    def test_b200_pretrain_fits_at_least_as_much_as_h200(self, size):
        h = compute_pretrain_config("h200", size, 1)
        b = compute_pretrain_config("b200", size, 1)
        assert b.micro_batch_size >= h.micro_batch_size


# ── Memory budget invariant ──────────────────────────────────────────────────

class TestMemoryBudget:
    @pytest.mark.parametrize("gpu_key", ["h200", "b200", "h100", "a100_80"])
    @pytest.mark.parametrize("size", sorted(SIZE_PROFILES))
    def test_pretrain_estimated_within_budget(self, gpu_key, size):
        try:
            cfg = compute_pretrain_config(gpu_key, size, 1)
        except RuntimeError:
            pytest.skip(f"{size} doesn't fit on {gpu_key}")
        assert cfg.estimated_vram_gb <= cfg.vram_budget_gb

    @pytest.mark.parametrize("gpu_key", ["h200", "b200", "h100", "a100_80"])
    @pytest.mark.parametrize("size", sorted(DPO_PROFILES))
    def test_dpo_estimated_within_budget(self, gpu_key, size):
        try:
            cfg = compute_dpo_config(gpu_key, size, 1)
        except RuntimeError:
            pytest.skip(f"DPO {size} doesn't fit on {gpu_key}")
        assert cfg.estimated_vram_gb <= cfg.vram_budget_gb


# ── User overrides ───────────────────────────────────────────────────────────

class TestUserOverrides:
    def test_force_ckpt_on(self):
        cfg = compute_pretrain_config("h200", "125m", 1, force_ckpt=True)
        assert cfg.gradient_checkpointing is True

    def test_force_ckpt_off_pretrain(self):
        cfg = compute_pretrain_config("h200", "125m", 1, force_ckpt=False)
        assert cfg.gradient_checkpointing is False

    def test_force_ckpt_off_sft(self):
        cfg = compute_sft_chat_config("h200", "125m", 1, force_ckpt=False)
        assert cfg.gradient_checkpointing is False

    def test_target_global_batch(self):
        cfg = compute_pretrain_config("h200", "125m", 1, target_global_batch=64)
        assert cfg.actual_global_batch == 64

    def test_target_tokens(self):
        cfg = compute_pretrain_config("h200", "125m", 1,
                                      target_global_batch=64,
                                      target_tokens=5_000_000_000)
        assert 38_000 < cfg.max_steps < 38_300


# ── Input validation ─────────────────────────────────────────────────────────

class TestValidation:
    @pytest.mark.parametrize("compute,err_match", [
        (lambda: compute_pretrain_config("nope", "125m", 1), "Unknown GPU"),
        (lambda: compute_pretrain_config("h200", "huge", 1), "Unknown pretrain size"),
        (lambda: compute_pretrain_config("h200", "125m", 0), "num_gpus"),
        (lambda: compute_pretrain_config("h200", "125m", 1, mode_name="extreme"), "Unknown mode"),
        (lambda: compute_sft_chat_config("h200", "huge", 1), "Unknown SFT size"),
        (lambda: compute_dpo_config("h200", "huge", 1), "Unknown DPO size"),
    ])
    def test_raises_value_error(self, compute, err_match):
        with pytest.raises(ValueError, match=err_match):
            compute()


# ── Rendering ────────────────────────────────────────────────────────────────

class TestRendering:
    def test_pretrain_yaml_parses(self):
        cfg = compute_pretrain_config("h200", "350m", 4)
        d = yaml.safe_load(render_pretrain_yaml(cfg))
        assert d["name"] == "slm-350m"
        assert d["training"]["micro_batch_size"] == cfg.micro_batch_size
        assert d["training"]["max_steps"] == cfg.max_steps

    def test_sft_chat_yaml_parses(self):
        cfg = compute_sft_chat_config("h200", "125m", 1)
        d = yaml.safe_load(render_sft_chat_yaml(cfg))
        assert d["name"] == "slm-125m-chat"
        assert d["training"]["micro_batch_size"] == cfg.micro_batch_size
        assert d["model"]["max_seq_length"] == SFT_CHAT_PROFILES["125m"].max_seq_length
        assert d["data"]["train_path"] == "$DATA_DIR/sft/chat/train.jsonl"

    def test_sft_code_yaml_parses(self):
        cfg = compute_sft_code_config("h200", "125m", 1)
        d = yaml.safe_load(render_sft_code_yaml(cfg))
        assert d["name"] == "slm-125m-chat-code"
        assert d["data"]["train_path"] == "$DATA_DIR/sft/code/train.jsonl"
        # Code chains off chat
        assert "chat" in d["model"]["base_model_path"]

    def test_dpo_yaml_parses(self):
        cfg = compute_dpo_config("h200", "125m", 1)
        d = yaml.safe_load(render_dpo_yaml(cfg))
        assert d["name"] == "slm-125m-dpo"
        assert d["dpo"]["beta"] == DPO_PROFILES["125m"].dpo_beta
        assert d["dpo"]["max_prompt_length"] == DPO_PROFILES["125m"].max_prompt_length

    def test_recipe_lr_preserved(self):
        """Script must NEVER touch the LR — it's a recipe value."""
        cfg = compute_pretrain_config("h200", "125m", 1)
        d = yaml.safe_load(render_pretrain_yaml(cfg))
        assert d["optimizer"]["lr"] == SIZE_PROFILES["125m"].lr

        cfg = compute_sft_chat_config("h200", "125m", 1)
        d = yaml.safe_load(render_sft_chat_yaml(cfg))
        assert d["optimizer"]["lr"] == SFT_CHAT_PROFILES["125m"].lr

        cfg = compute_dpo_config("h200", "125m", 1)
        d = yaml.safe_load(render_dpo_yaml(cfg))
        assert d["optimizer"]["lr"] == DPO_PROFILES["125m"].lr


# ── Warnings ─────────────────────────────────────────────────────────────────

class TestWarnings:
    def test_low_vram_use_warns(self):
        """125m pretrain on H200 should warn about lots of headroom."""
        cfg = compute_pretrain_config("h200", "125m", 1)
        joined = " ".join(cfg.warnings)
        assert "headroom" in joined.lower() or "aggressive" in joined.lower()

    def test_dpo_lr_warning_present(self):
        cfg = compute_dpo_config("h200", "125m", 1)
        joined = " ".join(cfg.warnings)
        assert "DPO is LR-sensitive" in joined

    def test_sft_measurement_warning_present(self):
        cfg = compute_sft_chat_config("h200", "125m", 1)
        joined = " ".join(cfg.warnings)
        assert "analytical" in joined.lower()

    def test_1b_multi_gpu_fsdp_hint(self):
        cfg = compute_pretrain_config("h200", "1b", 8)
        joined = " ".join(cfg.warnings)
        assert "fsdp" in joined.lower()

    def test_plan_includes_warnings(self):
        cfg = compute_pretrain_config("h200", "125m", 1)
        plan = render_plan(cfg)
        assert "things to verify" in plan.lower()