"""
tests/test_config_gen.py
------------------------
Unit tests for config_gen.config_gen — pure utility tests, no GPU or pipeline
outputs required.

Run with:
    pytest tests/test_config_gen.py -v
"""

from pathlib import Path

import pytest
import yaml

from model.config import CONFIGS as MODEL_CONFIGS
from config_gen.config_gen import (
    DPO_PROFILES,
    GPU_SPECS,
    MODES,
    SFT_INSTRUCT_PROFILES,
    SFT_CODE_PROFILES,
    SIZE_PROFILES,
    _round_down_pow2,
    compute_dpo_config,
    compute_pretrain_config,
    compute_sft_instruct_config,
    compute_sft_code_config,
    render_dpo_yaml,
    render_pretrain_yaml,
    render_sft_instruct_yaml,
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
        assert cfg.actual_global_batch == SIZE_PROFILES["125m"].ref_global_batch
        assert cfg.micro_batch_size * cfg.gradient_accumulation_steps == \
            SIZE_PROFILES["125m"].ref_global_batch

    def test_125m_h200_8gpu(self):
        cfg = compute_pretrain_config("h200", "125m", 8)
        assert cfg.actual_global_batch == SIZE_PROFILES["125m"].ref_global_batch
        assert cfg.micro_batch_size * cfg.gradient_accumulation_steps * 8 == \
            SIZE_PROFILES["125m"].ref_global_batch

    def test_1b_h200_auto_ckpt(self):
        """The auto-policy must produce a valid in-budget 1B H200 plan."""
        cfg = compute_pretrain_config("h200", "1b", 1)
        assert cfg.actual_global_batch == SIZE_PROFILES["1b"].ref_global_batch
        assert cfg.estimated_vram_gb <= cfg.vram_budget_gb


# ── Pretrain — invariants ────────────────────────────────────────────────────

class TestPretrainInvariants:
    @pytest.mark.parametrize("size", sorted(SIZE_PROFILES))
    @pytest.mark.parametrize("num_gpus", [1, 4, 8])
    def test_token_budget_within_2pct(self, size, num_gpus):
        cfg = compute_pretrain_config("h200", size, num_gpus)
        target = SIZE_PROFILES[size].consumed_tokens
        err = abs(cfg.actual_consumed_tokens - target) / target
        assert err < 0.02, f"{size}/{num_gpus}: err={err:.3%}"

    @pytest.mark.parametrize("size", sorted(SIZE_PROFILES))
    @pytest.mark.parametrize("num_gpus", [1, 4, 8])
    def test_global_batch_hits_reference(self, size, num_gpus):
        cfg = compute_pretrain_config("h200", size, num_gpus)
        assert cfg.actual_global_batch == SIZE_PROFILES[size].ref_global_batch


# ── SFT instruct ─────────────────────────────────────────────────────────────────

class TestSFTInstruct:
    def test_125m_h200_1gpu_global_batch(self):
        cfg = compute_sft_instruct_config("h200", "125m", 1)
        assert cfg.actual_global_batch == \
            SFT_INSTRUCT_PROFILES["125m"].ref_global_batch

    def test_global_batch_hits_reference_across_grid(self):
        for size in SFT_INSTRUCT_PROFILES:
            for gpus in [1, 4, 8]:
                cfg = compute_sft_instruct_config("h200", size, gpus)
                ref = SFT_INSTRUCT_PROFILES[size].ref_global_batch
                assert cfg.actual_global_batch == ref, \
                    f"{size}/{gpus}: ref={ref}, got={cfg.actual_global_batch}"

    def test_unknown_size_raises(self):
        with pytest.raises(ValueError, match="Unknown SFT size"):
            compute_sft_instruct_config("h200", "huge", 1)


# ── SFT code ─────────────────────────────────────────────────────────────────

class TestSFTCode:
    def test_global_batch_hits_reference_across_grid(self):
        for size in SFT_CODE_PROFILES:
            for gpus in [1, 4, 8]:
                cfg = compute_sft_code_config("h200", size, gpus)
                ref = SFT_CODE_PROFILES[size].ref_global_batch
                assert cfg.actual_global_batch == ref

    def test_chat_and_code_diverge_on_lr(self):
        """Instruct and code use different LRs — make sure profiles aren't aliased."""
        for size in SFT_INSTRUCT_PROFILES:
            assert SFT_INSTRUCT_PROFILES[size].lr > SFT_CODE_PROFILES[size].lr, \
                f"instruct LR should exceed code LR for {size}"


# ── DPO ──────────────────────────────────────────────────────────────────────

class TestDPO:
    def test_125m_h200_1gpu(self):
        cfg = compute_dpo_config("h200", "125m", 1)
        assert cfg.actual_global_batch == DPO_PROFILES["125m"].ref_global_batch

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
            assert DPO_PROFILES[size].state_gb > SFT_INSTRUCT_PROFILES[size].state_gb


# ── Modes ────────────────────────────────────────────────────────────────────

class TestModes:
    def test_modes_have_distinct_vram_fractions(self):
        fractions = [MODES[m].vram_fraction for m in ["conservative", "balanced", "aggressive"]]
        assert fractions == [0.70, 0.80, 0.90]
        # Each strictly larger than the previous
        assert fractions[0] < fractions[1] < fractions[2]

    def test_aggressive_fits_at_least_as_much_on_tight_gpu(self):
        """1b SFT on A100-40 — modes must produce different micro_batch values."""
        c = compute_sft_instruct_config("a100_40", "1b", 1, mode_name="conservative")
        b = compute_sft_instruct_config("a100_40", "1b", 1, mode_name="balanced")
        a = compute_sft_instruct_config("a100_40", "1b", 1, mode_name="aggressive")
        assert c.micro_batch_size <= b.micro_batch_size <= a.micro_batch_size

    def test_conservative_uses_smaller_vram_budget(self):
        c = compute_sft_instruct_config("a100_40", "1b", 1, mode_name="conservative")
        a = compute_sft_instruct_config("a100_40", "1b", 1, mode_name="aggressive")
        assert c.vram_budget_gb < a.vram_budget_gb

    def test_aggressive_allows_non_power_of_two(self):
        """Aggressive mode disables power-of-2 rounding — should sometimes pick odd numbers."""
        a = compute_sft_instruct_config("a100_40", "1b", 1, mode_name="aggressive")
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
        cfg = compute_sft_instruct_config("h200", "125m", 1, force_ckpt=False)
        assert cfg.gradient_checkpointing is False

    def test_target_global_batch(self):
        cfg = compute_pretrain_config("h200", "125m", 1, target_global_batch=64)
        assert cfg.actual_global_batch == 64

    def test_target_consumed_tokens_override(self):
        """
        Explicit consumed-token override should control max_steps.

        This tests the new API name. The deprecated CLI flag --target-tokens
        is covered in config_gen.config_gen argument parsing, not here.
        """
        cfg = compute_pretrain_config(
            "h200",
            "125m",
            1,
            target_global_batch=64,
            target_consumed_tokens=5_000_000_000,
        )
        assert 38_000 < cfg.max_steps < 38_300

    def test_rendered_pretrain_config_enables_realized_token_schedule(self):
        cfg = compute_pretrain_config("h200", "125m", 1)

        rendered = yaml.safe_load(render_pretrain_yaml(cfg))

        assert rendered["training"]["schedule_from_realized_tokens"] is True


# ── Input validation ─────────────────────────────────────────────────────────

class TestValidation:
    @pytest.mark.parametrize("compute,err_match", [
        (lambda: compute_pretrain_config("nope", "125m", 1), "Unknown GPU"),
        (lambda: compute_pretrain_config("h200", "huge", 1), "Unknown pretrain size"),
        (lambda: compute_pretrain_config("h200", "125m", 0), "num_gpus"),
        (lambda: compute_pretrain_config("h200", "125m", 1, mode_name="extreme"), "Unknown mode"),
        (lambda: compute_sft_instruct_config("h200", "huge", 1), "Unknown SFT size"),
        (lambda: compute_dpo_config("h200", "huge", 1), "Unknown DPO size"),
    ])
    def test_raises_value_error(self, compute, err_match):
        with pytest.raises(ValueError, match=err_match):
            compute()


# ── Rendering ────────────────────────────────────────────────────────────────

class TestRendering:
    @pytest.mark.parametrize("size", sorted(SIZE_PROFILES))
    def test_model_preset_rope_matches_pretrain_profile(self, size):
        assert MODEL_CONFIGS[size].rope_theta == SIZE_PROFILES[size].rope_theta

    def test_checked_in_125m_layers_match_profile(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "pretrain"
            / "configs"
            / "gpt_125m.yaml"
        )
        checked_in = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert checked_in["model"]["num_hidden_layers"] == 16
        assert checked_in["model"]["num_hidden_layers"] == SIZE_PROFILES["125m"].layers

    @pytest.mark.parametrize("size", sorted(SIZE_PROFILES))
    def test_checked_in_pretrain_recipe_matches_profile(self, size):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "pretrain"
            / "configs"
            / f"gpt_{size}.yaml"
        )
        checked_in = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        training = checked_in["training"]
        profile = SIZE_PROFILES[size]

        assert checked_in["model"]["num_hidden_layers"] == profile.layers
        assert (
            training["micro_batch_size"]
            * training["gradient_accumulation_steps"]
        ) == profile.ref_global_batch

    @pytest.mark.parametrize("size", sorted(SFT_INSTRUCT_PROFILES))
    def test_checked_in_sft_recipe_matches_profile(self, size):
        root = Path(__file__).resolve().parents[1]
        for variant, profiles in (
            ("instruct", SFT_INSTRUCT_PROFILES),
            ("code", SFT_CODE_PROFILES),
        ):
            checked_in = yaml.safe_load(
                (
                    root
                    / "finetune"
                    / "configs"
                    / f"sft_{variant}_{size}.yaml"
                ).read_text(encoding="utf-8")
            )
            training = checked_in["training"]
            assert checked_in["name"] == f"slm-{size}-{variant}"
            assert (
                training["micro_batch_size"]
                * training["gradient_accumulation_steps"]
            ) == profiles[size].ref_global_batch

    @pytest.mark.parametrize("size", sorted(DPO_PROFILES))
    def test_checked_in_dpo_recipe_matches_profile(self, size):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "alignment"
            / "configs"
            / f"dpo_chat_{size}.yaml"
        )
        checked_in = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        training = checked_in["training"]

        assert checked_in["name"] == f"slm-{size}-chat"
        assert checked_in["model"]["base_model_path"].endswith(
            "/sft_instruct/final"
        )
        assert (
            training["micro_batch_size"]
            * training["gradient_accumulation_steps"]
        ) == DPO_PROFILES[size].ref_global_batch

    @pytest.mark.parametrize("size", sorted(SIZE_PROFILES))
    def test_model_preset_architecture_matches_pretrain_profile(self, size):
        model_cfg = MODEL_CONFIGS[size]
        profile = SIZE_PROFILES[size]
        assert model_cfg.hidden_size == profile.hidden
        assert model_cfg.num_hidden_layers == profile.layers
        assert model_cfg.num_attention_heads == profile.heads
        assert model_cfg.num_key_value_heads == profile.kv_heads
        assert model_cfg.max_position_embeddings == profile.ctx

    def test_pretrain_yaml_parses(self):
        cfg = compute_pretrain_config("h200", "350m", 4)
        d = yaml.safe_load(render_pretrain_yaml(cfg))
        assert d["name"] == "slm-350m"
        assert d["training"]["micro_batch_size"] == cfg.micro_batch_size
        assert d["training"]["max_steps"] == cfg.max_steps

    def test_sft_instruct_yaml_parses(self):
        cfg = compute_sft_instruct_config("h200", "125m", 1)
        d = yaml.safe_load(render_sft_instruct_yaml(cfg))
        assert d["name"] == "slm-125m-instruct"
        assert d["training"]["micro_batch_size"] == cfg.micro_batch_size
        assert d["model"]["max_seq_length"] == SFT_INSTRUCT_PROFILES["125m"].max_seq_length
        assert d["data"]["train_path"] == "$DATA_DIR/runs/125m/sft_instruct/train.jsonl"
        assert d["data"]["loss_type"] == "chunked_nll"
        assert d["data"]["min_retention_ratio"] == 0.90

    def test_sft_code_yaml_parses(self):
        cfg = compute_sft_code_config("h200", "125m", 1)
        d = yaml.safe_load(render_sft_code_yaml(cfg))
        assert d["name"] == "slm-125m-code"
        assert d["data"]["train_path"] == "$DATA_DIR/runs/125m/sft_code/train.jsonl"
        # Code chains off instruct
        assert "sft_instruct" in d["model"]["base_model_path"]

    def test_dpo_yaml_parses(self):
        cfg = compute_dpo_config("h200", "125m", 1)
        d = yaml.safe_load(render_dpo_yaml(cfg))
        assert d["name"] == "slm-125m-chat"
        assert d["model"]["base_model_path"].endswith("/sft_instruct/final")
        assert d["data"]["train_path"] == \
            "$DATA_DIR/runs/125m/dpo_chat/train.jsonl"
        assert d["dpo"]["beta"] == DPO_PROFILES["125m"].dpo_beta
        assert d["dpo"]["loss_type"] == "sigmoid"
        assert d["dpo"]["precompute_ref_log_probs"] is True
        assert d["data"]["min_retention_ratio"] == 0.99

    def test_recipe_lr_preserved(self):
        """Script must NEVER touch the LR — it's a recipe value."""
        cfg = compute_pretrain_config("h200", "125m", 1)
        d = yaml.safe_load(render_pretrain_yaml(cfg))
        assert d["optimizer"]["lr"] == SIZE_PROFILES["125m"].lr

        cfg = compute_sft_instruct_config("h200", "125m", 1)
        d = yaml.safe_load(render_sft_instruct_yaml(cfg))
        assert d["optimizer"]["lr"] == SFT_INSTRUCT_PROFILES["125m"].lr

        cfg = compute_dpo_config("h200", "125m", 1)
        d = yaml.safe_load(render_dpo_yaml(cfg))
        assert d["optimizer"]["lr"] == DPO_PROFILES["125m"].lr


# ── Warnings ─────────────────────────────────────────────────────────────────

class TestWarnings:
    def test_low_vram_use_warns(self):
        """A deliberately tiny target should warn about unused headroom."""
        cfg = compute_pretrain_config(
            "h200", "125m", 1, target_global_batch=1, force_ckpt=False
        )
        joined = " ".join(cfg.warnings)
        assert "headroom" in joined.lower() or "aggressive" in joined.lower()

    def test_dpo_lr_warning_present(self):
        cfg = compute_dpo_config("h200", "125m", 1)
        joined = " ".join(cfg.warnings)
        assert "DPO is LR-sensitive" in joined

    def test_sft_measurement_warning_present(self):
        cfg = compute_sft_instruct_config("h200", "125m", 1)
        joined = " ".join(cfg.warnings)
        assert "analytical" in joined.lower()

    def test_1b_multi_gpu_fsdp_hint(self):
        cfg = compute_pretrain_config("h200", "1b", 8)
        joined = " ".join(cfg.warnings)
        assert "fsdp" in joined.lower()

    def test_plan_includes_warnings(self):
        cfg = compute_pretrain_config(
            "h200", "125m", 1, target_global_batch=1, force_ckpt=False
        )
        plan = render_plan(cfg)
        assert "things to verify" in plan.lower()
