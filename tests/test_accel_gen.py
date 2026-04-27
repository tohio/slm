"""
tests/test_accel_gen.py
-----------------------
Unit tests for slm.accel_gen — accelerate launch config generator.

Run with:
    pytest tests/test_accel_gen.py -v
"""

import pytest
import yaml

from slm.accel_gen import render_ddp, render_fsdp, DEFAULT_FSDP_TRANSFORMER_LAYER


# ── DDP ──────────────────────────────────────────────────────────────────────

class TestDDP:
    def test_yaml_parses(self):
        d = yaml.safe_load(render_ddp(num_gpus=8))
        assert d["distributed_type"] == "MULTI_GPU"
        assert d["num_processes"] == 8
        assert d["mixed_precision"] == "bf16"

    @pytest.mark.parametrize("gpus", [1, 2, 4, 8])
    def test_num_processes_set_correctly(self, gpus):
        d = yaml.safe_load(render_ddp(num_gpus=gpus))
        assert d["num_processes"] == gpus

    def test_no_fsdp_block(self):
        """DDP config must not include fsdp_config."""
        d = yaml.safe_load(render_ddp(num_gpus=8))
        assert "fsdp_config" not in d

    def test_alternative_precision(self):
        d = yaml.safe_load(render_ddp(num_gpus=4, mixed_precision="fp16"))
        assert d["mixed_precision"] == "fp16"


# ── FSDP ─────────────────────────────────────────────────────────────────────

class TestFSDP:
    def test_yaml_parses(self):
        d = yaml.safe_load(render_fsdp(num_gpus=8))
        assert d["distributed_type"] == "FSDP"
        assert d["num_processes"] == 8

    def test_default_transformer_layer(self):
        d = yaml.safe_load(render_fsdp(num_gpus=8))
        assert d["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"] == \
            DEFAULT_FSDP_TRANSFORMER_LAYER
        assert d["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"] == "SLMBlock"

    def test_default_sharding_full(self):
        d = yaml.safe_load(render_fsdp(num_gpus=8))
        assert d["fsdp_config"]["fsdp_sharding_strategy"] == "FULL_SHARD"

    def test_cpu_offload_default_off(self):
        d = yaml.safe_load(render_fsdp(num_gpus=8))
        assert d["fsdp_config"]["fsdp_offload_params"] is False

    def test_cpu_offload_can_enable(self):
        d = yaml.safe_load(render_fsdp(num_gpus=8, cpu_offload=True))
        assert d["fsdp_config"]["fsdp_offload_params"] is True

    def test_alternative_sharding(self):
        d = yaml.safe_load(render_fsdp(num_gpus=8, sharding_strategy="SHARD_GRAD_OP"))
        assert d["fsdp_config"]["fsdp_sharding_strategy"] == "SHARD_GRAD_OP"

    def test_custom_transformer_layer(self):
        d = yaml.safe_load(render_fsdp(num_gpus=8, transformer_layer="CustomBlock"))
        assert d["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"] == "CustomBlock"

    def test_use_orig_params_true(self):
        """use_orig_params=true is required for torch.compile to play nicely with FSDP."""
        d = yaml.safe_load(render_fsdp(num_gpus=8))
        assert d["fsdp_config"]["fsdp_use_orig_params"] is True