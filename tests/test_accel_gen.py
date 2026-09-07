"""
tests/test_accel_gen.py
-----------------------
Unit tests for config_gen.accel_gen — accelerate launch config generator.

Run with:
    pytest tests/test_accel_gen.py -v
"""

import pytest
import yaml

from config_gen.accel_gen import render_ddp


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
