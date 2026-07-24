"""
tests/gpu_pipeline/test_pipeline_training.py
---------------------------------------------
Validates real outputs from a pretraining run.

Run after: make pretrain SIZE=<size>     (or make pretrain-mini)
Command:   make test-training SIZE=<size>

Checks:
    - Checkpoint directory exists
    - final/ model can be loaded
    - Tokenizer is present alongside the model
    - Forward pass produces finite loss
    - Loss is lower than random initialization (~log(32000) ≈ 10.4)
    - Config matches gpt_<size>.yaml
"""

from pathlib import Path

import pytest
import torch
import yaml

from tests.data_pipeline.helpers import pipeline_path, requires_stage
from tests.conftest import skip_or_fail_missing_artifact


@pytest.fixture(scope="module")
def pretrain_artifact(
    pretrain_model_dir,
    require_artifacts,
):
    if not pretrain_model_dir.exists():
        skip_or_fail_missing_artifact(
            pretrain_model_dir,
            "run 'make pretrain SIZE=<size>' first",
            required=require_artifacts,
        )

    from model.model import SLMForCausalLM

    model = SLMForCausalLM.from_pretrained(str(pretrain_model_dir)).eval()
    yield model
    del model


@requires_stage("pretrain")
class TestPretrainOutputs:
    @pytest.fixture(autouse=True)
    def _require_loaded_artifact(self, pretrain_artifact):
        """Apply optional/strict missing-artifact policy to every class test."""

    def test_final_dir_exists(self, pretrain_model_dir, pretrain_artifact):
        assert pretrain_model_dir.exists()

    def test_model_files_exist(self, pretrain_model_dir):
        assert (pretrain_model_dir / "config.json").exists()
        assert (pretrain_model_dir / "model.safetensors").exists() or \
               (pretrain_model_dir / "pytorch_model.bin").exists()

    def test_tokenizer_saved_alongside_model(self, pretrain_model_dir):
        assert (pretrain_model_dir / "tokenizer" / "tokenizer_config.json").exists(), (
            "Tokenizer not found alongside model — "
            "pretrain/train.py should copy tokenizer to final/"
        )

    def test_model_loads(self, pretrain_artifact):
        assert pretrain_artifact is not None

    def test_config_matches_yaml(self, pretrain_artifact, model_size):
        """
        Verify the saved config matches the source YAML for this size.
        Reads pretrain/configs/gpt_<size>.yaml as ground truth so the test
        works for any size, not just mini.
        """
        yaml_path = Path(f"pretrain/configs/gpt_{model_size}.yaml")
        if not yaml_path.exists():
            pytest.skip(f"Config YAML not found: {yaml_path}")

        with open(yaml_path) as f:
            yaml_cfg = yaml.safe_load(f)
        expected = yaml_cfg["model"]

        cfg = pretrain_artifact.config

        # Map YAML field names to SLMConfig attributes. Only check fields
        # that are present in the YAML — keeps the test forward-compatible
        # if the YAML schema gains new fields.
        field_map = {
            "hidden_size":             "hidden_size",
            "num_hidden_layers":       "num_hidden_layers",
            "num_attention_heads":     "num_attention_heads",
            "num_key_value_heads":     "num_key_value_heads",
            "max_position_embeddings": "max_position_embeddings",
            "vocab_size":              "vocab_size",
        }
        for yaml_key, cfg_attr in field_map.items():
            if yaml_key in expected:
                actual = getattr(cfg, cfg_attr)
                assert actual == expected[yaml_key], (
                    f"{cfg_attr}: expected {expected[yaml_key]}, got {actual}"
                )

    def test_forward_pass_loss_finite(self, pretrain_artifact):
        model = pretrain_artifact
        input_ids = torch.randint(0, model.config.vocab_size, (1, 32))
        labels = input_ids.clone()
        with torch.no_grad():
            out = model(input_ids, labels=labels)
        assert torch.isfinite(out.loss), f"Loss is not finite: {out.loss}"

    def test_cached_and_uncached_greedy_generation_match(
        self,
        pretrain_artifact,
    ):
        model = pretrain_artifact
        input_ids = torch.randint(
            4,
            model.config.vocab_size,
            (1, 8),
        )
        kwargs = {
            "max_new_tokens": 4,
            "do_sample": False,
            "eos_token_id": None,
            "pad_token_id": model.config.pad_token_id,
        }
        with torch.no_grad():
            cached = model.generate(input_ids, use_cache=True, **kwargs)
            uncached = model.generate(input_ids, use_cache=False, **kwargs)
        assert torch.equal(cached, uncached)

    def test_loss_decreases_from_random_init(self, pretrain_model_dir):
        """
        Verify training reduced loss by reading train_loss from trainer_state.json.
        The trainer saves this in the latest checkpoint directory.
        """
        import json
        import glob

        # trainer_state.json is saved in checkpoint dirs, not final/
        checkpoint_dir = pretrain_model_dir.parent
        state_files = sorted(glob.glob(str(checkpoint_dir / "checkpoint-*" / "trainer_state.json")))
        if not state_files:
            pytest.skip("trainer_state.json not found in any checkpoint")

        with open(state_files[-1]) as f:
            state = json.load(f)

        log_history = state.get("log_history", [])
        train_losses = [e["loss"] for e in log_history if "loss" in e]
        if not train_losses:
            pytest.skip("No training loss entries in trainer_state.json")

        final_loss = train_losses[-1]
        assert final_loss < 5.0, (
            f"Final training loss {final_loss:.2f} is too high — "
            f"model may not have converged. Random init would be ~10.4."
        )


@requires_stage("tokenize")
class TestPretrainingDataset:
    def test_tokenized_bin_exists(self):
        assert pipeline_path("tokenized", "train.bin").exists(), (
            "tokenized/train.bin not found — run 'make tokenize' first"
        )

    def test_tokenized_metadata_exists(self):
        assert pipeline_path("tokenized", "train.json").exists()

    def test_dataset_loads_and_returns_correct_shape(self):
        from pretrain.data.dataset import PretrainingDataset
        bin_path = pipeline_path("tokenized", "train.bin")
        if not bin_path.exists():
            pytest.skip("tokenized/train.bin not found")

        dataset = PretrainingDataset(bin_path, seq_len=64)
        assert len(dataset) > 0

        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
        assert item["input_ids"].shape == (64,)
        assert item["labels"].shape == (64,)

    def test_labels_match_input_ids_for_causal_lm(self):
        """
        Dataset labels should match input_ids.

        The causal-LM shift happens inside SLMForCausalLM.forward() when
        labels are provided: logits[:, :-1] predict labels[:, 1:]. The dataset
        should not pre-shift labels, otherwise training would be shifted twice.
        """
        from pretrain.data.dataset import PretrainingDataset
        bin_path = pipeline_path("tokenized", "train.bin")
        if not bin_path.exists():
            pytest.skip("tokenized/train.bin not found")

        dataset = PretrainingDataset(bin_path, seq_len=32)
        item = dataset[0]

        assert torch.equal(item["input_ids"], item["labels"])
