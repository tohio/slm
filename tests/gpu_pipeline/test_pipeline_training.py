"""
tests/gpu_pipeline/test_pipeline_training.py
---------------------------------------------
Validates real outputs from 'make pretrain-mini'.

Run after: make pretrain-mini
Command:   make test-training

Checks:
    - Checkpoint directory exists
    - final/ model can be loaded
    - Tokenizer is present alongside the model
    - Forward pass produces finite loss
    - Loss is lower than random initialization (~log(32000) ≈ 10.4)
    - Config matches gpt_mini.yaml
"""

import os
from pathlib import Path

import pytest
import torch

from tests.conftest import DATA_DIR, requires_stage, pipeline_path


pytestmark = requires_stage("pretrain-mini")

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
MINI_MODEL_DIR = RESULTS_DIR / "slm-mini" / "final"


def skip_if_no_model():
    return pytest.mark.skipif(
        not MINI_MODEL_DIR.exists(),
        reason=f"Mini model not found at {MINI_MODEL_DIR} — run 'make pretrain-mini' first",
    )


class TestPretrainMiniOutputs:
    @skip_if_no_model()
    def test_final_dir_exists(self):
        assert MINI_MODEL_DIR.exists()

    @skip_if_no_model()
    def test_model_files_exist(self):
        assert (MINI_MODEL_DIR / "config.json").exists()
        assert (MINI_MODEL_DIR / "model.safetensors").exists() or \
               (MINI_MODEL_DIR / "pytorch_model.bin").exists()

    @skip_if_no_model()
    def test_tokenizer_saved_alongside_model(self):
        assert (MINI_MODEL_DIR / "tokenizer" / "tokenizer_config.json").exists(), (
            "Tokenizer not found alongside model — "
            "pretrain/train.py should copy tokenizer to final/"
        )

    @skip_if_no_model()
    def test_model_loads(self):
        from transformers import AutoConfig
        from model.config import SLMConfig
        from model.model import SLMForCausalLM

        AutoConfig.register("slm", SLMConfig)
        model = SLMForCausalLM.from_pretrained(str(MINI_MODEL_DIR))
        assert model is not None

    @skip_if_no_model()
    def test_config_matches_mini_yaml(self):
        from model.config import SLMConfig
        from model.model import SLMForCausalLM
        from transformers import AutoConfig
        AutoConfig.register("slm", SLMConfig)
        model = SLMForCausalLM.from_pretrained(str(MINI_MODEL_DIR))
        cfg = model.config
        assert cfg.hidden_size == 384
        assert cfg.num_hidden_layers == 6
        assert cfg.num_attention_heads == 6
        assert cfg.num_key_value_heads == 2
        assert cfg.max_position_embeddings == 1024
        assert cfg.vocab_size == 32000

    @skip_if_no_model()
    def test_forward_pass_loss_finite(self):
        from transformers import AutoConfig
        from model.config import SLMConfig
        from model.model import SLMForCausalLM
        AutoConfig.register("slm", SLMConfig)
        model = SLMForCausalLM.from_pretrained(str(MINI_MODEL_DIR))
        model.eval()

        input_ids = torch.randint(0, 32000, (1, 32))
        labels = input_ids.clone()
        with torch.no_grad():
            out = model(input_ids, labels=labels)
        assert torch.isfinite(out.loss), f"Loss is not finite: {out.loss}"

    @skip_if_no_model()
    def test_loss_below_random_initialization(self):
        """
        After 5000 steps the model should have learned something.
        Random init loss ≈ log(32000) ≈ 10.4.
        After training we expect loss < 6.0 on training data samples.

        Samples directly from the tokenized binary so this test works
        on the GPU instance without needing curated/train.jsonl.
        """
        from transformers import AutoConfig
        from model.config import SLMConfig
        from model.model import SLMForCausalLM
        import numpy as np
        AutoConfig.register("slm", SLMConfig)

        bin_path = pipeline_path("tokenized", "train.bin")
        if not bin_path.exists():
            pytest.skip("tokenized/train.bin not found")

        model = SLMForCausalLM.from_pretrained(str(MINI_MODEL_DIR))
        model.eval()

        # Sample from the middle of the training data rather than the start.
        # The first few tokens of the binary may be edge cases (document
        # boundaries, unusual tokens) that produce artificially high loss.
        data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        seq_len = 128
        mid = len(data) // 2
        input_ids = torch.from_numpy(data[mid:mid + seq_len].astype("int64")).unsqueeze(0)
        labels = input_ids.clone()

        with torch.no_grad():
            out = model(input_ids, labels=labels)

        loss = out.loss.item()
        assert loss < 6.0, (
            f"Loss {loss:.2f} is too high — model may not have learned. "
            f"Random init would be ~10.4. Check training logs for convergence."
        )


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

    def test_labels_are_input_ids_shifted_left(self):
        from pretrain.data.dataset import PretrainingDataset
        bin_path = pipeline_path("tokenized", "train.bin")
        if not bin_path.exists():
            pytest.skip("tokenized/train.bin not found")

        dataset = PretrainingDataset(bin_path, seq_len=32)
        item = dataset[0]
        next_item = dataset[1] if len(dataset) > 1 else None

        # labels[i] should equal input_ids[i+1]
        # Check by verifying input_ids and labels overlap by seq_len-1
        assert torch.equal(item["input_ids"][1:], item["labels"][:-1])