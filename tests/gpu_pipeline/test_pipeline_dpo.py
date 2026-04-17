"""
tests/gpu_pipeline/test_pipeline_dpo.py
----------------------------------------
Validates real outputs from 'make dpo-mini'.

Run after: make dpo-mini
Command:   make test-dpo

Checks:
    - DPO data files exist and have correct format (prompt/chosen/rejected)
    - DPO model checkpoint exists and loads
    - Forward pass produces finite loss
    - Model can generate responses
    - DPO data stats file exists and is consistent
"""

import json
import os
from pathlib import Path

import pytest
import torch

from tests.conftest import DATA_DIR, requires_stage, pipeline_path


RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
DPO_MODEL_DIR = RESULTS_DIR / "slm-mini-dpo" / "final"


def skip_if_no_dpo_model():
    return pytest.mark.skipif(
        not DPO_MODEL_DIR.exists(),
        reason=f"DPO model not found at {DPO_MODEL_DIR} — run 'make dpo-mini' first",
    )


def load_model_and_tokenizer(model_dir: Path):
    from transformers import AutoConfig, PreTrainedTokenizerFast
    from model.config import SLMConfig
    from model.model import SLMForCausalLM
    AutoConfig.register("slm", SLMConfig)
    model = SLMForCausalLM.from_pretrained(str(model_dir))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir / "tokenizer"))
    return model, tokenizer


# ── DPO data ───────────────────────────────────────────────────────────────────

class TestDPOData:
    @requires_stage("prepare-dpo")
    def test_dpo_train_exists(self):
        assert pipeline_path("dpo", "train.jsonl").exists()

    @requires_stage("prepare-dpo")
    def test_dpo_val_exists(self):
        assert pipeline_path("dpo", "val.jsonl").exists()

    @requires_stage("prepare-dpo")
    def test_dpo_stats_exists(self):
        assert pipeline_path("dpo", "stats.json").exists()

    @requires_stage("prepare-dpo")
    def test_dpo_records_have_required_fields(self):
        path = pipeline_path("dpo", "train.jsonl")
        failures = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                record = json.loads(line)
                for field in ["prompt", "chosen", "rejected"]:
                    if field not in record:
                        failures.append(f"Line {i}: missing '{field}'")
        assert len(failures) == 0, "\n".join(failures)

    @requires_stage("prepare-dpo")
    def test_dpo_chosen_differs_from_rejected(self):
        """Chosen and rejected must be different — identical pairs are useless."""
        path = pipeline_path("dpo", "train.jsonl")
        identical = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                record = json.loads(line)
                chosen = record.get("chosen", "")
                rejected = record.get("rejected", "")
                if chosen == rejected:
                    identical.append(f"Line {i}: chosen == rejected")
        assert len(identical) == 0, (
            f"{len(identical)} DPO pairs have identical chosen and rejected:\n"
            + "\n".join(identical[:3])
        )

    @requires_stage("prepare-dpo")
    def test_dpo_prompt_is_list_of_messages(self):
        """DPO prompt should be a list of message dicts for trl conversational format."""
        path = pipeline_path("dpo", "train.jsonl")
        with open(path) as f:
            record = json.loads(f.readline())
        prompt = record.get("prompt", [])
        assert isinstance(prompt, list), "DPO prompt should be a list"
        assert len(prompt) >= 1
        assert "role" in prompt[0]
        assert "content" in prompt[0]

    @requires_stage("prepare-dpo")
    def test_dpo_stats_consistent(self):
        stats_path = pipeline_path("dpo", "stats.json")
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["train"] + stats["val"] == stats["total"], (
            "DPO stats: train + val != total"
        )
        assert stats["train"] > 0
        assert stats["val"] > 0


# ── DPO model ──────────────────────────────────────────────────────────────────

class TestDPOModel:
    @skip_if_no_dpo_model()
    def test_dpo_model_dir_exists(self):
        assert DPO_MODEL_DIR.exists()

    @skip_if_no_dpo_model()
    def test_dpo_model_loads(self):
        model, tokenizer = load_model_and_tokenizer(DPO_MODEL_DIR)
        assert model is not None

    @skip_if_no_dpo_model()
    def test_dpo_tokenizer_has_chat_template(self):
        _, tokenizer = load_model_and_tokenizer(DPO_MODEL_DIR)
        assert getattr(tokenizer, "chat_template", None), (
            "DPO model tokenizer has no chat_template"
        )

    @skip_if_no_dpo_model()
    def test_dpo_model_forward_pass_finite_loss(self):
        model, tokenizer = load_model_and_tokenizer(DPO_MODEL_DIR)
        model.eval()

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoding["input_ids"]
        labels = input_ids.clone()

        with torch.no_grad():
            out = model(input_ids, labels=labels)
        assert torch.isfinite(out.loss), f"DPO model loss not finite: {out.loss}"

    @skip_if_no_dpo_model()
    def test_dpo_model_generation_does_not_crash(self):
        model, tokenizer = load_model_and_tokenizer(DPO_MODEL_DIR)
        model.eval()

        messages = [{"role": "user", "content": "Hello."}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoding = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **encoding,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or 0,
            )
        assert output.shape[1] > encoding["input_ids"].shape[1]