"""
tests/gpu_pipeline/test_pipeline_dpo.py
----------------------------------------
Validates real outputs from a DPO run.

Run after: make dpo SIZE=<size>
Command:   make test-dpo SIZE=<size>

Checks:
    - DPO data files exist and have correct format (prompt/chosen/rejected)
    - DPO model checkpoint exists and loads
    - Forward pass produces finite loss
    - Model can generate responses
    - DPO data manifest exists and is consistent
"""

import json
import re
from pathlib import Path

import pytest
import torch

from tests.data_pipeline.helpers import requires_stage, pipeline_path


def load_model_and_tokenizer(model_dir: Path):
    from transformers import AutoConfig, PreTrainedTokenizerFast
    from model.config import SLMConfig
    from model.model import SLMForCausalLM
    AutoConfig.register("slm", SLMConfig)
    model = SLMForCausalLM.from_pretrained(str(model_dir))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir / "tokenizer"))
    return model, tokenizer


def assistant_only_batch(tokenizer, messages):
    batch = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
    )
    mask = batch.pop("assistant_masks")
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    labels = batch["input_ids"].clone()
    labels[mask == 0] = -100
    assert (labels == -100).any()
    assert (labels != -100).any()
    batch["labels"] = labels
    return batch


# ── DPO data ───────────────────────────────────────────────────────────────────

class TestDPOData:
    @requires_stage("prepare-dpo")
    def test_dpo_train_exists(self):
        assert pipeline_path("dpo", "train.jsonl").exists()

    @requires_stage("prepare-dpo")
    def test_dpo_val_exists(self):
        assert pipeline_path("dpo", "val.jsonl").exists()

    @requires_stage("prepare-dpo")
    def test_dpo_manifest_exists(self):
        assert pipeline_path("dpo", "manifest.json").exists()

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
        path = pipeline_path("dpo", "train.jsonl")
        with open(path) as f:
            record = json.loads(f.readline())
        prompt = record.get("prompt", [])
        assert isinstance(prompt, list), "DPO prompt should be a list"
        assert len(prompt) >= 1
        assert "role" in prompt[0]
        assert "content" in prompt[0]

    @requires_stage("prepare-dpo")
    def test_dpo_manifest_is_consistent(self):
        manifest = json.loads(pipeline_path("dpo", "manifest.json").read_text())
        assert manifest["contract"]["source"]["revision"]
        assert manifest["split"]["train_pairs"] > 0
        assert manifest["split"]["validation_pairs"] > 0
        assert manifest["split"]["prompt_overlap"] == 0
        assert manifest["quality"]["reverse_conflicts"] == 0
        assert manifest["token_filter"]["retention_ratio"] >= \
            manifest["contract"]["quality"]["min_token_retention_ratio"]

    @requires_stage("prepare-dpo")
    def test_dpo_prompt_sets_are_disjoint(self):
        def prompts(path):
            values = set()
            with open(path) as handle:
                for line in handle:
                    prompt = json.loads(line)["prompt"]
                    normalized = [
                        {
                            "role": turn["role"],
                            "content": re.sub(
                                r"\s+", " ", turn["content"].strip().lower()
                            ),
                        }
                        for turn in prompt
                    ]
                    values.add(json.dumps(normalized, sort_keys=True))
            return values

        train = prompts(pipeline_path("dpo", "train.jsonl"))
        validation = prompts(pipeline_path("dpo", "val.jsonl"))
        assert train and validation
        assert train.isdisjoint(validation)


# ── DPO model ──────────────────────────────────────────────────────────────────

class TestDPOModel:
    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self, dpo_model_dir):
        if not dpo_model_dir.exists():
            pytest.skip(
                f"DPO model not found at {dpo_model_dir} — "
                f"run 'make dpo SIZE=<size>' first"
            )

    def test_dpo_model_dir_exists(self, dpo_model_dir):
        assert dpo_model_dir.exists()

    def test_dpo_model_loads(self, dpo_model_dir):
        model, tokenizer = load_model_and_tokenizer(dpo_model_dir)
        assert model is not None

    def test_dpo_tokenizer_has_chat_template(self, dpo_model_dir):
        _, tokenizer = load_model_and_tokenizer(dpo_model_dir)
        assert getattr(tokenizer, "chat_template", None), (
            "DPO model tokenizer has no chat_template"
        )

    def test_dpo_model_forward_pass_finite_loss(self, dpo_model_dir):
        model, tokenizer = load_model_and_tokenizer(dpo_model_dir)
        model.eval()

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
        batch = assistant_only_batch(tokenizer, messages)

        with torch.no_grad():
            out = model(**batch)
        assert torch.isfinite(out.loss), f"DPO model loss not finite: {out.loss}"

    def test_dpo_model_generation_does_not_crash(self, dpo_model_dir):
        model, tokenizer = load_model_and_tokenizer(dpo_model_dir)
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
