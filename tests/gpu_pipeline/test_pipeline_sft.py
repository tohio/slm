"""
tests/gpu_pipeline/test_pipeline_sft.py
----------------------------------------
Validates real outputs from chat SFT and code SFT runs.

Run after: make sft SIZE=<size> && make sft-code SIZE=<size>
Command:   make test-sft-chat SIZE=<size> && make test-sft-code SIZE=<size>

Checks for each SFT stage:
    - Checkpoint directory exists with final/
    - Model loads correctly
    - Tokenizer present alongside model
    - Forward pass with chat-formatted input produces finite loss
    - Chat template still works after fine-tuning
    - SFT data files exist and have correct format
"""

import json
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


# ── SFT data ───────────────────────────────────────────────────────────────────

class TestSFTData:
    @requires_stage("prepare-sft")
    def test_chat_train_exists(self):
        assert pipeline_path("sft", "chat", "train.jsonl").exists()

    @requires_stage("prepare-sft")
    def test_chat_val_exists(self):
        assert pipeline_path("sft", "chat", "val.jsonl").exists()

    @requires_stage("prepare-sft")
    def test_code_train_exists(self):
        assert pipeline_path("sft", "code", "train.jsonl").exists()

    @requires_stage("prepare-sft")
    def test_code_val_exists(self):
        assert pipeline_path("sft", "code", "val.jsonl").exists()

    @requires_stage("prepare-sft")
    def test_chat_data_has_conversations_field(self):
        path = pipeline_path("sft", "chat", "train.jsonl")
        with open(path) as f:
            record = json.loads(f.readline())
        assert "conversations" in record
        assert isinstance(record["conversations"], list)
        assert len(record["conversations"]) >= 2

    @requires_stage("prepare-sft")
    def test_chat_data_ends_with_assistant_turn(self):
        path = pipeline_path("sft", "chat", "train.jsonl")
        failures = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                record = json.loads(line)
                turns = record.get("conversations", [])
                if turns and turns[-1]["role"] != "assistant":
                    failures.append(f"Line {i}: last turn is '{turns[-1]['role']}'")
        assert len(failures) == 0, (
            f"{len(failures)} chat examples don't end with assistant turn:\n"
            + "\n".join(failures[:3])
        )

    @requires_stage("prepare-sft")
    def test_chat_data_contains_response_control_examples(self):
        """
        Chat SFT should include the generated response-control supplement.

        This catches regressions where prepare_sft.py still downloads
        OpenHermes but silently stops adding local behavior-control records.
        """
        path = pipeline_path("sft", "chat", "train.jsonl")
        found = 0
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("source") == "response_control":
                    found += 1
                    if found >= 10:
                        break

        assert found > 0, (
            "No response_control examples found in chat SFT train.jsonl. "
            "Run make prepare-sft after ensuring response_control.py is wired "
            "into prepare_sft.py."
        )

    @requires_stage("prepare-sft")
    def test_response_control_examples_have_expected_types(self):
        """
        Response-control records should cover the behavior categories that
        sanity_eval.py checks: direct facts, AI concepts, factual restraint,
        and concise stopping.
        """
        path = pipeline_path("sft", "chat", "train.jsonl")
        expected = {
            "simple_factual",
            "ai_concept",
            "factual_restraint",
            "concise_answer",
        }
        seen = set()

        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("source") != "response_control":
                    continue
                sft_type = record.get("sft_type")
                if sft_type:
                    seen.add(sft_type)
                if expected <= seen:
                    break

        missing = expected - seen
        assert not missing, (
            f"Missing response-control categories in chat SFT data: {missing}. "
            f"Seen: {seen}"
        )

    @requires_stage("prepare-sft")
    def test_response_control_examples_do_not_include_arithmetic(self):
        """
        Arithmetic was moved out of response-control SFT and into the
        synthetic_arithmetic pretraining source. Response-control should
        not be the mechanism for teaching base arithmetic.
        """
        path = pipeline_path("sft", "chat", "train.jsonl")
        bad = 0

        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("source") != "response_control":
                    continue
                if record.get("sft_type") == "arithmetic":
                    bad += 1
                    break

        assert bad == 0, (
            "Found arithmetic response-control examples. Arithmetic should "
            "come from the synthetic_arithmetic pretraining source instead."
        )

    @requires_stage("prepare-sft")
    def test_code_data_has_code_system_prompt(self):
        path = pipeline_path("sft", "code", "train.jsonl")
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                turns = record.get("conversations", [])
                if turns and turns[0]["role"] == "system":
                    assert "code" in turns[0]["content"].lower() or \
                           "programming" in turns[0]["content"].lower(), (
                        "Code SFT system prompt doesn't mention code/programming"
                    )
                    break


# ── Chat SFT model ─────────────────────────────────────────────────────────────

class TestChatSFTModel:
    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self, chat_sft_model_dir):
        if not chat_sft_model_dir.exists():
            pytest.skip(
                f"Chat SFT model not found at {chat_sft_model_dir} — "
                f"run 'make sft SIZE=<size>' first"
            )

    def test_chat_model_dir_exists(self, chat_sft_model_dir):
        assert chat_sft_model_dir.exists()

    def test_chat_model_loads(self, chat_sft_model_dir):
        model, tokenizer = load_model_and_tokenizer(chat_sft_model_dir)
        assert model is not None

    def test_chat_tokenizer_has_chat_template(self, chat_sft_model_dir):
        _, tokenizer = load_model_and_tokenizer(chat_sft_model_dir)
        assert getattr(tokenizer, "chat_template", None), (
            "Tokenizer saved with chat model has no chat_template"
        )

    def test_chat_model_forward_pass_finite_loss(self, chat_sft_model_dir):
        model, tokenizer = load_model_and_tokenizer(chat_sft_model_dir)
        model.eval()

        messages = [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 = 4."},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoding["input_ids"]
        labels = input_ids.clone()

        with torch.no_grad():
            out = model(input_ids, labels=labels)
        assert torch.isfinite(out.loss), f"Chat SFT loss not finite: {out.loss}"

    def test_chat_model_generation_does_not_crash(self, chat_sft_model_dir):
        model, tokenizer = load_model_and_tokenizer(chat_sft_model_dir)
        model.eval()

        messages = [{"role": "user", "content": "Say hello."}]
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


# ── Code SFT model ─────────────────────────────────────────────────────────────

class TestCodeSFTModel:
    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self, code_sft_model_dir):
        if not code_sft_model_dir.exists():
            pytest.skip(
                f"Code SFT model not found at {code_sft_model_dir} — "
                f"run 'make sft-code SIZE=<size>' first"
            )

    def test_code_model_dir_exists(self, code_sft_model_dir):
        assert code_sft_model_dir.exists()

    def test_code_model_loads(self, code_sft_model_dir):
        model, tokenizer = load_model_and_tokenizer(code_sft_model_dir)
        assert model is not None

    def test_code_model_forward_pass_finite_loss(self, code_sft_model_dir):
        model, tokenizer = load_model_and_tokenizer(code_sft_model_dir)
        model.eval()

        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write a Python hello world."},
            {"role": "assistant", "content": "print('Hello, world!')"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoding["input_ids"]
        labels = input_ids.clone()

        with torch.no_grad():
            out = model(input_ids, labels=labels)
        assert torch.isfinite(out.loss), f"Code SFT loss not finite: {out.loss}"

    def test_code_special_tokens_in_vocab(self, code_sft_model_dir):
        _, tokenizer = load_model_and_tokenizer(code_sft_model_dir)
        assert tokenizer.convert_tokens_to_ids("<|code|>") != tokenizer.unk_token_id
        assert tokenizer.convert_tokens_to_ids("<|endofcode|>") != tokenizer.unk_token_id