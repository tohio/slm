"""
tests/gpu_pipeline/test_pipeline_sft.py
----------------------------------------
Validates real outputs from 'make sft-mini' and 'make sft-code-mini'.

Run after: make sft-mini && make sft-code-mini
Command:   make test-sft-chat && make test-sft-code

Checks for each SFT stage:
    - Checkpoint directory exists with final/
    - Model loads correctly
    - Tokenizer present alongside model
    - Forward pass with chat-formatted input produces finite loss
    - Chat template still works after fine-tuning
    - SFT data files exist and have correct format
"""

import json
import os
from pathlib import Path

import pytest
import torch

from tests.conftest import DATA_DIR, requires_stage, pipeline_path


RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
CHAT_MODEL_DIR = RESULTS_DIR / "slm-mini-chat" / "final"
CODE_MODEL_DIR = RESULTS_DIR / "slm-mini-chat-code" / "final"


def skip_if_no_chat_model():
    return pytest.mark.skipif(
        not CHAT_MODEL_DIR.exists(),
        reason=f"Chat SFT model not found at {CHAT_MODEL_DIR} — run 'make sft-mini' first",
    )


def skip_if_no_code_model():
    return pytest.mark.skipif(
        not CODE_MODEL_DIR.exists(),
        reason=f"Code SFT model not found at {CODE_MODEL_DIR} — run 'make sft-code-mini' first",
    )


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
    @skip_if_no_chat_model()
    def test_chat_model_dir_exists(self):
        assert CHAT_MODEL_DIR.exists()

    @skip_if_no_chat_model()
    def test_chat_model_loads(self):
        model, tokenizer = load_model_and_tokenizer(CHAT_MODEL_DIR)
        assert model is not None

    @skip_if_no_chat_model()
    def test_chat_tokenizer_has_chat_template(self):
        _, tokenizer = load_model_and_tokenizer(CHAT_MODEL_DIR)
        assert getattr(tokenizer, "chat_template", None), (
            "Tokenizer saved with chat model has no chat_template"
        )

    @skip_if_no_chat_model()
    def test_chat_model_forward_pass_finite_loss(self):
        model, tokenizer = load_model_and_tokenizer(CHAT_MODEL_DIR)
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

    @skip_if_no_chat_model()
    def test_chat_model_generation_does_not_crash(self):
        model, tokenizer = load_model_and_tokenizer(CHAT_MODEL_DIR)
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
    @skip_if_no_code_model()
    def test_code_model_dir_exists(self):
        assert CODE_MODEL_DIR.exists()

    @skip_if_no_code_model()
    def test_code_model_loads(self):
        model, tokenizer = load_model_and_tokenizer(CODE_MODEL_DIR)
        assert model is not None

    @skip_if_no_code_model()
    def test_code_model_forward_pass_finite_loss(self):
        model, tokenizer = load_model_and_tokenizer(CODE_MODEL_DIR)
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

    @skip_if_no_code_model()
    def test_code_special_tokens_in_vocab(self):
        _, tokenizer = load_model_and_tokenizer(CODE_MODEL_DIR)
        assert tokenizer.convert_tokens_to_ids("<|code|>") != tokenizer.unk_token_id
        assert tokenizer.convert_tokens_to_ids("<|endofcode|>") != tokenizer.unk_token_id