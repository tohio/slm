"""Cheap unit tests for the controlled SFT comparison's data boundary."""

import torch

from scripts.sft_model_comparison import (
    CompletionOnlyCollator,
    _tokenized_training_row,
)
from tests.test_trl_smoke import _tokenizer


def test_explicit_completion_labels_retain_eos_and_mask_prompt():
    tokenizer = _tokenizer()
    row = _tokenized_training_row(
        tokenizer,
        user_text="Say hello",
        assistant_text="Hello world",
        max_length=64,
    )

    assert row is not None
    assert len(row["input_ids"]) == len(row["labels"])
    assert -100 in row["labels"]
    supervised = [token for token in row["labels"] if token != -100]
    assert supervised
    assert tokenizer.eos_token_id in supervised


def test_explicit_token_budget_rejects_complete_example():
    tokenizer = _tokenizer()
    row = _tokenized_training_row(
        tokenizer,
        user_text="Say hello",
        assistant_text="Hello world",
        max_length=2,
    )

    assert row is None


def test_collator_masks_padding_from_loss():
    collator = CompletionOnlyCollator(pad_token_id=0)
    batch = collator(
        [
            {
                "input_ids": [1, 5, 6, 2],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, -100, 6, 2],
            },
            {
                "input_ids": [1, 6, 2],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 6, 2],
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"][1, -1].item() == 0
    assert batch["labels"][1, -1].item() == -100
    assert all(value.dtype == torch.long for value in batch.values())
