"""Unit tests for the external SFT data consumer contract."""

import pytest

from finetune.data.prepare_sft import (
    first_user_prompt,
    grouped_split,
    prepare_records,
)


SOURCE = {"format": "conversational", "source": "fixture"}
QUALITY = {
    "max_invalid_fraction": 0.05,
    "max_duplicate_fraction": 0.05,
    "min_unique_prompt_ratio": 0.90,
}


def row(index: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"Question {index}"},
            {"role": "assistant", "content": f"Answer {index}"},
        ]
    }


def test_grouped_split_has_no_prompt_leakage():
    records, _ = prepare_records([row(index) for index in range(100)], SOURCE, QUALITY)
    train, val = grouped_split(records, validation_fraction=0.10, seed=42)

    train_prompts = {first_user_prompt(item["conversations"]) for item in train}
    val_prompts = {first_user_prompt(item["conversations"]) for item in val}

    assert len(train) == 90
    assert len(val) == 10
    assert train_prompts.isdisjoint(val_prompts)


def test_duplicate_heavy_source_is_rejected_instead_of_silently_repaired():
    rows = [row(index) for index in range(10)]
    rows.extend([row(0)] * 10)

    with pytest.raises(RuntimeError, match="duplicates"):
        prepare_records(rows, SOURCE, QUALITY)


def test_invalid_validation_fraction_is_rejected():
    records, _ = prepare_records([row(0), row(1)], SOURCE, QUALITY)
    with pytest.raises(ValueError, match="validation_fraction"):
        grouped_split(records, validation_fraction=0.0, seed=42)
