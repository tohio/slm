"""Unit tests for the external DPO preference-data contract."""

import pytest

from alignment.data.prepare_dpo import (
    apply_token_budget,
    canonical_prompt,
    grouped_split,
    prepare_records,
)


SOURCE = {
    "format": "conversational_preference",
    "source": "fixture",
}
QUALITY = {
    "max_invalid_fraction": 0.05,
    "max_duplicate_fraction": 0.05,
    "min_unique_prompt_ratio": 0.90,
    "max_reverse_conflicts": 0,
    "min_token_retention_ratio": 0.80,
}


def row(index: int, chosen: str | None = None, rejected: str | None = None) -> dict:
    prompt = [
        {"role": "user", "content": f"Question {index}"},
    ]
    return {
        "chosen": prompt + [
            {"role": "assistant", "content": chosen or f"Good answer {index}"}
        ],
        "rejected": prompt + [
            {"role": "assistant", "content": rejected or f"Bad answer {index}"}
        ],
    }


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered = "<BOS>"
        for message in messages:
            rendered += f"<{message['role']}>{message['content']}<END>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(character) for character in rendered]


def test_grouped_split_has_no_prompt_leakage():
    records, _ = prepare_records([row(index) for index in range(100)], SOURCE, QUALITY)
    train, validation = grouped_split(records, validation_fraction=0.10, seed=42)

    train_prompts = {canonical_prompt(item["prompt"]) for item in train}
    validation_prompts = {canonical_prompt(item["prompt"]) for item in validation}

    assert len(train) == 90
    assert len(validation) == 10
    assert train_prompts.isdisjoint(validation_prompts)


def test_duplicate_heavy_preferences_are_rejected():
    rows = [row(index) for index in range(10)]
    rows.extend([row(0)] * 10)
    with pytest.raises(RuntimeError, match="duplicates"):
        prepare_records(rows, SOURCE, QUALITY)


def test_reversed_preference_conflict_is_rejected():
    rows = [row(index) for index in range(20)]
    rows.append(row(50, chosen="A", rejected="B"))
    rows.append(row(50, chosen="B", rejected="A"))
    with pytest.raises(RuntimeError, match="reversed preference"):
        prepare_records(rows, SOURCE, QUALITY)


def test_token_budget_uses_full_chat_rendering():
    records, _ = prepare_records([row(index) for index in range(10)], SOURCE, QUALITY)
    kept, stats = apply_token_budget(
        records,
        CharacterTokenizer(),
        max_prompt_tokens=200,
        max_total_tokens=300,
    )
    assert len(kept) == 10
    assert stats["prefix_mismatch"] == 0


def test_string_preference_schema_can_replace_current_source():
    rows = [
        {
            "id": f"pair-{index}",
            "prompt": f"Question {index}",
            "chosen": f"Good answer {index}",
            "rejected": f"Bad answer {index}",
        }
        for index in range(10)
    ]
    records, stats = prepare_records(rows, SOURCE, QUALITY)
    assert len(records) == 10
    assert records[0]["source_id"].startswith("pair-")
    assert stats["invalid_rows"] == 0
