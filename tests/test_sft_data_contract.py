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


def test_tool_examples_follow_the_existing_grouped_sft_contract():
    import json
    from pathlib import Path
    from config.chat import validate_tool_conversation

    path = Path(__file__).resolve().parents[1] / "finetune/examples/tool_conversations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    records, _ = prepare_records(rows, SOURCE, QUALITY)
    assert len(records) == 6
    for record in records:
        validate_tool_conversation(record["conversations"])
    train, val = grouped_split(records, validation_fraction=0.2, seed=42)
    assert {first_user_prompt(r["conversations"]) for r in train}.isdisjoint(
        first_user_prompt(r["conversations"]) for r in val
    )


@pytest.mark.parametrize("fault", [None, "bytes", "overlap", "stage", "size"])
def test_sft_manifest_checks_full_prepared_identity(tmp_path, fault):
    import json
    from finetune.data.prepare_sft import validate_data_manifest, sha256_file, sha256_json

    train, val, path = (tmp_path / name for name in ("train.jsonl", "val.jsonl", "manifest.json"))
    train.write_text(json.dumps(row(0)) + "\n")
    validation = row(1)
    if fault == "overlap":
        validation["messages"][0]["content"] = "  QUESTION   0  "
    val.write_text(json.dumps(validation) + "\n")
    contract = {"stage": "code" if fault == "stage" else "instruct",
                "size": "350m" if fault == "size" else "mini"}
    manifest = {"schema_version": 1, "contract": contract,
                "contract_sha256": sha256_json(contract), "split": {"prompt_overlap": 0},
                "files": {p.name: {"records": 1, "sha256": sha256_file(p)} for p in (train, val)}}
    path.write_text(json.dumps(manifest))
    if fault == "bytes":
        train.write_text(json.dumps(row(2)) + "\n")
    if fault is None:
        assert validate_data_manifest(path, train, val, stage="instruct", size="mini") == manifest
    else:
        with pytest.raises(RuntimeError):
            validate_data_manifest(path, train, val, stage="instruct", size="mini")


@pytest.mark.parametrize("audit_name", ["sft_run_audit.json", "dpo_run_audit.json"])
def test_posttraining_audit_is_immutable_and_resume_is_explicit(tmp_path, audit_name):
    from config.checkpoints import resolve_training_checkpoint, validate_or_write_run_audit

    contract = {"config": "same", "base_checkpoint": "weights-a", "reference_checkpoint": "ref-a"}
    def check(value, *, write=False, resume=False):
        return validate_or_write_run_audit(tmp_path, value, audit_filename=audit_name,
                                          schema_version=1, resume=resume, write=write)
    check(contract)
    assert not list(tmp_path.iterdir())  # preflight creates no audit
    path = check(contract, write=True)
    original = path.read_bytes()
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-10").mkdir()
    assert resolve_training_checkpoint(tmp_path, resume=True, audit_filename=audit_name).name == "checkpoint-10"
    with pytest.raises(RuntimeError, match="already contains"):
        resolve_training_checkpoint(tmp_path, resume=False, audit_filename=audit_name)
    for changed in ({**contract, "config": "changed"}, {**contract, "reference_checkpoint": "ref-b"}):
        with pytest.raises(RuntimeError, match="contract mismatch"):
            check(changed, write=True, resume=True)
    check(contract, resume=True)
    assert path.read_bytes() == original
