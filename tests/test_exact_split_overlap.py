"""Focused tests for the finalized-corpus exact split-overlap gate."""

import orjson
import pytest

from curator.filters.overlap import audit_exact_split_overlap


def _write_jsonl(path, records):
    with open(path, "wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")


def test_exact_split_overlap_passes_disjoint_splits(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(train, [{"text": "Training document.", "source": "wikipedia"}])
    _write_jsonl(val, [{"text": "Validation document.", "source": "pg19"}])

    report = audit_exact_split_overlap(train, val)

    assert report["passed"] is True
    assert report["train_documents"] == 1
    assert report["validation_documents"] == 1
    assert report["validation_duplicate_hashes"] == 0
    assert report["train_validation_overlap_hashes"] == 0


def test_exact_split_overlap_uses_dedup_normalization(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(train, [{"text": "Same, DOCUMENT!", "source": "fineweb"}])
    _write_jsonl(val, [{"text": "same document", "source": "common_crawl"}])

    report = audit_exact_split_overlap(train, val)

    assert report["passed"] is False
    assert report["train_validation_overlap_documents"] == 1
    assert report["train_validation_overlap_hashes"] == 1
    assert report["samples"][0]["kind"] == "train_validation_overlap"


def test_exact_split_overlap_rejects_validation_duplicates(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(train, [{"text": "Unique train text", "source": "wikipedia"}])
    _write_jsonl(
        val,
        [
            {"text": "Repeated validation text", "source": "pg19"},
            {"text": "repeated validation text!", "source": "pg19"},
        ],
    )

    report = audit_exact_split_overlap(train, val)

    assert report["passed"] is False
    assert report["validation_duplicate_hashes"] == 1
    assert report["validation_unique_hashes"] == 1


def test_exact_split_overlap_fails_closed_on_invalid_record(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_bytes(b'{"source":"wikipedia"}\n')
    _write_jsonl(val, [{"text": "Valid text", "source": "pg19"}])

    with pytest.raises(RuntimeError, match="Missing non-empty text"):
        audit_exact_split_overlap(train, val)
