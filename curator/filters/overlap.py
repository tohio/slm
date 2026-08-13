"""Full-corpus overlap audits for finalized pretraining JSONL splits."""

from pathlib import Path
from typing import Callable

import orjson

from curator.filters.dedup import exact_hash


def _load_record(line: bytes, path: Path, line_number: int) -> dict:
    try:
        record = orjson.loads(line)
    except Exception as exc:
        raise RuntimeError(f"Invalid JSONL record in {path}:{line_number}") from exc
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Missing non-empty text in {path}:{line_number}")
    return record


def audit_exact_split_overlap(
    train_path: Path,
    val_path: Path,
    *,
    sample_limit: int = 20,
    record_observer: Callable[[str, int, dict], None] | None = None,
) -> dict:
    """Audit normalized exact duplicates within val and across train/val.

    Only validation hashes are retained in memory. Validation is configured as
    a small fraction of the corpus, so this remains bounded relative to the
    full training split while every record in both files is still inspected.
    """
    train_path = Path(train_path)
    val_path = Path(val_path)
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    val_hashes: set[bytes] = set()
    val_duplicate_hashes: set[bytes] = set()
    val_documents = 0
    samples: list[dict] = []

    with open(val_path, "rb", buffering=8 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            record = _load_record(line, val_path, line_number)
            if record_observer is not None:
                record_observer("validation", line_number, record)
            val_documents += 1
            digest = exact_hash(record["text"])
            if digest in val_hashes:
                val_duplicate_hashes.add(digest)
                if len(samples) < sample_limit:
                    samples.append({
                        "kind": "validation_duplicate",
                        "hash": digest.hex(),
                        "val_line": line_number,
                        "source": record.get("source"),
                    })
            else:
                val_hashes.add(digest)

    train_documents = 0
    overlap_documents = 0
    overlap_hashes: set[bytes] = set()
    with open(train_path, "rb", buffering=8 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            record = _load_record(line, train_path, line_number)
            if record_observer is not None:
                record_observer("train", line_number, record)
            train_documents += 1
            digest = exact_hash(record["text"])
            if digest not in val_hashes:
                continue
            overlap_documents += 1
            overlap_hashes.add(digest)
            if len(samples) < sample_limit:
                samples.append({
                    "kind": "train_validation_overlap",
                    "hash": digest.hex(),
                    "train_line": line_number,
                    "source": record.get("source"),
                })

    return {
        "schema_version": 1,
        "algorithm": "sha256-normalized-prefix-128",
        "scope": "full_corpus",
        "train_documents": train_documents,
        "validation_documents": val_documents,
        "validation_unique_hashes": len(val_hashes),
        "validation_duplicate_hashes": len(val_duplicate_hashes),
        "train_validation_overlap_documents": overlap_documents,
        "train_validation_overlap_hashes": len(overlap_hashes),
        "passed": not val_duplicate_hashes and not overlap_hashes,
        "samples": samples,
    }
