"""Portable frozen-holdout identities, shared by curation and training.

No curation/ML imports: GPU hosts must not need DataTrove, KenLM or FastText
in order to verify or restore an already curated dataset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from curator.state import atomic_write_json, stable_digest

SPLITS = ("train", "val", "test")
CONTRACT_NAME = "test_contract.json"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_identity(path: Path) -> dict:
    """Full byte identity and physical record accounting, in bounded memory."""
    digest = hashlib.sha256()
    documents = characters = byte_count = 0
    sources: dict[str, dict[str, int]] = {}
    with Path(path).open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            byte_count += len(line)
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"Invalid JSONL: {path}:{line_number}") from exc
            text = record.get("text") if isinstance(record, dict) else None
            source = record.get("source") if isinstance(record, dict) else None
            if not isinstance(text, str) or not text.strip() or not isinstance(source, str) or not source:
                raise RuntimeError(f"Missing text/source: {path}:{line_number}")
            documents += 1
            characters += len(text)
            row = sources.setdefault(source, {"documents": 0, "characters": 0})
            row["documents"] += 1
            row["characters"] += len(text)
    return {"sha256": digest.hexdigest(), "bytes": byte_count,
            "documents": documents, "characters": characters,
            "sources": dict(sorted(sources.items()))}


def write_contract(directory: Path, contract: dict) -> dict:
    payload = {"contract": contract, "sha256": stable_digest(contract)}
    atomic_write_json(Path(directory) / CONTRACT_NAME, payload)
    return payload


def load_contract(directory: Path, *, stage: str | None = None) -> dict:
    path = Path(directory) / CONTRACT_NAME
    if not path.is_file():
        raise RuntimeError(f"Missing frozen holdout contract: {path}. Restore it; do not re-split test.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("contract"), dict):
        raise RuntimeError(f"Malformed frozen holdout contract: {path}")
    contract = payload["contract"]
    if payload.get("sha256") != stable_digest(contract):
        raise RuntimeError(f"Frozen holdout contract checksum mismatch: {path}")
    if contract.get("schema_version") != SCHEMA_VERSION or contract.get("status") != "frozen":
        raise RuntimeError(f"Unsupported or unfinished frozen holdout contract: {path}")
    if set(contract.get("splits", {})) != set(SPLITS):
        raise RuntimeError(f"Frozen contract must contain train, val and test: {path}")
    if stage is not None and contract.get("stage") != stage:
        raise RuntimeError(f"Expected {stage} frozen holdout contract: {path}")
    for split, identity in contract["splits"].items():
        if (not isinstance(identity, dict)
                or not isinstance(identity.get("bytes"), int) or identity["bytes"] <= 0
                or not isinstance(identity.get("documents"), int)
                or isinstance(identity["documents"], bool)
                or identity["documents"] <= 0
                or not isinstance(identity.get("sha256"), str)
                or len(identity["sha256"]) != 64):
            raise RuntimeError(f"Invalid {split} identity in {path}")
    return payload


def verify_jsonl_contract(directory: Path, *, include_train: bool = True,
                          stage: str | None = None) -> dict:
    payload = load_contract(directory, stage=stage)
    for split in SPLITS if include_train else ("val", "test"):
        path = Path(directory) / f"{split}.jsonl"
        expected = payload["contract"]["splits"][split]
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Frozen {split} identity changed: {path}. Restore the original artifacts.")
    membership_sha = payload["contract"].get("test_membership_sha256")
    if membership_sha:
        path = Path(directory) / "test_membership.jsonl"
        if not path.is_file() or sha256_file(path) != membership_sha:
            raise RuntimeError("Frozen test membership index changed; restore the original metadata")
    return payload
