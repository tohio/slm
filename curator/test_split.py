"""Establish a test split during curation without reshuffling validation.

This belongs to the existing curated stage, not a separate artifact stage.
All work happens beside the original directory and is promoted only after all
three existing exact/MinHash pair policies and physical counts pass.
"""
from __future__ import annotations

import heapq
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from config.holdout import (
    CONTRACT_NAME, SCHEMA_VERSION, SPLITS, jsonl_identity,
    verify_jsonl_contract, write_contract,
)
from curator.state import load_manifest, manifest_outputs_match, stable_digest, write_manifest

log = logging.getLogger(__name__)


def _records(path: Path):
    from curator.filters.overlap import _load_record
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        for number, line in enumerate(handle, 1):
            yield line, _load_record(line, path, number)


def _remove_exact(candidate: Path, reference: Path, *, dedup_candidate: bool = False) -> dict:
    """Use the existing normalized exact hash, always preserving the reference."""
    from curator.filters.dedup import exact_hash
    protected = {exact_hash(record["text"]) for _, record in _records(reference)}
    seen = set()
    before = removed = removed_characters = 0
    tmp = candidate.with_suffix(".exact.tmp")
    with tmp.open("wb") as handle:
        for line, record in _records(candidate):
            before += 1
            key = exact_hash(record["text"])
            if key in protected or (dedup_candidate and key in seen):
                removed += 1
                removed_characters += len(record["text"])
            else:
                handle.write(line)
                if dedup_candidate:
                    seen.add(key)
        handle.flush()
        os.fsync(handle.fileno())
    after = jsonl_identity(tmp)["documents"]
    if before - after != removed:
        raise RuntimeError("Exact remediation counts do not match physical removals")
    tmp.replace(candidate)
    return {"audited_documents": before, "retained_documents": after,
            "removed_documents": removed, "removed_characters": removed_characters}


def _near_pair(candidate: Path, reference: Path, scratch: Path, workers: int | None) -> dict:
    from curator.filters.near_overlap import audit_minhash_split_overlap
    before = jsonl_identity(candidate)
    report = audit_minhash_split_overlap(candidate, reference, scratch, workers=workers)
    after = jsonl_identity(candidate)
    if (not report.get("passed")
            or before["documents"] - after["documents"] != report.get("removed_train_documents")
            or before["characters"] - after["characters"] != report.get("removed_train_characters")
            or report.get("train_documents") != after["documents"]):
        raise RuntimeError("MinHash remediation counts do not match physical removals")
    return report


def establish_test_split(directory: Path, *, size: str, seed: int = 42,
                      test_fraction: float = 0.005, workers: int | None = None) -> dict:
    from curator.filters.dedup import exact_hash, MINHASH_CONTRACT
    from curator.filters.overlap import audit_exact_split_overlap

    directory = Path(directory)
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must lie strictly between 0 and 0.5")
    if (directory / CONTRACT_NAME).exists():
        existing = verify_jsonl_contract(directory, stage="curated")
        policy = existing["contract"]["selection"]
        if (policy["seed"] != seed or policy["target_total_fraction"] != test_fraction
                or existing["contract"]["size"] != size):
            raise RuntimeError("Test selection cannot change; use a new dataset run")
        if not manifest_outputs_match(directory, output_pattern="*.json*"):
            raise RuntimeError("Test curated stage is not manifest-complete; restore it")
        log.info("Reusing verified holdout train/val/test membership: %s", directory)
        return existing

    stage = directory.with_name(f".{directory.name}.split-pending")
    backup = directory.with_name(f".{directory.name}.before-split")
    if stage.exists() or backup.exists():
        raise RuntimeError(
            f"Interrupted test split transaction: {stage} / {backup}. "
            "Recover the complete directory before retrying; do not reselect test."
        )
    if (directory / "test.jsonl").exists():
        raise RuntimeError("test.jsonl exists without its holdout contract; restore its metadata")
    if not manifest_outputs_match(directory, output_pattern="*.json*"):
        raise RuntimeError(f"Curated inputs are not manifest-complete: {directory}")
    manifest = load_manifest(directory)
    origin = {split: jsonl_identity(directory / f"{split}.jsonl") for split in ("train", "val")}
    if not origin["train"]["documents"] or not origin["val"]["documents"]:
        raise RuntimeError("Both original train and validation must be non-empty")
    count = max(1, round(sum(x["documents"] for x in origin.values()) * test_fraction))
    if count >= origin["train"]["documents"]:
        raise RuntimeError("Not enough training documents to create a non-empty holdout test")

    # Bottom-k stable hash groups: memory is proportional to the small holdout,
    # not to the training corpus. Normalized duplicates stay on the same side.
    heap: list[tuple[int, bytes]] = []
    salt = str(seed).encode("ascii") + b"\0"
    for _, record in _records(directory / "train.jsonl"):
        key = exact_hash(record["text"])
        priority = int.from_bytes(hashlib.sha256(salt + key).digest(), "big")
        item = (-priority, key)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = {key for _, key in heap}

    # Hard links avoid copying the existing large corpus. Every changed file
    # is replaced, never edited through a hard link to the original directory.
    shutil.copytree(directory, stage, copy_function=os.link)
    promoted = False
    try:
        (stage / "train.jsonl").unlink()
        with (stage / "train.jsonl").open("wb") as train, (stage / "test.jsonl").open("wb") as test:
            for line, record in _records(directory / "train.jsonl"):
                (test if exact_hash(record["text"]) in selected else train).write(line)
            for handle in (train, test):
                handle.flush()
                os.fsync(handle.fileno())

        scratch = stage / ".near-audit"
        reports = {}
        # Priority: existing val > newly selected test > train. Never modify val.
        for candidate, reference, dedup_candidate in (("test", "val", True), ("train", "val", False), ("train", "test", False)):
            key = f"{candidate}_{reference}"
            candidate_path, reference_path = stage / f"{candidate}.jsonl", stage / f"{reference}.jsonl"
            exact_removals = _remove_exact(candidate_path, reference_path, dedup_candidate=dedup_candidate)
            if not jsonl_identity(candidate_path)["documents"]:
                raise RuntimeError(f"{candidate} became empty after exact decontamination")
            near = _near_pair(candidate_path, reference_path, scratch / key, workers)
            if not jsonl_identity(candidate_path)["documents"]:
                raise RuntimeError(f"{candidate} became empty after MinHash decontamination")
            reports[key] = {"candidate": candidate, "protected": reference,
                            "exact_removals": exact_removals, "near": near}
        shutil.rmtree(scratch)
        for candidate, reference in (("train", "val"), ("train", "test"), ("test", "val")):
            report = audit_exact_split_overlap(stage / f"{candidate}.jsonl", stage / f"{reference}.jsonl")
            if not report["passed"]:
                raise RuntimeError(f"Final exact overlap gate failed: {candidate}/{reference}")
            reports[f"{candidate}_{reference}"]["final_exact"] = report
        identities = {split: jsonl_identity(stage / f"{split}.jsonl") for split in SPLITS}
        if identities["val"] != origin["val"]:
            raise RuntimeError("Validation changed during test split; refusing promotion")
        removed = sum(r["exact_removals"]["removed_documents"] + r["near"]["removed_train_documents"] for r in reports.values())
        if sum(x["documents"] for x in origin.values()) != sum(x["documents"] for x in identities.values()) + removed:
            raise RuntimeError("Test split total/removal accounting mismatch")
        membership = hashlib.sha256()
        membership_path = stage / "test_membership.jsonl"
        with membership_path.open("wb") as handle:
            for number, (_, record) in enumerate(_records(stage / "test.jsonl"), 1):
                row = {"line": number, "hash": exact_hash(record["text"]).hex(),
                       "source": record["source"], "id": record.get("id")}
                line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
                membership.update(line)
                handle.write(line)
        holdout = write_contract(stage, {
            "schema_version": SCHEMA_VERSION, "status": "established", "stage": "curated",
            "size": size, "selection": {"algorithm": "bottom_k_sha256_seed_normalized_exact_hash",
                "seed": seed, "target_total_fraction": test_fraction, "candidate_groups": len(selected)},
            "origin": {"splits": origin, "blend_manifest_sha256": stable_digest(manifest),
                       "blend_contract": manifest["contract"]},
            "splits": identities, "test_membership_sha256": membership.hexdigest(),
            "minhash": MINHASH_CONTRACT, "pair_audits": reports,
            "training_contract": {"train": "gradient_updates", "val": "training_time_selection",
                                  "test": "final_only"},
        })
        stats_path = stage / "blend_stats.json"
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text())
            stats["before_test_split"] = {k: stats.get(k) for k in ("total_documents", "train_documents", "val_documents", "source_mix")}
            stats.update({"total_documents": sum(x["documents"] for x in identities.values()),
                          **{f"{split}_documents": identities[split]["documents"] for split in SPLITS},
                          "test_split_removed_documents": removed,
                          "test_split_sha256": holdout["sha256"]})
            # The pre-split mix remains auditable; report final physical source
            # counts separately instead of pretending removals never happened.
            stats["final_split_source_counts"] = {split: identities[split]["sources"] for split in SPLITS}
            from curator.state import atomic_write_json
            atomic_write_json(stats_path, stats)
        write_manifest(stage, stage=manifest["stage"], contract=manifest["contract"],
                       input_signature=manifest["input_signature"], output_pattern="*.json*",
                       metadata={**manifest.get("metadata", {}), "test_split_sha256": holdout["sha256"]})
        directory.rename(backup)
        try:
            stage.rename(directory)
        except BaseException:
            backup.rename(directory)
            raise
        promoted = True
        shutil.rmtree(backup)
        log.info("Test established: %s", {s: x["documents"] for s, x in identities.items()})
        return holdout
    finally:
        if not promoted and stage.exists() and directory.exists():
            shutil.rmtree(stage)
