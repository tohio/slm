"""Integrity checks for the existing artifact stages; no retention policy.

Selection, skip/overwrite, and generic object transfers remain in upload_s3.py.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath
from config.holdout import CONTRACT_NAME, SPLITS, load_contract, sha256_file, verify_jsonl_contract


def safe_path(name: str) -> str:
    path = PurePosixPath(name)
    if (name in {"", ".", ".."} or path.is_absolute() or ".." in path.parts or "\\" in name
            or "\0" in name or str(path) != name):
        raise RuntimeError(f"Unsafe artifact path: {name!r}")
    return name


def verify_restored(api, root, size, run_id, stages, *, inventory=None):
    for stage in stages:
        api._assert_artifact_stage_complete(size, run_id, stage, root / stage)
    if "curated" in stages and (root / "curated" / CONTRACT_NAME).is_file():
        verify_jsonl_contract(root / "curated", stage="curated")
    if "validated" in stages:
        verify_jsonl_contract(root / "validated", stage="validated",
                              include_train=True)
    if "tokenized" in stages:
        holdout = load_contract(root / "tokenized", stage="validated")
        reference = None
        for split in SPLITS:
            meta = json.loads((root / "tokenized" / f"{split}.json").read_text())
            expected = holdout["contract"]["splits"][split]
            if (meta.get("test_split_sha256") != holdout["sha256"]
                    or meta.get("input_sha256") != expected["sha256"]
                    or meta.get("n_docs") != expected["documents"]
                    or meta.get("dtype") != "uint16"
                    or (root / "tokenized" / f"{split}.bin").stat().st_size != meta.get("n_tokens", -1) * 2):
                raise RuntimeError(f"Incomplete holdout tokenized {split} artifact contract")
            fingerprint = (meta.get("tokenizer_sha256"), meta.get("bos_id"), meta.get("eos_id"), meta.get("vocab_size"),
                           meta.get("tokenizer_file_sha256"))
            if reference is not None and reference != fingerprint:
                raise RuntimeError("Mixed tokenized split tokenizer identities")
            reference = fingerprint
            actual_sha = (inventory.get("tokenized", {}).get(f"{split}.bin", {}).get("sha256")
                          if inventory is not None else sha256_file(root / "tokenized" / f"{split}.bin"))
            if not meta.get("binary_sha256") or actual_sha != meta["binary_sha256"]:
                raise RuntimeError(f"Tokenized {split} checksum disagrees with its metadata")
        if "tokenizer" in stages:
            tokenizer_sha = sha256_file(root / "tokenizer" / "slm_tokenizer.json")
            if not reference[-1] or tokenizer_sha != reference[-1]:
                raise RuntimeError("Restored tokenizer file does not match tokenized split fingerprints")
        if "curated" in stages and load_contract(root / "curated")["sha256"] != holdout["contract"].get("curated_contract_sha256"):
            raise RuntimeError("Curated provenance does not match the validated/tokenized contract")
        if "validated" in stages and load_contract(root / "validated")["sha256"] != holdout["sha256"]:
            raise RuntimeError("Validated and tokenized holdout contracts disagree")
    metadata = root / "metadata" / "pipeline_manifest.json"
    if metadata.is_file():
        pipeline = json.loads(metadata.read_text())
        for name, expected in pipeline.get("file_sha256", {}).items():
            safe_path(name)
            stage = PurePosixPath(name).parts[0]
            if stage in stages:
                path = root / name
                if not path.is_file() or sha256_file(path) != expected:
                    raise RuntimeError(f"Pipeline metadata fingerprint mismatch: {name}")
