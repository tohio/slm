"""Small, portable training-history records carried by final checkpoints.

Only audits, data manifests, measured token metadata and checkpoint fingerprints
are bundled. No ancestor weights or corpus files are duplicated. A legacy parent
must be explicitly available and match the identity recorded by its child.
"""
from __future__ import annotations

import json
from pathlib import Path

from config.checkpoints import checkpoint_identity
from curator.state import atomic_write_json, stable_digest

PROVENANCE_FILENAME = "training_provenance.json"
_STAGE_FILES = {
    "pretrain": ("pretrain_run_audit.json", None),
    "sft_instruct": ("sft_run_audit.json", "sft_data_manifest.json"),
    "sft_code": ("sft_run_audit.json", "sft_data_manifest.json"),
    "dpo_chat": ("dpo_run_audit.json", "dpo_data_manifest.json"),
}


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing training provenance: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid provenance object: {path}")
    return value


def _envelope(stages: list[dict]) -> dict:
    payload = {"schema_version": 1, "stages": stages}
    return {**payload, "sha256": stable_digest(payload)}


def _without_provenance(identity: dict) -> dict:
    return {k: v for k, v in identity.items() if k != "provenance_sha256"}


def _entry(checkpoint: Path) -> dict:
    audit_paths = [checkpoint / name for name in
                   ("pretrain_run_audit.json", "sft_run_audit.json", "dpo_run_audit.json")
                   if (checkpoint / name).is_file()]
    if len(audit_paths) != 1:
        raise RuntimeError(f"Expected exactly one stage audit in {checkpoint}")
    audit = _read(audit_paths[0])
    contract = audit.get("contract")
    if not isinstance(contract, dict) or audit.get("contract_sha256") != stable_digest(contract):
        raise RuntimeError(f"Invalid run audit: {audit_paths[0]}")
    stage = contract.get("stage", "pretrain" if audit_paths[0].name == "pretrain_run_audit.json" else None)
    if stage not in _STAGE_FILES or _STAGE_FILES[stage][0] != audit_paths[0].name:
        raise RuntimeError(f"Unsupported provenance stage in {audit_paths[0]}: {stage}")
    entry = {"stage": stage, "audit": audit,
             "checkpoint_identity": checkpoint_identity(checkpoint, include_provenance=False)}
    manifest_name = _STAGE_FILES[stage][1]
    if manifest_name:
        entry["data_manifest"] = _read(checkpoint / manifest_name)
        if stable_digest(entry["data_manifest"]) != contract.get("data_manifest_sha256"):
            raise RuntimeError(f"Stage data manifest differs from the training audit: {checkpoint}")
    return entry


def _validate_stages(stages: list[dict]) -> None:
    if not isinstance(stages, list) or not stages or stages[0].get("stage") != "pretrain":
        raise RuntimeError("Training provenance must begin with pretraining")
    allowed = {"pretrain": {"sft_instruct"}, "sft_instruct": {"sft_code", "dpo_chat"}}
    for index, entry in enumerate(stages):
        audit = entry["audit"]
        contract = audit["contract"]
        if audit.get("contract_sha256") != stable_digest(contract):
            raise RuntimeError("Bundled audit checksum mismatch")
        stage = entry["stage"]
        if stage != contract.get("stage", "pretrain"):
            raise RuntimeError("Bundled stage does not match its audit")
        if stage != "pretrain":
            if stable_digest(entry.get("data_manifest")) != contract.get("data_manifest_sha256"):
                raise RuntimeError("Bundled manifest does not match its stage audit")
        if index:
            previous = stages[index - 1]
            if stage not in allowed.get(previous["stage"], set()):
                raise RuntimeError("Invalid training-provenance lineage")
            expected = contract.get("base_checkpoint", {})
            if _without_provenance(expected) != previous["checkpoint_identity"]:
                raise RuntimeError("Ancestor fingerprint does not match the child's recorded parent")
            if (expected.get("provenance_sha256") is not None
                    and expected["provenance_sha256"] != _envelope(stages[:index])["sha256"]):
                raise RuntimeError("Ancestor provenance differs from the recorded parent bundle")
        mixture = entry.get("token_mixture")
        if mixture is not None:
            identity = contract.get("tokenized_data", {})
            if stable_digest(mixture) != identity.get("realized_mixture", {}).get("report_sha256"):
                raise RuntimeError("Measured corpus mixture does not match the pretraining audit")
            for split, recorded in identity.get("splits", {}).items():
                if mixture.get("split_metadata_sha256", {}).get(split) != recorded["metadata_sha256"]:
                    raise RuntimeError("Measured corpus mixture is from another tokenized dataset")


def collect_training_provenance(checkpoint: Path, *, parents=(), _seen=()) -> dict:
    """Read a self-contained bundle, or verify explicitly available legacy parents.

    Historical paths are a legacy fallback only. Explicit parents are matched by
    full checkpoint identity, never by their filename or model-size label.
    """
    checkpoint = Path(checkpoint)
    resolved = checkpoint.resolve()
    if resolved in _seen:
        raise RuntimeError("Cycle in checkpoint provenance")
    current = _entry(checkpoint)
    bundled = checkpoint / PROVENANCE_FILENAME
    if bundled.exists():
        payload = _read(bundled)
        stages = payload.get("stages")
        if payload != _envelope(stages):
            raise RuntimeError(f"Provenance checksum/schema mismatch: {bundled}")
        _validate_stages(stages)
        last = stages[-1]
        for key in ("stage", "audit", "checkpoint_identity", "data_manifest"):
            if last.get(key) != current.get(key):
                raise RuntimeError(f"Bundled {key} does not describe this checkpoint: {checkpoint}")
        return payload
    return _extend_history(current, checkpoint, parents=parents, seen=(*_seen, resolved))


def _extend_history(current: dict, checkpoint: Path, *, parents, seen) -> dict:
    stages = []
    if current["stage"] != "pretrain":
        expected = current["audit"]["contract"]["base_checkpoint"]
        candidates = [Path(p) for p in parents]
        historical = current["audit"].get("base_model")
        if historical:
            import os
            candidates.append(Path(os.path.expandvars(historical)).expanduser())
        match = None
        for candidate in dict.fromkeys(candidates):
            if not candidate.is_dir():
                continue
            try:
                actual = checkpoint_identity(candidate)
            except (FileNotFoundError, ValueError, RuntimeError):
                continue
            if (_without_provenance(actual) == _without_provenance(expected)
                    and ("provenance_sha256" not in expected
                         or actual.get("provenance_sha256") == expected["provenance_sha256"])):
                match = candidate
                break
        if match is None:
            raise RuntimeError(
                f"{checkpoint} has no portable ancestor bundle and its verified parent is unavailable. "
                "Supply --provenance-parent PATH for the original parent (repeat for older ancestors). "
                "Do not substitute a same-size model or edit audit paths/hashes."
            )
        stages = collect_training_provenance(match, parents=parents, _seen=seen)["stages"]
    stages = [*stages, current]
    _validate_stages(stages)
    return _envelope(stages)


def save_training_provenance(checkpoint: Path, *, parent: Path | None = None,
                             token_mixture: dict | None = None,
                             training_metrics: dict | None = None) -> None:
    """Capture the saved final checkpoint; called only by the rank-zero save path."""
    checkpoint = Path(checkpoint)
    current = _entry(checkpoint)
    if token_mixture is not None:
        if current["stage"] != "pretrain":
            raise RuntimeError("Only pretraining owns tokenized corpus provenance")
        current["token_mixture"] = token_mixture
    if training_metrics is not None:
        current["training_metrics"] = training_metrics
    payload = _extend_history(current, checkpoint, parents=[parent] if parent else [],
                              seen=(checkpoint.resolve(),))
    atomic_write_json(checkpoint / PROVENANCE_FILENAME, payload)
