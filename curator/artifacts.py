"""Retention and restore integrity for the existing six artifact stages.

S3 file transfer remains in upload_s3.py. HF is an alternative transport, not
an additional upload. Descriptor publication is the bundle completion marker.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from config.holdout import CONTRACT_NAME, SPLITS, load_contract, sha256_file, verify_jsonl_contract
from curator.state import atomic_write_json, file_snapshot, stable_digest, write_manifest

BUNDLE_NAME = "bundle_manifest.json"
RESTORE_MARKER = "_RESTORE_PENDING.json"
PROFILES = {
    "full": ("raw", "curated", "validated", "tokenized", "tokenizer", "metadata"),
    "training-ready": ("validated", "tokenized", "tokenizer", "metadata"),
}


def stages_for(api, stages, profile):
    if profile not in PROFILES:
        raise ValueError(f"Unsupported retention profile: {profile}")
    chosen = api._normalize_stages(stages) if stages else list(PROFILES[profile])
    if profile == "training-ready" and set(chosen) - set(PROFILES[profile]):
        raise ValueError("training-ready does not archive raw or curated stages; use full")
    return list(dict.fromkeys([stage for stage in chosen if stage != "metadata"] + ["metadata"]))


def safe_path(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise RuntimeError(f"Unsafe artifact path: {name!r}")
    return name


def _hf(operation, root, relative, bucket, prefix):
    executable = os.environ.get("ARTIFACT_HF_PYTHON", sys.executable)
    job = {"operation": operation, "root": str(root), "files": relative,
           "bucket": bucket, "prefix": prefix}
    result = subprocess.run([executable, "-m", "curator.artifact_hf"],
        input=json.dumps(job), text=True, capture_output=True, check=False,
        cwd=Path(__file__).resolve().parents[1])
    if result.returncode:
        raise RuntimeError(f"HF bucket transfer failed: {result.stderr[-4000:]}")
    return json.loads(result.stdout)


def _get_descriptor(api, size, run_id, bucket, prefix, workers, backend, work):
    relative = f"{size}/{run_id}/metadata/{BUNDLE_NAME}"
    dest = work / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if backend == "hf":
        response = _hf("read", work, [relative], bucket, prefix)
        if response.get("missing"):
            return None
    elif backend == "s3":
        from botocore.exceptions import ClientError
        try:
            api.get_s3_client(workers).download_file(bucket, api.build_key(prefix, relative), str(dest))
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
    else:
        raise ValueError(f"Unsupported artifact backend: {backend}")
    descriptor = json.loads(dest.read_text())
    contract = descriptor.get("contract", {})
    if descriptor.get("sha256") != stable_digest(contract) or contract.get("version") != 1:
        raise RuntimeError("Invalid artifact bundle checksum/version")
    if contract.get("size") != size or contract.get("run_id") != run_id:
        raise RuntimeError("Artifact bundle SIZE/RUN_ID mismatch")
    for stage, files in contract.get("files", {}).items():
        if stage not in api.ALL_ARTIFACT_STAGES or not isinstance(files, dict):
            raise RuntimeError("Invalid artifact bundle stages")
        for name, identity in files.items():
            safe_path(name)
            if not isinstance(identity, dict) or len(identity.get("sha256", "")) != 64 or identity.get("bytes", -1) < 0:
                raise RuntimeError("Invalid artifact file identity")
    return descriptor


def _copy_selected(src, dst, stage, profile):
    if stage == "validated" and profile == "full" and not (src / "train.jsonl").is_file():
        raise RuntimeError("Full validated retention needs train.jsonl; the available copy is training-ready only")
    for path in sorted(src.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Symlink cannot enter an artifact bundle: {path}")
        if not path.is_file() or (stage == "metadata" and path.name == BUNDLE_NAME):
            continue
        relative = path.relative_to(src)
        if profile == "training-ready" and stage == "validated" and relative == Path("train.jsonl"):
            continue
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Hardlinks do not mutate input; every generated manifest uses atomic replace.
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    if profile == "training-ready" and stage == "validated" and (src / "train.jsonl").is_file():
        original = json.loads((src / "_SUCCESS.json").read_text())
        atomic_write_json(dst / "_RETENTION.json", {
            "profile": profile, "omitted": ["train.jsonl"], "source_manifest": original})
        write_manifest(dst, stage="validated", contract={"retention_profile": profile,
            "source_manifest_sha256": stable_digest(original),
            "frozen_split_sha256": load_contract(dst, stage="validated")["sha256"]},
            input_signature=original["input_signature"], output_pattern="*.json*")


def _identity(root):
    identities = {}
    for stage in ("curated", "validated", "tokenized"):
        if (root / stage / CONTRACT_NAME).is_file():
            identities[stage] = load_contract(root / stage)["sha256"]
    for stage in ("validated", "tokenized"):
        if (root / stage / CONTRACT_NAME).is_file():
            contract = load_contract(root / stage)["contract"]
            if contract.get("curated_contract_sha256"):
                expected = contract["curated_contract_sha256"]
                if identities.get("curated", expected) != expected:
                    raise RuntimeError("Curated and validated/tokenized provenance disagree")
                identities["curated"] = expected
    for split in SPLITS:
        path = root / "tokenized" / f"{split}.json"
        if path.is_file():
            row = json.loads(path.read_text())
            identities[f"{split}_binary"] = row.get("binary_sha256")
            identities["tokenizer"] = row.get("tokenizer_sha256")
            identities["tokenizer_file"] = row.get("tokenizer_file_sha256")
    return identities


def verify_restored(api, root, size, run_id, stages, *, inventory=None):
    for stage in stages:
        api._assert_artifact_stage_complete(size, run_id, stage, root / stage)
    if "curated" in stages and (root / "curated" / CONTRACT_NAME).is_file():
        verify_jsonl_contract(root / "curated", stage="curated")
    if "validated" in stages:
        verify_jsonl_contract(root / "validated", stage="validated",
                              include_train=(root / "validated" / "train.jsonl").is_file())
    if "tokenized" in stages:
        frozen = load_contract(root / "tokenized", stage="validated")
        reference = None
        for split in SPLITS:
            meta = json.loads((root / "tokenized" / f"{split}.json").read_text())
            expected = frozen["contract"]["splits"][split]
            if (meta.get("frozen_split_sha256") != frozen["sha256"]
                    or meta.get("input_sha256") != expected["sha256"]
                    or meta.get("n_docs") != expected["documents"]
                    or meta.get("dtype") != "uint16"
                    or (root / "tokenized" / f"{split}.bin").stat().st_size != meta.get("n_tokens", -1) * 2):
                raise RuntimeError(f"Incomplete frozen tokenized {split} artifact contract")
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
        if "curated" in stages and load_contract(root / "curated")["sha256"] != frozen["contract"].get("curated_contract_sha256"):
            raise RuntimeError("Curated provenance does not match the validated/tokenized contract")
        if "validated" in stages and load_contract(root / "validated")["sha256"] != frozen["sha256"]:
            raise RuntimeError("Validated and tokenized holdout contracts disagree")
    metadata = root / "metadata" / "pipeline_manifest.json"
    if metadata.is_file():
        pipeline = json.loads(metadata.read_text())
        for name, expected in pipeline.get("file_sha256", {}).items():
            safe_path(name)
            stage = PurePosixPath(name).parts[0]
            # Retention projects the validated completion manifest; its original
            # full-data manifest remains in _RETENTION.json, not masquerading as complete.
            projected = name == "validated/_SUCCESS.json" and (root / "validated" / "_RETENTION.json").exists()
            if stage in stages and not projected:
                path = root / name
                if not path.is_file() or sha256_file(path) != expected:
                    raise RuntimeError(f"Pipeline metadata fingerprint mismatch: {name}")


def _results_into(metadata, model_size):
    if model_size not in {"smoke", "mini", "125m", "350m", "1b"}:
        raise ValueError("--include-results requires a valid --model-size")
    source = Path(os.environ.get("RESULTS_DIR", "results")) / "runs" / model_size
    selected = []
    for stage in ("pretrain", "sft_instruct", "sft_code", "sft_code_completion", "dpo_chat"):
        if (source / stage / "final").is_dir():
            selected.append(source / stage / "final")
        for name in ("final_pretraining_eval.json", "pretraining-final.json", "train_results.json", "trainer_state.json"):
            if (source / stage / name).is_file():
                selected.append(source / stage / name)
    if (source / "eval").is_dir():
        selected.append(source / "eval")
    if not selected:
        raise FileNotFoundError(f"No final model/evaluation artifacts found: {source}")
    for path in selected:
        target = metadata / "model_results" / model_size / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        if path.is_dir():
            _copy_selected(path, target, "results", "full")
        else:
            shutil.copy2(path, target)


def upload(api, size, run_id, stages, bucket, prefix, workers, backend, profile, include_results, model_size):
    api.validate_run_id(size, run_id)
    selected = stages_for(api, stages, profile)
    source = api.DATA_DIR / "runs" / size
    if (source / RESTORE_MARKER).exists():
        raise RuntimeError("An interrupted restore must be recovered before upload")
    api._prepare_metadata(size, run_id)
    totals = {"uploaded": 0, "skipped": 0, "failed": 0}
    with tempfile.TemporaryDirectory(prefix=".slm-artifacts-", dir=source.parent) as temporary:
        work = Path(temporary)
        previous = _get_descriptor(api, size, run_id, bucket, prefix, workers, backend, work / "remote")
        identity = _identity(source)
        if previous:
            for key, expected in previous["contract"].get("dataset_identity", {}).items():
                if expected is not None and identity.get(key) != expected:
                    raise RuntimeError(f"Frozen artifact identity changed ({key}); use a new RUN_ID, never overwrite test")
        staging = work / "upload"
        inventory = {}
        for stage in selected:
            src = source / stage
            api._assert_artifact_stage_complete(size, run_id, stage, src)
            dst = staging / stage
            _copy_selected(src, dst, stage, profile)
            if stage == "metadata" and include_results:
                _results_into(dst, model_size)
            inventory[stage] = {str(path.relative_to(dst)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                                for path in sorted(dst.rglob("*")) if path.is_file()}
        verify_restored(api, staging, size, run_id, selected, inventory=inventory)
        descriptor_contract = {"version": 1, "size": size, "run_id": run_id,
            "retention_profile": profile, "dataset_identity": identity, "files": inventory}
        # Preserve inventories for previously uploaded unselected stages. Retention
        # changes select what to transfer; they never silently delete remote data.
        if previous:
            descriptor_contract["files"] = {**previous["contract"]["files"], **inventory}
        descriptor = {"contract": descriptor_contract, "sha256": stable_digest(descriptor_contract)}
        for stage in selected:
            previous_files = previous["contract"]["files"].get(stage) if previous else None
            if previous_files == inventory[stage]:
                totals["skipped"] += len(inventory[stage])
                continue
            if backend == "s3":
                counts = api.upload_directory(staging / stage, f"{size}/{run_id}/{stage}", bucket, prefix,
                                               workers=workers, overwrite=True, mirror=False)
                if counts.get("failed"):
                    raise RuntimeError(f"Artifact stage upload failed: {stage}")
                totals["uploaded"] += counts.get("uploaded", 0)
            elif backend == "hf":
                _hf("upload", staging / stage, list(inventory[stage]), bucket,
                    "/".join(part for part in (prefix, size, run_id, stage) if part))
                totals["uploaded"] += len(inventory[stage])
            else:
                raise ValueError(f"Unsupported artifact backend: {backend}")
        atomic_write_json(staging / "metadata" / BUNDLE_NAME, descriptor)
        # Publish last. A failed transfer never advertises a completed bundle.
        if backend == "s3":
            api.get_s3_client(workers).upload_file(str(staging / "metadata" / BUNDLE_NAME), bucket,
                api.build_key(prefix, f"{size}/{run_id}/metadata/{BUNDLE_NAME}"), Config=api._transfer_config(workers))
        else:
            _hf("upload", staging / "metadata", [BUNDLE_NAME], bucket,
                "/".join(part for part in (prefix, size, run_id, "metadata") if part))
    api._write_run_id_record(size, run_id)
    return totals


def download(api, size, run_id, stages, bucket, prefix, workers, backend, profile, overwrite, restore_results, model_size):
    from concurrent.futures import ThreadPoolExecutor
    api.validate_run_id(size, run_id)
    selected = stages_for(api, stages, profile)
    root = api.DATA_DIR / "runs" / size
    root.mkdir(parents=True, exist_ok=True)
    marker = root / RESTORE_MARKER
    if marker.exists():
        raise RuntimeError(f"Interrupted restore: recover paths recorded in {marker} before retrying")
    old_id = api._read_run_id_record(root / "RUN_ID")
    if old_id and old_id["run_id"] != run_id and not overwrite:
        raise RuntimeError("Restore would replace another dataset RUN_ID; choose a new DATA_DIR or pass --overwrite")
    with tempfile.TemporaryDirectory(prefix=f".{size}-restore-", dir=root.parent) as temporary:
        work = Path(temporary)
        descriptor = _get_descriptor(api, size, run_id, bucket, prefix, workers, backend, work / "remote")
        if descriptor is None:
            raise RuntimeError("Missing completed bundle manifest. Re-upload the existing frozen artifacts; do not regenerate test")
        inventory = descriptor["contract"]["files"]
        if set(selected) - set(inventory):
            raise RuntimeError(f"Bundle is missing requested stages: {sorted(set(selected) - set(inventory))}")
        staging = work / "staging"
        for stage in selected:
            destination = staging / stage
            destination.mkdir(parents=True, exist_ok=True)
            names = list(inventory[stage])
            if backend == "hf":
                _hf("download", destination, names, bucket,
                    "/".join(part for part in (prefix, size, run_id, stage) if part))
            else:
                client = api.get_s3_client(workers)
                config = api._transfer_config(workers)
                def get(name):
                    path = destination / safe_path(name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(bucket, api.build_key(prefix, f"{size}/{run_id}/{stage}/{name}"), str(path), Config=config)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    list(executor.map(get, names))
            for name, expected in inventory[stage].items():
                path = destination / name
                if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                    raise RuntimeError(f"Downloaded artifact checksum mismatch: {stage}/{name}")
        atomic_write_json(staging / "metadata" / BUNDLE_NAME, descriptor)
        verify_restored(api, staging, size, run_id, selected, inventory=inventory)
        # Stage swaps are rolled back on ordinary failures. A power interruption
        # leaves an explicit marker which every dataset resolver rejects.
        backup = work / "backup"
        backup.mkdir()
        atomic_write_json(marker, {"run_id": run_id, "stages": selected, "backup": str(backup), "staging": str(staging)})
        promoted, saved = [], []
        try:
            for stage in selected:
                if (root / stage).exists():
                    (root / stage).rename(backup / stage)
                    saved.append(stage)
                (staging / stage).rename(root / stage)
                promoted.append(stage)
            api._write_run_id_record(size, run_id)
            marker.unlink()
        except BaseException:
            for stage in reversed(promoted):
                shutil.rmtree(root / stage)
            for stage in reversed(saved):
                (backup / stage).rename(root / stage)
            marker.unlink(missing_ok=True)
            raise
        if restore_results:
            source = root / "metadata" / "model_results" / str(model_size)
            if model_size not in {"smoke", "mini", "125m", "350m", "1b"} or not source.is_dir():
                raise RuntimeError("Requested model-size final artifacts are not in the restored metadata stage")
            target = Path(os.environ.get("RESULTS_DIR", "results")) / "runs" / model_size
            if target.exists():
                raise RuntimeError("Refusing to overwrite model results; restore into a new RESULTS_DIR")
            shutil.copytree(source, target)
    return {"downloaded": sum(len(inventory[stage]) for stage in selected), "skipped": 0, "failed": 0}
