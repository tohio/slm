"""HF routing: dataset stages to Dataset repos; generic objects to Buckets.

Uses the active role environment. Bucket access uses HF's documented S3
compatibility gateway, not Hub-1.x-only APIs or another Python environment.
"""
from __future__ import annotations

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from curator.artifacts import safe_path


def bucket_transport(bucket: str, workers: int):
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    if workers < 1:
        raise ValueError("workers must be positive")
    parts = bucket.split("/")
    if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", p) for p in parts):
        raise ValueError("HF_ARTIFACT_BUCKET must be namespace/bucket")
    namespace, name = parts
    key = os.environ.get("HF_S3_ACCESS_KEY_ID")
    secret = os.environ.get("HF_S3_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise RuntimeError("Set HF_S3_ACCESS_KEY_ID and HF_S3_SECRET_ACCESS_KEY (HF-generated S3 credentials); AWS credentials are not reused")
    concurrency = min(max(4, workers), 32)
    client = boto3.client("s3", endpoint_url=f"https://s3.hf.co/{namespace}",
        aws_access_key_id=key, aws_secret_access_key=secret,
        config=Config(region_name="us-east-1", s3={"addressing_style": "path"},
            request_checksum_calculation="when_required", response_checksum_validation="when_required",
            max_pool_connections=max(16, workers * concurrency),
            retries={"max_attempts": 5, "mode": "adaptive"}, connect_timeout=10, read_timeout=120))
    transfer = TransferConfig(multipart_threshold=2 * 1024**3,
        multipart_chunksize=2 * 1024**3, max_concurrency=concurrency)
    return name, {"client": client, "transfer_config": transfer}


def _dataset():
    from huggingface_hub import HfApi
    repo = os.environ.get("HF_DATASET_REPO", "")
    if len(repo.split("/")) != 2 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", p) for p in repo.split("/")):
        raise ValueError("HF_DATASET_REPO must be namespace/dataset-name")
    return HfApi(token=os.environ.get("HF_TOKEN")), repo


def dataset_revision() -> str:
    api, repo = _dataset()
    # Pin a single commit across all stages in one restore, even if main moves.
    return api.repo_info(repo, repo_type="dataset").sha


def _card(api, repo: str, revision: str, remote: str, size: str, run_id: str, stage: str, splits: list[str]):
    import yaml
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        path = hf_hub_download(repo, "README.md", repo_type="dataset", revision=revision, token=api.token)
        text = Path(path).read_text(encoding="utf-8")
    except EntryNotFoundError:
        text = "# Pretraining data\n\nEnglish-language corpus artifacts, organized by size, run, and stage.\n"
    metadata, body = {}, text
    if text.startswith("---\n"):
        front, delimiter, body = text[4:].partition("\n---\n")
        if not delimiter:
            raise RuntimeError("Existing Dataset README has malformed YAML; refusing to overwrite it")
        metadata = yaml.safe_load(front) or {}
        if not isinstance(metadata, dict):
            raise RuntimeError("Existing Dataset README metadata must be a mapping")
    configs = metadata.setdefault("configs", [])
    if not isinstance(configs, list):
        raise RuntimeError("Existing Dataset configs metadata must be a list")
    name = f"{size}-{run_id}-{stage}"
    entry = {"config_name": name, "data_files": [
        {"split": "validation" if split == "val" else split, "path": f"{remote}/{split}.jsonl"}
        for split in splits]}
    configs[:] = [cfg for cfg in configs if cfg.get("config_name") != name] + [entry]
    return ("---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n" + body).encode("utf-8")


def upload_dataset_stage(src: Path, remote: str, *, size: str, run_id: str,
                         stage: str, overwrite: bool, pattern: str = "**/*") -> dict[str, int]:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    api, repo = _dataset()
    # Creating a Dataset repo is allowed by an explicit HF stage push. Never
    # create a Model repo here. Existing visibility is left untouched.
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    revision = api.repo_info(repo, repo_type="dataset").sha
    remote = safe_path(remote)
    existing = {name for name in api.list_repo_files(repo, repo_type="dataset", revision=revision)
                if name.startswith(remote + "/")}
    files = sorted(p for p in src.glob(pattern) if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(src.resolve()) for p in files):
        raise RuntimeError("Dataset uploads cannot follow symlinks outside the stage")
    operations, local_keys = [], set()
    skipped = 0
    for path in files:
        key = f"{remote}/{safe_path(path.relative_to(src).as_posix())}"
        local_keys.add(key)
        if not overwrite and key in existing:
            skipped += 1
            continue
        operations.append(CommitOperationAdd(path_in_repo=key, path_or_fileobj=str(path)))
    uploaded = len(operations)
    if overwrite:
        operations.extend(CommitOperationDelete(path_in_repo=key) for key in sorted(existing - local_keys))
    # Name only real dataset split files in the Hub card. Manifests and the
    # test membership index travel with the stage but are not dataset rows.
    available = local_keys | (existing if not overwrite else set())
    splits = [split for split in ("train", "val", "test") if f"{remote}/{split}.jsonl" in available]
    if splits:
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=
            _card(api, repo, revision, remote, size, run_id, stage, splits)))
    if operations:
        api.create_commit(repo_id=repo, repo_type="dataset", parent_commit=revision,
            operations=operations, commit_message=f"Upload {size}/{run_id}/{stage}")
    return {"uploaded": uploaded, "skipped": skipped, "failed": 0}


def download_dataset_stage(remote: str, dst: Path, *, revision: str,
                           workers: int, overwrite: bool) -> dict[str, int]:
    from huggingface_hub import hf_hub_download
    if workers < 1:
        raise ValueError("workers must be positive")
    api, repo = _dataset()
    prefix = safe_path(remote) + "/"
    # list_repo_tree provides sizes without a HEAD per object.
    entries = [entry for entry in api.list_repo_tree(repo_id=repo, path_in_repo=remote,
        repo_type="dataset", revision=revision, recursive=True) if hasattr(entry, "size")]
    if not entries:
        raise FileNotFoundError(f"No Dataset files at {repo}/{remote}@{revision}")
    def download(entry):
        if not entry.path.startswith(prefix):
            raise RuntimeError("Dataset API returned a path outside the requested stage")
        name = safe_path(entry.path[len(prefix):])
        target = dst / name
        if not target.resolve().is_relative_to(dst.resolve()):
            raise RuntimeError("Dataset download escapes destination")
        if not overwrite and target.is_file() and target.stat().st_size == entry.size:
            return "skipped"
        cached = hf_hub_download(repo_id=repo, filename=entry.path, repo_type="dataset",
            revision=revision, token=api.token)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        return "downloaded"
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(download, entries):
            counts[result] += 1
    return counts
