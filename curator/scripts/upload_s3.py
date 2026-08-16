"""
curator/scripts/upload_s3.py
-----------------------------
S3 upload and download utilities for the SLM data pipeline.

Changes from prior version:
    - Uses boto3's default credential chain (env vars → ~/.aws/credentials
      → EC2 IAM role). Previously required AWS_ACCESS_KEY_ID/SECRET to be
      set explicitly, which broke on instances with attached IAM roles.
    - Adds adaptive retry config — large overnight uploads no longer die
      on transient S3 throttling.
    - Adds per-file transfer progress callback for multi-GB uploads.
    - Uses ListObjectsV2 to build an existing-keys set once instead of
      HEAD-ing every file — much cheaper when most files already exist.

Env vars:
    S3_BUCKET           — S3 bucket name (required)
    S3_PREFIX           — key prefix (default: slm/data)
    RUN_ID              — optional explicit artifact run id
    AWS_DEFAULT_REGION  — (default: us-east-1)

    AWS credentials: standard boto3 chain — env vars, profile, IAM role.
"""

import argparse
import json
import logging
import os
import re
import secrets
import sys
import threading
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Direct execution (``python curator/scripts/upload_s3.py``) places only the
# scripts directory on sys.path. Add the repository root before importing the
# shared config and curator packages, matching the other pipeline entry points.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.data_mix import ALL_SOURCES
from curator.state import file_snapshot, manifest_outputs_match

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
ARTIFACT_STAGES = ("raw", "tokenized", "tokenizer", "metadata")
OPTIONAL_ARTIFACT_STAGES = ("curated", "validated")
ALL_ARTIFACT_STAGES = ARTIFACT_STAGES + OPTIONAL_ARTIFACT_STAGES
RUN_ID_FILENAME = "RUN_ID"

# S3 transfer configuration is built per command so large single-file artifacts
# can use multipart concurrency. File-level concurrency is still controlled by
# --workers; each large-file transfer uses a bounded inner pool so files such as
# curated/train.jsonl and validated/train.jsonl do not upload as a single slow
# stream.
def _transfer_config(workers: int) -> TransferConfig:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got: {workers}")

    multipart_workers = min(max(4, workers), 32)
    return TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=multipart_workers,
        use_threads=True,
    )


def get_s3_client(workers: int = 16):
    """
    Build an S3 client whose HTTP connection pool matches transfer concurrency.

    Explicitly passing aws_access_key_id/secret breaks IAM role auth on EC2.
    Boto3 will find creds automatically from: env vars → ~/.aws → IAM role.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got: {workers}")

    pool_connections = max(16, workers * 4)
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=120,
            max_pool_connections=pool_connections,
        ),
    )


def get_bucket_and_prefix() -> tuple[str, str]:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError(
            "S3_BUCKET env var is not set. Configure it in .env "
            "or export it before running."
        )
    prefix = os.environ.get("S3_PREFIX", "slm/data").rstrip("/")
    return bucket, prefix


def build_key(prefix: str, relative_path: str) -> str:
    """Build a full S3 key from prefix and a relative path."""
    return f"{prefix}/{relative_path.lstrip('/')}"


def _today_iso() -> str:
    return date.today().isoformat()


def _today_compact() -> str:
    return date.today().strftime("%Y%m%d")


def _run_id_path(size: str) -> Path:
    return DATA_DIR / "runs" / size / RUN_ID_FILENAME


def _validate_size(size: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", size):
        raise ValueError(f"Invalid size: {size}")


def validate_run_id(size: str, run_id: str) -> None:
    """Validate artifact RUN_ID.

    RUN_ID is intentionally opaque after validation, but generated IDs use:
      {SIZE}-{YYYYMMDD}-{random_hex}
    """
    _validate_size(size)
    if "/" in run_id or "\\" in run_id or run_id in {"", ".", ".."}:
        raise ValueError(f"Invalid RUN_ID: {run_id}")

    pattern = rf"{re.escape(size)}-\d{{8}}-[A-Za-z0-9._-]+"
    if not re.fullmatch(pattern, run_id):
        raise ValueError(
            f"RUN_ID must match {size}-YYYYMMDD-<id>, got: {run_id}"
        )


def _read_run_id_record(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None

    raw = path.read_text().strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {
                "run_id": str(obj.get("run_id", "")),
                "date": str(obj.get("date", "")),
            }
    except json.JSONDecodeError:
        return {"run_id": raw, "date": ""}

    return None


def _write_run_id_record(size: str, run_id: str) -> None:
    path = _run_id_path(size)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "date": _today_iso(),
        "size": size,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _new_run_id(size: str) -> str:
    return f"{size}-{_today_compact()}-{secrets.token_hex(3)}"


def resolve_upload_run_id(size: str, provided_run_id: str | None) -> str:
    """Resolve the run id for artifact upload.

    Rules:
      1. Explicit RUN_ID wins.
      2. Today's local RUN_ID file wins.
      3. Missing/stale local RUN_ID file is replaced.
    """
    _validate_size(size)

    if provided_run_id:
        validate_run_id(size, provided_run_id)
        _write_run_id_record(size, provided_run_id)
        return provided_run_id

    path = _run_id_path(size)
    today = _today_iso()
    record = _read_run_id_record(path)
    if record and record.get("date") == today and record.get("run_id"):
        run_id = record["run_id"]
        validate_run_id(size, run_id)
        return run_id

    run_id = _new_run_id(size)
    _write_run_id_record(size, run_id)
    return run_id


def require_run_id(size: str, run_id: str | None) -> str:
    if not run_id:
        raise ValueError("RUN_ID is required for artifact download/restore")
    validate_run_id(size, run_id)
    return run_id


def _prepare_metadata(size: str, run_id: str) -> None:
    """Materialize metadata artifacts before upload."""
    metadata_dir = DATA_DIR / "runs" / size / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    run_payload = {
        "run_id": run_id,
        "date": _today_iso(),
        "size": size,
    }
    (metadata_dir / "run_id.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n"
    )

    blend_stats = DATA_DIR / "runs" / size / "curated" / "blend_stats.json"
    if blend_stats.exists():
        (metadata_dir / "blend_stats.json").write_text(blend_stats.read_text())

    run_dir = DATA_DIR / "runs" / size
    tracked = [
        run_dir / "curated" / "_SUCCESS.json",
        run_dir / "curated" / "blend_stats.json",
        run_dir / "validated" / "_SUCCESS.json",
        run_dir / "validated" / "validation_stats.json",
        run_dir / "tokenized" / "train.json",
        run_dir / "tokenized" / "val.json",
        run_dir / "tokenizer" / "slm_tokenizer.json",
    ]
    pipeline_manifest = {
        "run_id": run_id,
        "size": size,
        "artifacts": {
            item["path"]: {
                key: value
                for key, value in item.items()
                if key != "path"
            }
            for item in file_snapshot(
                [path for path in tracked if path.exists()],
                root=run_dir,
                exclude_manifest=False,
            )
        },
    }
    (metadata_dir / "pipeline_manifest.json").write_text(
        json.dumps(pipeline_manifest, indent=2, sort_keys=True) + "\n"
    )


def _list_existing_keys(
    client, bucket: str, full_prefix: str,
) -> set[str]:
    """List all existing keys under a prefix — used to skip already-uploaded files."""
    existing: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])
    return existing


class _ProgressCallback:
    """
    Thread-safe progress callback for boto3 upload_file / download_file.

    Boto3 invokes this callback from its transfer threads with the number
    of bytes just transferred.
    """

    def __init__(self, total_bytes: int, desc: str, position: int = 0):
        self._pbar = tqdm(
            total=total_bytes,
            desc=desc,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            position=position,
        )
        self._lock = threading.Lock()

    def __call__(self, bytes_transferred: int) -> None:
        with self._lock:
            self._pbar.update(bytes_transferred)

    def close(self) -> None:
        self._pbar.close()


# ── Upload ─────────────────────────────────────────────────────────────────────

def _upload_one(
    local: Path,
    key: str,
    bucket: str,
    client,
    show_progress: bool,
    transfer_config: TransferConfig,
) -> bool:
    """Upload one file, optionally with a progress callback."""
    if show_progress:
        size = local.stat().st_size
        cb = _ProgressCallback(size, desc=local.name)
        try:
            client.upload_file(
                str(local), bucket, key, Callback=cb, Config=transfer_config
            )
        finally:
            cb.close()
    else:
        client.upload_file(str(local), bucket, key, Config=transfer_config)
    return True


def upload_directory(
    src: Path,
    dst_prefix: str,
    bucket: str,
    prefix: str,
    workers: int = 16,
    overwrite: bool = False,
    glob: str = "**/*",
    large_file_bytes: int = 100 * 1024 * 1024,
    mirror: bool = False,
) -> dict[str, int]:
    """
    Upload a local directory to S3 recursively.

    Files larger than large_file_bytes (default 100MB) get a per-file
    progress bar so multi-GB uploads don't appear frozen.

    Args:
        src: Local source directory.
        dst_prefix: Destination prefix within S3_PREFIX (e.g. "curated").
        bucket: S3 bucket name.
        prefix: S3_PREFIX from environment.
        workers: Concurrent upload threads. Default: 16.
        overwrite: If False, skip files that already exist in S3.
        glob: File pattern. Default: all files.
        large_file_bytes: Threshold above which we show per-file progress.
        mirror: After a successful upload, delete remote keys under this exact
            prefix that are not present locally. Intended for immutable
            RUN_ID/stage prefixes, not generic partial uploads.
    """
    client = get_s3_client(workers)
    transfer_config = _transfer_config(workers)
    files = [f for f in src.glob(glob) if f.is_file()]
    if not files:
        log.warning(f"No files found in {src} matching '{glob}'")
        return {"uploaded": 0, "skipped": 0, "failed": 0}

    full_prefix = f"{prefix}/{dst_prefix}".rstrip("/") + "/"
    log.info(f"Uploading {len(files)} files → s3://{bucket}/{full_prefix}")

    # Build existing-keys set once rather than HEAD-ing every file.
    existing: set[str] = set()
    if not overwrite or mirror:
        purpose = "mirror validation" if overwrite else "resume/skip"
        log.info(f"  Listing existing objects for {purpose}...")
        existing = _list_existing_keys(client, bucket, full_prefix)
        log.info(f"  {len(existing)} objects already present")

    local_keys = {
        build_key(
            f"{prefix}/{dst_prefix}",
            str(path.relative_to(src)),
        )
        for path in files
    }

    counts = {"uploaded": 0, "skipped": 0, "failed": 0}

    def _upload(f: Path) -> str:
        relative = f.relative_to(src)
        key = build_key(f"{prefix}/{dst_prefix}", str(relative))
        if not overwrite and key in existing:
            return "skipped"
        try:
            show_progress = f.stat().st_size >= large_file_bytes
            _upload_one(
                f, key, bucket, client,
                show_progress=show_progress,
                transfer_config=transfer_config,
            )
            return "uploaded"
        except Exception as e:
            log.error(f"Failed to upload {f}: {e}")
            return "failed"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_upload, f): f for f in files}
        with tqdm(total=len(files), desc="Uploading", unit="file") as pbar:
            for future in as_completed(futures):
                counts[future.result()] += 1
                pbar.update(1)
                pbar.set_postfix(counts)

    log.info(
        f"Upload complete — "
        f"uploaded: {counts['uploaded']}, "
        f"skipped: {counts['skipped']}, "
        f"failed: {counts['failed']}"
    )
    if mirror and not counts["failed"]:
        stale_keys = sorted(existing - local_keys)
        for start in range(0, len(stale_keys), 1_000):
            batch = stale_keys[start:start + 1_000]
            response = client.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [{"Key": key} for key in batch],
                    "Quiet": True,
                },
            )
            errors = response.get("Errors", [])
            if errors:
                raise RuntimeError(
                    f"Failed to remove {len(errors)} stale S3 object(s) "
                    f"under s3://{bucket}/{full_prefix}"
                )
        if stale_keys:
            log.info(
                f"Removed {len(stale_keys)} stale object(s) so the remote "
                "artifact exactly mirrors the completed local stage"
            )
    return counts


# ── Download ───────────────────────────────────────────────────────────────────

def download_prefix(
    src_prefix: str,
    dst: Path,
    bucket: str,
    prefix: str,
    workers: int = 16,
    overwrite: bool = False,
    large_file_bytes: int = 100 * 1024 * 1024,
) -> dict[str, int]:
    """Download all objects under an S3 prefix to a local directory."""
    client = get_s3_client(workers)
    transfer_config = _transfer_config(workers)
    full_prefix = f"{prefix}/{src_prefix}".rstrip("/") + "/"

    paginator = client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        objects.extend(page.get("Contents", []))

    if not objects:
        raise FileNotFoundError(
            f"No objects found at requested artifact prefix "
            f"s3://{bucket}/{full_prefix}"
        )

    log.info(f"Downloading {len(objects)} objects → {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}

    def _download(obj: dict) -> str:
        key = obj["Key"]
        size = obj.get("Size", 0)
        relative = key[len(full_prefix):]
        local_path = dst / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not overwrite and local_path.exists():
            if local_path.stat().st_size == size:
                return "skipped"
            log.warning(
                f"Replacing size-mismatched local artifact {local_path}: "
                f"local={local_path.stat().st_size:,}, remote={size:,} bytes"
            )
        try:
            if size >= large_file_bytes:
                cb = _ProgressCallback(size, desc=local_path.name)
                try:
                    client.download_file(
                        bucket, key, str(local_path), Callback=cb,
                        Config=transfer_config,
                    )
                finally:
                    cb.close()
            else:
                client.download_file(
                    bucket, key, str(local_path), Config=transfer_config
                )
            return "downloaded"
        except Exception as e:
            log.error(f"Failed to download {key}: {e}")
            return "failed"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download, obj) for obj in objects]
        with tqdm(total=len(objects), desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                counts[future.result()] += 1
                pbar.update(1)
                pbar.set_postfix(counts)

    log.info(
        f"Download complete — "
        f"downloaded: {counts['downloaded']}, "
        f"skipped: {counts['skipped']}, "
        f"failed: {counts['failed']}"
    )
    return counts


# ── List ───────────────────────────────────────────────────────────────────────

def list_prefix(prefix_path: str, bucket: str, prefix: str) -> list[dict]:
    client = get_s3_client()
    full_prefix = f"{prefix}/{prefix_path}".rstrip("/") + "/"

    paginator = client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        objects.extend(page.get("Contents", []))
    return objects


# ── Artifact sync ──────────────────────────────────────────────────────────────

def _artifact_paths(size: str, run_id: str, stage: str) -> tuple[Path, str]:
    """Return (local_path, s3_prefix) for a named artifact stage.

    S3 keys are grouped by size/run_id so one logical run restores as a unit:
      <size>/<run_id>/<stage>/...
    """
    if stage == "raw":
        return DATA_DIR / "runs" / size / "raw", f"{size}/{run_id}/raw"
    if stage == "curated":
        return DATA_DIR / "runs" / size / "curated", f"{size}/{run_id}/curated"
    if stage == "validated":
        return DATA_DIR / "runs" / size / "validated", f"{size}/{run_id}/validated"
    if stage == "tokenized":
        return DATA_DIR / "runs" / size / "tokenized", f"{size}/{run_id}/tokenized"
    if stage == "tokenizer":
        return DATA_DIR / "runs" / size / "tokenizer", f"{size}/{run_id}/tokenizer"
    if stage == "metadata":
        return DATA_DIR / "runs" / size / "metadata", f"{size}/{run_id}/metadata"
    raise ValueError(
        f"Unknown artifact stage: {stage}. "
        f"Valid stages: {','.join(ALL_ARTIFACT_STAGES)}"
    )


def _artifact_upload_paths(size: str, run_id: str, stage: str) -> tuple[Path, str, str]:
    """Return (local_path, s3_prefix, glob) for artifact uploads."""
    src, dst_prefix = _artifact_paths(size, run_id, stage)
    return src, dst_prefix, "**/*"


def _assert_artifact_stage_complete(
    size: str,
    run_id: str,
    stage: str,
    path: Path,
) -> None:
    """Reject incomplete, stale, or locally mixed artifact stages."""
    if stage == "metadata":
        required = path / "pipeline_manifest.json"
        if not required.exists():
            raise RuntimeError(f"Metadata is incomplete: missing {required}")
        try:
            payload = json.loads(required.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Metadata manifest is invalid: {required}") from exc
        if payload.get("size") != size or payload.get("run_id") != run_id:
            raise RuntimeError(
                f"Metadata manifest does not identify size={size}, "
                f"run_id={run_id}: {required}"
            )
        return

    if stage == "raw":
        invalid = [
            source
            for source in ALL_SOURCES
            if not manifest_outputs_match(path / source)
        ]
        if invalid:
            raise RuntimeError(
                "Raw artifact is not manifest-complete for: "
                + ", ".join(invalid)
            )
        return

    patterns = {
        "curated": "*.json*",
        "validated": "*.json*",
        "tokenized": "[tv]*",
        "tokenizer": "*",
    }
    pattern = patterns.get(stage)
    if pattern is None or not manifest_outputs_match(
        path,
        output_pattern=pattern,
    ):
        raise RuntimeError(
            f"Artifact stage '{stage}' is not manifest-complete: {path}"
        )


def _normalize_stages(stages: str | None) -> list[str]:
    if not stages:
        return list(ARTIFACT_STAGES)

    normalized = [stage.strip() for stage in stages.split(",") if stage.strip()]
    if not normalized:
        raise ValueError("At least one artifact stage is required")

    for stage in normalized:
        if stage not in ALL_ARTIFACT_STAGES:
            raise ValueError(
                f"Unknown artifact stage: {stage}. "
                f"Valid stages: {','.join(ALL_ARTIFACT_STAGES)}"
            )
    return normalized


def upload_artifacts(
    size: str,
    run_id: str,
    stages: str | None,
    bucket: str,
    prefix: str,
    workers: int = 16,
    overwrite: bool = False,
    glob: str = "**/*",
) -> dict[str, int]:
    """Upload selected artifact stages for a size/run_id."""
    totals = {"uploaded": 0, "skipped": 0, "failed": 0}

    _prepare_metadata(size, run_id)

    for stage in _normalize_stages(stages):
        src, dst_prefix, stage_glob = _artifact_upload_paths(size, run_id, stage)
        if not src.exists():
            raise FileNotFoundError(
                f"Requested artifact stage '{stage}' is missing: {src}"
            )
        _assert_artifact_stage_complete(size, run_id, stage, src)

        effective_glob = stage_glob if stage == "metadata" else glob
        log.info(f"Uploading artifact stage '{stage}': {src} → {dst_prefix}")
        counts = upload_directory(
            src=src,
            dst_prefix=dst_prefix,
            bucket=bucket,
            prefix=prefix,
            workers=workers,
            # Metadata is the mutable index for a RUN_ID: later stage uploads
            # must refresh it (for example, validated artifacts are added
            # after curated artifacts). Corpus/model stages remain immutable
            # unless the caller explicitly passes --overwrite.
            overwrite=(overwrite or stage == "metadata"),
            glob=effective_glob,
            mirror=(overwrite or stage == "metadata"),
        )
        for key in totals:
            totals[key] += counts.get(key, 0)

    log.info(
        f"Artifact upload complete — "
        f"uploaded: {totals['uploaded']}, "
        f"skipped: {totals['skipped']}, "
        f"failed: {totals['failed']}"
    )
    if totals["failed"]:
        raise RuntimeError(
            f"Artifact upload failed for {totals['failed']} file(s)"
        )
    return totals


def download_artifacts(
    size: str,
    run_id: str,
    stages: str | None,
    bucket: str,
    prefix: str,
    workers: int = 16,
    overwrite: bool = False,
) -> dict[str, int]:
    """Download selected artifact stages for a size/run_id into local pipeline paths."""
    totals = {"downloaded": 0, "skipped": 0, "failed": 0}

    for stage in _normalize_stages(stages):
        dst, src_prefix = _artifact_paths(size, run_id, stage)
        dst.mkdir(parents=True, exist_ok=True)

        log.info(f"Downloading artifact stage '{stage}': {src_prefix} → {dst}")
        counts = download_prefix(
            src_prefix=src_prefix,
            dst=dst,
            bucket=bucket,
            prefix=prefix,
            workers=workers,
            overwrite=overwrite,
        )
        for key in totals:
            totals[key] += counts.get(key, 0)
        _assert_artifact_stage_complete(size, run_id, stage, dst)

    log.info(
        f"Artifact download complete — "
        f"downloaded: {totals['downloaded']}, "
        f"skipped: {totals['skipped']}, "
        f"failed: {totals['failed']}"
    )
    if totals["failed"]:
        raise RuntimeError(
            f"Artifact download failed for {totals['failed']} file(s)"
        )
    return totals


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLM S3 data utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    up = subparsers.add_parser("upload")
    up.add_argument("--src", type=Path, required=True)
    up.add_argument("--dst", type=str, required=True)
    up.add_argument("--workers", type=int, default=16)
    up.add_argument("--overwrite", action="store_true")
    up.add_argument("--glob", type=str, default="**/*")

    dl = subparsers.add_parser("download")
    dl.add_argument("--src", type=str, required=True)
    dl.add_argument("--dst", type=Path, required=True)
    dl.add_argument("--workers", type=int, default=16)
    dl.add_argument("--overwrite", action="store_true")

    ls = subparsers.add_parser("list")
    ls.add_argument("--prefix", type=str, default="")

    up_artifacts = subparsers.add_parser("artifacts-upload")
    up_artifacts.add_argument("--size", type=str, required=True)
    up_artifacts.add_argument("--run-id", type=str, default=os.environ.get("RUN_ID"))
    up_artifacts.add_argument(
        "--stages",
        type=str,
        default=",".join(ARTIFACT_STAGES),
        help=("Comma-separated artifact stages to upload. Default: raw,tokenized,tokenizer,metadata. Optional archival stages: curated,validated."),
    )
    up_artifacts.add_argument("--workers", type=int, default=16)
    up_artifacts.add_argument("--overwrite", action="store_true")
    up_artifacts.add_argument("--glob", type=str, default="**/*")

    dl_artifacts = subparsers.add_parser("artifacts-download")
    dl_artifacts.add_argument("--size", type=str, required=True)
    dl_artifacts.add_argument("--run-id", type=str, default=os.environ.get("RUN_ID"))
    dl_artifacts.add_argument(
        "--stages",
        type=str,
        default=",".join(ARTIFACT_STAGES),
        help=("Comma-separated artifact stages to download. Default: raw,tokenized,tokenizer,metadata. Optional archival stages: curated,validated."),
    )
    dl_artifacts.add_argument("--workers", type=int, default=16)
    dl_artifacts.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    bucket, prefix = get_bucket_and_prefix()

    if args.command == "upload":
        counts = upload_directory(
            src=args.src,
            dst_prefix=args.dst,
            bucket=bucket,
            prefix=prefix,
            workers=args.workers,
            overwrite=args.overwrite,
            glob=args.glob,
        )
        if counts["failed"]:
            raise SystemExit(1)
    elif args.command == "download":
        counts = download_prefix(
            src_prefix=args.src,
            dst=args.dst,
            bucket=bucket,
            prefix=prefix,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        if counts["failed"]:
            raise SystemExit(1)
    elif args.command == "list":
        objects = list_prefix(args.prefix, bucket, prefix)
        total_size = sum(o["Size"] for o in objects)
        print(f"\n{'Key':<80} {'Size':>10}")
        print("-" * 92)
        for obj in objects:
            print(f"{obj['Key']:<80} {obj['Size']:>10,}")
        print("-" * 92)
        print(f"Total: {len(objects)} objects, {total_size / 1024**3:.2f} GB")
    elif args.command == "artifacts-upload":
        run_id = resolve_upload_run_id(args.size, args.run_id)
        log.info(f"Using RUN_ID={run_id}")
        upload_artifacts(
            size=args.size,
            run_id=run_id,
            stages=args.stages,
            bucket=bucket,
            prefix=prefix,
            workers=args.workers,
            overwrite=args.overwrite,
            glob=args.glob,
        )
    elif args.command == "artifacts-download":
        run_id = require_run_id(args.size, args.run_id)
        log.info(f"Using RUN_ID={run_id}")
        download_artifacts(
            size=args.size,
            run_id=run_id,
            stages=args.stages,
            bucket=bucket,
            prefix=prefix,
            workers=args.workers,
            overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()
