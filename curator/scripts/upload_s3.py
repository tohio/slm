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
    AWS_DEFAULT_REGION  — (default: us-east-1)

    AWS credentials: standard boto3 chain — env vars, profile, IAM role.
"""

import argparse
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

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


def validate_date(date: str) -> None:
    """Validate artifact DATE format: YYYY-MM-DD."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"DATE must be YYYY-MM-DD, got: {date}")


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
    if not overwrite:
        log.info("  Listing existing objects to skip already-uploaded files...")
        existing = _list_existing_keys(client, bucket, full_prefix)
        log.info(f"  {len(existing)} objects already present")

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
        log.warning(f"No objects found at s3://{bucket}/{full_prefix}")
        return {"downloaded": 0, "skipped": 0, "failed": 0}

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
            return "skipped"
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

def _artifact_paths(size: str, date: str, stage: str) -> tuple[Path, str]:
    """Return (local_path, s3_prefix) for a named curation/training artifact.

    Stage names are intentionally user-facing and stable:
      - raw:        downloaded source data under data/runs/<size>/raw
      - tokenized:  pretraining binaries under data/runs/<size>/tokenized
      - tokenizer:  tokenizer files under data/runs/<size>/tokenizer
      - metadata:   run metadata under data/runs/<size>/metadata
      - curated:    optional blended train/val JSONL under data/runs/<size>/curated
      - validated:  optional validated train/val JSONL under data/runs/<size>/validated

    S3 keys are grouped by size/date so a prior run can be restored exactly:
      <size>/<date>/<artifact>/...
    """
    if stage == "raw":
        return DATA_DIR / "runs" / size / "raw", f"{size}/{date}/raw"
    if stage == "curated":
        return DATA_DIR / "runs" / size / "curated", f"{size}/{date}/curated"
    if stage == "validated":
        return DATA_DIR / "runs" / size / "validated", f"{size}/{date}/validated"
    if stage == "tokenized":
        return DATA_DIR / "runs" / size / "tokenized", f"{size}/{date}/tokenized"
    if stage == "tokenizer":
        return DATA_DIR / "runs" / size / "tokenizer", f"{size}/{date}/tokenizer"
    if stage == "metadata":
        return DATA_DIR / "runs" / size / "metadata", f"{size}/{date}/metadata"
    raise ValueError(
        f"Unknown artifact stage: {stage}. "
        f"Valid stages: {','.join(ALL_ARTIFACT_STAGES)}"
    )


def _artifact_upload_paths(size: str, date: str, stage: str) -> tuple[Path, str, str]:
    """Return (local_path, s3_prefix, glob) for artifact uploads.

    The metadata stage is a stable downstream contract:
      local: data/runs/<size>/curated/blend_stats.json
      s3:   <size>/<date>/metadata/blend_stats.json

    Curated and validated JSONL stages remain optional archival stages and are
    not included in ARTIFACT_STAGES by default.
    """
    if stage == "metadata":
        return (
            DATA_DIR / "runs" / size / "curated",
            f"{size}/{date}/metadata",
            "blend_stats.json",
        )

    src, dst_prefix = _artifact_paths(size, date, stage)
    return src, dst_prefix, "**/*"


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
    date: str,
    stages: str | None,
    bucket: str,
    prefix: str,
    workers: int = 16,
    overwrite: bool = False,
    glob: str = "**/*",
) -> dict[str, int]:
    """Upload selected artifact stages for a size/date."""
    totals = {"uploaded": 0, "skipped": 0, "failed": 0}

    for stage in _normalize_stages(stages):
        src, dst_prefix, stage_glob = _artifact_upload_paths(size, date, stage)
        if not src.exists():
            log.warning(f"Skipping missing artifact stage '{stage}': {src}")
            continue

        effective_glob = stage_glob if stage == "metadata" else glob
        log.info(f"Uploading artifact stage '{stage}': {src} → {dst_prefix}")
        counts = upload_directory(
            src=src,
            dst_prefix=dst_prefix,
            bucket=bucket,
            prefix=prefix,
            workers=workers,
            overwrite=overwrite,
            glob=effective_glob,
        )
        for key in totals:
            totals[key] += counts.get(key, 0)

    log.info(
        f"Artifact upload complete — "
        f"uploaded: {totals['uploaded']}, "
        f"skipped: {totals['skipped']}, "
        f"failed: {totals['failed']}"
    )
    return totals


def download_artifacts(
    size: str,
    date: str,
    stages: str | None,
    bucket: str,
    prefix: str,
    workers: int = 16,
    overwrite: bool = False,
) -> dict[str, int]:
    """Download selected artifact stages for a size/date into local pipeline paths."""
    totals = {"downloaded": 0, "skipped": 0, "failed": 0}

    for stage in _normalize_stages(stages):
        dst, src_prefix = _artifact_paths(size, date, stage)
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

    log.info(
        f"Artifact download complete — "
        f"downloaded: {totals['downloaded']}, "
        f"skipped: {totals['skipped']}, "
        f"failed: {totals['failed']}"
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
    up_artifacts.add_argument("--date", type=str, required=True)
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
    dl_artifacts.add_argument("--date", type=str, required=True)
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
        upload_directory(
            src=args.src,
            dst_prefix=args.dst,
            bucket=bucket,
            prefix=prefix,
            workers=args.workers,
            overwrite=args.overwrite,
            glob=args.glob,
        )
    elif args.command == "download":
        download_prefix(
            src_prefix=args.src,
            dst=args.dst,
            bucket=bucket,
            prefix=prefix,
            workers=args.workers,
            overwrite=args.overwrite,
        )
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
        validate_date(args.date)
        upload_artifacts(
            size=args.size,
            date=args.date,
            stages=args.stages,
            bucket=bucket,
            prefix=prefix,
            workers=args.workers,
            overwrite=args.overwrite,
            glob=args.glob,
        )
    elif args.command == "artifacts-download":
        validate_date(args.date)
        download_artifacts(
            size=args.size,
            date=args.date,
            stages=args.stages,
            bucket=bucket,
            prefix=prefix,
            workers=args.workers,
            overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()