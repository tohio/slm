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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
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

# Default transfer config: adaptive retries, reasonable connect/read timeouts.
_BOTO_CONFIG = Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=120,
    max_pool_connections=64,
)


def get_s3_client():
    """
    S3 client using boto3's default credential chain.

    Explicitly passing aws_access_key_id/secret breaks IAM role auth on EC2.
    Boto3 will find creds automatically from: env vars → ~/.aws → IAM role.
    """
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=_BOTO_CONFIG,
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
) -> bool:
    """Upload one file, optionally with a progress callback."""
    if show_progress:
        size = local.stat().st_size
        cb = _ProgressCallback(size, desc=local.name)
        try:
            client.upload_file(str(local), bucket, key, Callback=cb)
        finally:
            cb.close()
    else:
        client.upload_file(str(local), bucket, key)
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
    client = get_s3_client()
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
            _upload_one(f, key, bucket, client, show_progress=show_progress)
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
    client = get_s3_client()
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
                    )
                finally:
                    cb.close()
            else:
                client.download_file(bucket, key, str(local_path))
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


if __name__ == "__main__":
    main()