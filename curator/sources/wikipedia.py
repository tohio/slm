"""
curator/scripts/upload_s3.py
-----------------------------
S3 upload and download utilities for the SLM data pipeline.

All configuration comes from environment variables — no hardcoded
bucket names, prefixes, or credentials.

Required environment variables (set in .env):
    S3_BUCKET           — S3 bucket name
    S3_PREFIX           — key prefix (e.g. slm/data)
    AWS_ACCESS_KEY_ID   — AWS credentials
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION

Usage:
    # Upload a directory
    python curator/scripts/upload_s3.py upload --src data/curated --dst curated

    # Download a prefix
    python curator/scripts/upload_s3.py download --src curated --dst data/curated

    # List contents
    python curator/scripts/upload_s3.py list --prefix curated
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
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


def get_s3_client():
    """Create and return an S3 client using environment credentials."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def get_bucket_and_prefix() -> tuple[str, str]:
    """Read S3 bucket and prefix from environment."""
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "slm/data").rstrip("/")
    return bucket, prefix


def s3_key(prefix: str, relative_path: str) -> str:
    """Build a full S3 key from the configured prefix and a relative path."""
    return f"{prefix}/{relative_path.lstrip('/')}"


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_file(
    local_path: Path,
    s3_key: str,
    bucket: str,
    client,
    overwrite: bool = False,
) -> bool:
    """
    Upload a single file to S3.

    Args:
        local_path: Path to the local file.
        s3_key: Destination S3 key.
        bucket: S3 bucket name.
        client: Boto3 S3 client.
        overwrite: If False, skip files that already exist in S3.

    Returns:
        True if uploaded, False if skipped.
    """
    if not overwrite:
        try:
            client.head_object(Bucket=bucket, Key=s3_key)
            return False  # already exists, skip
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise

    client.upload_file(str(local_path), bucket, s3_key)
    return True


def upload_directory(
    src: Path,
    dst_prefix: str,
    bucket: str,
    prefix: str,
    workers: int = 8,
    overwrite: bool = False,
    glob: str = "**/*",
) -> dict[str, int]:
    """
    Upload a local directory to S3 recursively.

    Args:
        src: Local source directory.
        dst_prefix: Destination prefix within S3_PREFIX (e.g. "curated").
        bucket: S3 bucket name.
        prefix: S3_PREFIX from environment.
        workers: Number of concurrent upload threads.
        overwrite: If False, skip files that already exist.
        glob: File pattern to match. Default: all files.

    Returns:
        Dict with counts: {"uploaded": N, "skipped": N, "failed": N}
    """
    client = get_s3_client()
    files = [f for f in src.glob(glob) if f.is_file()]

    if not files:
        log.warning(f"No files found in {src} matching pattern '{glob}'")
        return {"uploaded": 0, "skipped": 0, "failed": 0}

    log.info(f"Uploading {len(files)} files from {src} → s3://{bucket}/{prefix}/{dst_prefix}/")

    counts = {"uploaded": 0, "skipped": 0, "failed": 0}

    def _upload(f: Path) -> tuple[str, bool]:
        relative = f.relative_to(src)
        key = s3_key(f"{prefix}/{dst_prefix}", str(relative))
        try:
            uploaded = upload_file(f, key, bucket, client, overwrite=overwrite)
            return "uploaded" if uploaded else "skipped", key
        except Exception as e:
            log.error(f"Failed to upload {f}: {e}")
            return "failed", key

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_upload, f): f for f in files}
        with tqdm(total=len(files), desc="Uploading", unit="file") as pbar:
            for future in as_completed(futures):
                result, key = future.result()
                counts[result] += 1
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
    workers: int = 8,
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Download all objects under an S3 prefix to a local directory.

    Args:
        src_prefix: Source prefix within S3_PREFIX (e.g. "curated").
        dst: Local destination directory.
        bucket: S3 bucket name.
        prefix: S3_PREFIX from environment.
        workers: Number of concurrent download threads.
        overwrite: If False, skip files that already exist locally.

    Returns:
        Dict with counts: {"downloaded": N, "skipped": N, "failed": N}
    """
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
        relative = key[len(full_prefix):]
        local_path = dst / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not overwrite and local_path.exists():
            return "skipped"
        try:
            client.download_file(bucket, key, str(local_path))
            return "downloaded"
        except Exception as e:
            log.error(f"Failed to download {key}: {e}")
            return "failed"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download, obj) for obj in objects]
        with tqdm(total=len(objects), desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                counts[result] += 1
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
    """List all objects under a given S3 prefix."""
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

    # Upload
    up = subparsers.add_parser("upload", help="Upload a local directory to S3")
    up.add_argument("--src", type=Path, required=True, help="Local source directory")
    up.add_argument("--dst", type=str, required=True, help="Destination prefix within S3_PREFIX")
    up.add_argument("--workers", type=int, default=8)
    up.add_argument("--overwrite", action="store_true")
    up.add_argument("--glob", type=str, default="**/*")

    # Download
    dl = subparsers.add_parser("download", help="Download an S3 prefix to a local directory")
    dl.add_argument("--src", type=str, required=True, help="Source prefix within S3_PREFIX")
    dl.add_argument("--dst", type=Path, required=True, help="Local destination directory")
    dl.add_argument("--workers", type=int, default=8)
    dl.add_argument("--overwrite", action="store_true")

    # List
    ls = subparsers.add_parser("list", help="List objects under an S3 prefix")
    ls.add_argument("--prefix", type=str, default="", help="Prefix within S3_PREFIX")

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