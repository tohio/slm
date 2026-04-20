"""
pretrain/data/upload_tokenized.py
----------------------------------
Upload and download the tokenized binary dataset to/from S3.

The tokenized binary (train.bin + val.bin) is large — at 1b scale it can
exceed 50GB — and expensive to regenerate on a GPU instance. Uploading
from the CPU curation instance and downloading on the GPU training
instance avoids re-running make tokenize on expensive hardware.

S3 path structure:
    {S3_PREFIX}/{target}/{date}/tokenized/
    e.g. slm/data/125m/2026-04-02/tokenized/train.bin

Each run is versioned by target and date. If you re-upload on the same
day, this script WARNS that existing objects will be overwritten and
then proceeds — same-day re-uploads are frequent during pipeline
development. Runs on different days are preserved independently.

Note: The tokenizer itself (data/tokenizer/) does not need S3 — it is
pushed to HuggingFace Hub as part of make export and can be pulled from
there in seconds. Only the tokenized binaries need S3.

Usage:
    # Upload (run on CPU curation instance after make tokenize)
    python pretrain/data/upload_tokenized.py upload --target 125m

    # Download (run on GPU training instance before make pretrain)
    python pretrain/data/upload_tokenized.py download --target 125m --date 2026-04-12

    # Or via make:
    make tokenize-upload SIZE=125m
    make tokenize-download SIZE=125m DATE=2026-04-12
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from curator.scripts.upload_s3 import (
    upload_directory,
    download_prefix,
    get_bucket_and_prefix,
    get_s3_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR      = Path(os.environ.get("DATA_DIR", "data"))
TOKENIZED_DIR = DATA_DIR / "tokenized"

TARGETS = ["mini", "125m", "350m", "1b"]


def _list_existing_objects(bucket: str, full_prefix: str) -> list[dict]:
    """List objects currently at the S3 prefix. Returns [] if none."""
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        objects.extend(page.get("Contents", []))
    return objects


def cmd_upload(target: str) -> None:
    """Upload tokenized binaries to S3 under a versioned path."""
    if not TOKENIZED_DIR.exists() or not list(TOKENIZED_DIR.glob("*.bin")):
        log.error(
            f"No tokenized binary found in {TOKENIZED_DIR}. "
            f"Run 'make tokenize' first."
        )
        sys.exit(1)

    bucket, prefix = get_bucket_and_prefix()
    date = datetime.now().strftime("%Y-%m-%d")
    dst_prefix = f"{target}/{date}/tokenized"
    full_prefix = f"{prefix}/{dst_prefix}"
    s3_path = f"s3://{bucket}/{full_prefix}/"

    # ── Warn on same-day overwrite ────────────────────────────────────────────
    # Upload is unconditional (same-day re-uploads are frequent during
    # development), but we log a clear warning listing what's about to be
    # overwritten so the user knows they're replacing an existing version.
    existing = _list_existing_objects(bucket, f"{full_prefix}/")
    if existing:
        total_bytes = sum(o["Size"] for o in existing)
        log.warning(
            f"OVERWRITE: {len(existing)} objects ({total_bytes / 1e9:.2f} GB) "
            f"already exist at {s3_path} and will be replaced."
        )
        for obj in existing:
            log.warning(f"  will overwrite: {obj['Key']} ({obj['Size'] / 1e9:.2f} GB)")

    # Log file sizes so the user knows what they're uploading
    log.info("Uploading files:")
    for f in sorted(TOKENIZED_DIR.iterdir()):
        size_gb = f.stat().st_size / 1e9
        log.info(f"  {f.name}: {size_gb:.2f} GB")

    log.info(f"Uploading {TOKENIZED_DIR} → {s3_path}")

    upload_directory(
        src=TOKENIZED_DIR,
        dst_prefix=dst_prefix,
        bucket=bucket,
        prefix=prefix,
        overwrite=True,
    )

    log.info(f"Upload complete → {s3_path}")
    log.info(f"To download on GPU instance: make tokenize-download SIZE={target} DATE={date}")


def cmd_download(target: str, date: str) -> None:
    """Download tokenized binaries from S3 to local TOKENIZED_DIR."""
    bucket, prefix = get_bucket_and_prefix()
    src_prefix = f"{target}/{date}/tokenized"
    s3_path = f"s3://{bucket}/{prefix}/{src_prefix}/"

    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading {s3_path} → {TOKENIZED_DIR}")

    download_prefix(
        src_prefix=src_prefix,
        dst=TOKENIZED_DIR,
        bucket=bucket,
        prefix=prefix,
    )

    # Verify the binaries landed correctly
    bins = list(TOKENIZED_DIR.glob("*.bin"))
    if not bins:
        log.error("Download completed but no .bin file found — check S3 path")
        sys.exit(1)

    for b in bins:
        log.info(f"  {b.name}: {b.stat().st_size / 1e9:.2f} GB")

    # Warn if val.bin is missing — expected after the val-split refactor
    if not (TOKENIZED_DIR / "val.bin").exists():
        log.warning(
            "val.bin not found in download — this upload predates the val "
            "split refactor. Training will fail unless you retokenize. "
            "Re-run: make tokenize && make tokenize-upload SIZE={target}"
        )

    log.info("Download complete — ready for make pretrain")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload/download tokenized binaries to/from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload after make tokenize (run on CPU instance)
  python pretrain/data/upload_tokenized.py upload --target 125m

  # Download before make pretrain (run on GPU instance)
  python pretrain/data/upload_tokenized.py download --target 125m --date 2026-04-12

  # Via make
  make tokenize-upload SIZE=125m
  make tokenize-download SIZE=125m DATE=2026-04-12
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload subcommand
    up = subparsers.add_parser("upload", help="Upload tokenized binaries to S3")
    up.add_argument(
        "--target",
        choices=TARGETS,
        default="125m",
        help="Model size target — used to construct the S3 path. Default: 125m",
    )

    # Download subcommand
    down = subparsers.add_parser("download", help="Download tokenized binaries from S3")
    down.add_argument(
        "--target",
        choices=TARGETS,
        default="125m",
        help="Model size target. Default: 125m",
    )
    down.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Upload date (YYYY-MM-DD). Defaults to today.",
    )

    args = parser.parse_args()

    if args.command == "upload":
        cmd_upload(args.target)
    elif args.command == "download":
        cmd_download(args.target, args.date)


if __name__ == "__main__":
    main()