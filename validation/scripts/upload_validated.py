"""
validation/scripts/upload_validated.py
---------------------------------------
Upload validated data to S3 under a versioned path.

S3 path structure:
    {S3_PREFIX}/{target}/{date}/validated/
    e.g. slm/data/125m/2026-04-02/validated/train.jsonl

Mirrors the curated upload in curator/scripts/curate.py but points at
data/validated/ and uses the 'validated' path segment so curated and
validated artifacts are stored independently and never overwrite each other.

Each run gets its own dated folder per target. Re-uploading on the same
day overwrites that day's run. Runs on different days are preserved.

Usage:
    python validation/scripts/upload_validated.py --target 125m

    # Or via make:
    make validate-upload SIZE=125m
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from curator.scripts.upload_s3 import upload_directory, get_bucket_and_prefix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Data directory — override with DATA_DIR env var
DATA_DIR      = Path(os.environ.get("DATA_DIR", "data"))
VALIDATED_DIR = DATA_DIR / "validated"

# Valid targets — must match TARGET_CONFIGS in curate.py
TARGETS = ["mini", "125m", "350m", "1b"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload validated data to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validation/scripts/upload_validated.py --target 125m
  make validate-upload SIZE=125m
        """,
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="125m",
        help="Model size target — used to construct the S3 path. Default: 125m",
    )
    args = parser.parse_args()

    # Verify the validated directory exists and has content
    if not VALIDATED_DIR.exists():
        log.error(
            f"Validated directory not found: {VALIDATED_DIR}. "
            f"Run 'make validate' first."
        )
        sys.exit(1)

    files = list(VALIDATED_DIR.glob("*"))
    if not files:
        log.error(
            f"Validated directory is empty: {VALIDATED_DIR}. "
            f"Run 'make validate' first."
        )
        sys.exit(1)

    bucket, prefix = get_bucket_and_prefix()

    date = datetime.now().strftime("%Y-%m-%d")
    dst_prefix = f"{args.target}/{date}/validated"
    s3_path = f"s3://{bucket}/{prefix}/{dst_prefix}/"

    log.info(f"Uploading {VALIDATED_DIR} → {s3_path}")

    upload_directory(
        src=VALIDATED_DIR,
        dst_prefix=dst_prefix,
        bucket=bucket,
        prefix=prefix,
        overwrite=True,
    )

    log.info(f"Upload complete → {s3_path}")


if __name__ == "__main__":
    main()