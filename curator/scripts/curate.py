"""
curator/scripts/curate.py
--------------------------
Main data curation pipeline.

Orchestrates all data sources, quality filters, and deduplication
into a single configurable pipeline. Produces clean JSONL files
ready for tokenizer training and model pretraining.

Pipeline:
    1. Download sources (Wikipedia, CodeSearchNet, Common Crawl)
    2. Apply quality filters (heuristics from FineWeb/Gopher)
    3. Deduplicate (exact + MinHash LSH)
    4. Blend sources to target token ratios
    5. Upload to S3

Output structure:
    data/
    ├── raw/
    │   ├── wikipedia/       raw Wikipedia JSONL shards
    │   ├── code/            raw CodeSearchNet JSONL shards
    │   └── common_crawl/    raw Common Crawl JSONL shards
    ├── filtered/
    │   ├── wikipedia/       quality filtered
    │   ├── code/            quality filtered
    │   └── common_crawl/    quality filtered + deduped
    └── curated/
        └── train.jsonl      final blended dataset

Usage:
    # Full pipeline
    python curator/scripts/curate.py --target 125m

    # Individual stages
    python curator/scripts/curate.py --target 125m --stage download
    python curator/scripts/curate.py --target 125m --stage filter
    python curator/scripts/curate.py --target 125m --stage dedup
    python curator/scripts/curate.py --target 125m --stage blend
    python curator/scripts/curate.py --target 125m --stage upload
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from curator.filters.dedup import Deduplicator
from curator.filters.quality import QualityFilter, QualityConfig
from curator.sources.common_crawl import CommonCrawlSource
from curator.sources.code_search_net import CodeSearchNetSource
from curator.sources.wikipedia import WikipediaSource
from curator.scripts.upload_s3 import upload_directory, download_prefix, get_bucket_and_prefix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Target configurations ──────────────────────────────────────────────────────
# Token targets per model size — source mix stays constant at 70/20/10

TARGET_CONFIGS = {
    "125m": {
        "total_tokens": 3_000_000_000,
        "cc_segments": 10,
        "cc_crawls": ["CC-MAIN-2024-10"],
    },
    "350m": {
        "total_tokens": 10_000_000_000,
        "cc_segments": 40,
        "cc_crawls": ["CC-MAIN-2024-10", "CC-MAIN-2023-50"],
    },
    "1b": {
        "total_tokens": 25_000_000_000,
        "cc_segments": 100,
        "cc_crawls": ["CC-MAIN-2024-10", "CC-MAIN-2023-50", "CC-MAIN-2023-40"],
    },
}

# Source mix — fraction of total tokens per source
SOURCE_MIX = {
    "common_crawl": 0.70,
    "wikipedia": 0.20,
    "code": 0.10,
}

# Data directories
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
CURATED_DIR = DATA_DIR / "curated"


# ── Stage 1: Download ──────────────────────────────────────────────────────────

def stage_download(target: str) -> None:
    """Download all data sources."""
    cfg = TARGET_CONFIGS[target]
    log.info(f"=== Stage 1: Download (target={target}) ===")

    # Wikipedia
    log.info("Downloading Wikipedia EN...")
    wiki = WikipediaSource(output_dir=RAW_DIR / "wikipedia")
    wiki.download()
    log.info(f"Wikipedia stats: {wiki.stats()}")

    # CodeSearchNet
    log.info("Downloading CodeSearchNet...")
    code = CodeSearchNetSource(output_dir=RAW_DIR / "code")
    code.download()
    log.info(f"CodeSearchNet stats: {code.stats()}")

    # Common Crawl
    log.info("Downloading Common Crawl...")
    cc = CommonCrawlSource(
        output_dir=RAW_DIR / "common_crawl",
        crawls=cfg["cc_crawls"],
        max_segments=cfg["cc_segments"],
    )
    cc.download()
    log.info(f"Common Crawl stats: {cc.stats()}")


# ── Stage 2: Filter ────────────────────────────────────────────────────────────

def stage_filter() -> None:
    """Apply quality filters to all raw data."""
    log.info("=== Stage 2: Quality Filter ===")

    sources = ["wikipedia", "code", "common_crawl"]
    for source in sources:
        src_dir = RAW_DIR / source
        dst_dir = FILTERED_DIR / source
        dst_dir.mkdir(parents=True, exist_ok=True)

        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"No shards found in {src_dir} — skipping")
            continue

        log.info(f"Filtering {source}: {len(shards)} shards...")
        qf = QualityFilter()

        for shard in shards:
            out_path = dst_dir / shard.name
            if out_path.exists():
                log.debug(f"  Skipping {shard.name} — already filtered")
                continue

            with open(shard) as fin, open(out_path, "w") as fout:
                for line in fin:
                    record = json.loads(line)
                    kept, reason = qf.check(record)
                    if kept:
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        log.info(qf.report())


# ── Stage 3: Deduplicate ───────────────────────────────────────────────────────

def stage_dedup() -> None:
    """Deduplicate filtered data with exact + MinHash LSH."""
    log.info("=== Stage 3: Deduplication ===")

    # Dedup index persisted across shards for cross-shard dedup
    index_path = DATA_DIR / "dedup_index.pkl"

    sources = ["wikipedia", "code", "common_crawl"]
    for source in sources:
        src_dir = FILTERED_DIR / source
        dst_dir = FILTERED_DIR / f"{source}_deduped"
        dst_dir.mkdir(parents=True, exist_ok=True)

        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"No filtered shards found in {src_dir} — skipping")
            continue

        log.info(f"Deduplicating {source}: {len(shards)} shards...")
        dedup = Deduplicator(index_path=index_path)

        for shard in shards:
            out_path = dst_dir / shard.name
            if out_path.exists():
                log.debug(f"  Skipping {shard.name} — already deduped")
                continue
            dedup.deduplicate_jsonl(shard, out_path)

        # Save index after each source for resumability
        dedup.save(index_path)
        log.info(dedup.report())


# ── Stage 4: Blend ─────────────────────────────────────────────────────────────

def stage_blend(target: str, seed: int = 42) -> None:
    """
    Blend sources to the target token ratio and write final train.jsonl.

    Samples from each source proportionally to the SOURCE_MIX ratios.
    Uses character count as a proxy for token count (4 chars ≈ 1 token).
    """
    log.info(f"=== Stage 4: Blend (target={target}) ===")
    cfg = TARGET_CONFIGS[target]
    total_tokens = cfg["total_tokens"]

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CURATED_DIR / "train.jsonl"

    if output_path.exists():
        log.info(f"train.jsonl already exists — delete to re-blend")
        return

    rng = random.Random(seed)
    source_dirs = {
        "common_crawl": FILTERED_DIR / "common_crawl_deduped",
        "wikipedia": FILTERED_DIR / "wikipedia_deduped",
        "code": FILTERED_DIR / "code_deduped",
    }

    # Collect all records per source
    source_records: dict[str, list[dict]] = {}
    for source, src_dir in source_dirs.items():
        shards = sorted(src_dir.glob("*.jsonl"))
        records = []
        for shard in shards:
            with open(shard) as f:
                for line in f:
                    records.append(json.loads(line))
        log.info(f"  {source}: {len(records):,} documents")
        source_records[source] = records

    # Target chars per source (4 chars ≈ 1 token)
    target_chars = {
        source: int(total_tokens * fraction * 4)
        for source, fraction in SOURCE_MIX.items()
    }

    # Sample from each source to hit target chars
    blended = []
    for source, records in source_records.items():
        target = target_chars[source]
        rng.shuffle(records)
        chars = 0
        for record in records:
            if chars >= target:
                break
            blended.append(record)
            chars += len(record["text"])
        log.info(
            f"  {source}: sampled {len([r for r in blended if r.get('source') == source]):,} docs "
            f"({chars / 1e9:.2f}B chars)"
        )

    # Shuffle the final blend
    rng.shuffle(blended)

    # Write
    log.info(f"Writing {len(blended):,} documents to {output_path}...")
    with open(output_path, "w") as f:
        for record in blended:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_chars = sum(len(r["text"]) for r in blended)
    log.info(
        f"Blend complete — "
        f"{len(blended):,} documents, "
        f"{total_chars / 1e9:.2f}B chars, "
        f"~{total_chars // 4 / 1e9:.2f}B tokens"
    )

    # Write blend stats
    stats_path = CURATED_DIR / "blend_stats.json"
    source_counts = {}
    for r in blended:
        s = r.get("source", "unknown")
        source_counts[s] = source_counts.get(s, 0) + 1
    with open(stats_path, "w") as f:
        json.dump({
            "target": target,
            "total_documents": len(blended),
            "total_chars": total_chars,
            "estimated_tokens": total_chars // 4,
            "source_mix": source_counts,
        }, f, indent=2)
    log.info(f"Blend stats written to {stats_path}")


# ── Stage 5: Upload ────────────────────────────────────────────────────────────

def stage_upload() -> None:
    """Upload curated data to S3."""
    log.info("=== Stage 5: Upload to S3 ===")
    bucket, prefix = get_bucket_and_prefix()

    upload_directory(
        src=CURATED_DIR,
        dst_prefix="curated",
        bucket=bucket,
        prefix=prefix,
        overwrite=False,
    )
    log.info("Upload complete")


# ── CLI ────────────────────────────────────────────────────────────────────────

STAGES = ["download", "filter", "dedup", "blend", "upload", "all"]


def main():
    parser = argparse.ArgumentParser(
        description="SLM data curation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=list(TARGET_CONFIGS.keys()),
        default="125m",
        help="Model size target — controls how much data to collect",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Pipeline stage to run. Default: all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for blend stage",
    )
    args = parser.parse_args()

    log.info(f"SLM Curation Pipeline — target={args.target}, stage={args.stage}")

    if args.stage in ("download", "all"):
        stage_download(args.target)

    if args.stage in ("filter", "all"):
        stage_filter()

    if args.stage in ("dedup", "all"):
        stage_dedup()

    if args.stage in ("blend", "all"):
        stage_blend(args.target, seed=args.seed)

    if args.stage in ("upload", "all"):
        stage_upload()

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()