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
    3. Deduplicate (exact hash + datatrove disk-based MinHash LSH)
    4. Blend sources to target token ratios
    5. Upload to S3

Deduplication uses datatrove's 4-stage disk-based MinHash pipeline,
replacing the datasketch in-memory LSH. RAM usage is bounded by shard
size, not corpus size — scales to 125m, 350m, and 1b with the same
memory footprint.

Blend uses streaming reservoir sampling + offset-based shuffle.
Peak RAM during blend is O(1) regardless of corpus size.

S3 upload path structure:
    {S3_PREFIX}/{target}/{date}/curated/
    e.g. slm/data/125m/2026-04-02/curated/train.jsonl

CC segment calibration (empirical, from 125m run):
    10 segments → ~127k raw docs → ~84k deduped → ~60M tokens
    ~6M tokens per segment after filtering and dedup
    Target CC tokens = total_tokens * 0.70 (SOURCE_MIX)
    Segments needed  = target_cc_tokens / 6M

Output structure:
    data/
    ├── raw/
    │   ├── wikipedia/          raw Wikipedia JSONL shards
    │   ├── code/               raw CodeSearchNet JSONL shards
    │   └── common_crawl/       raw Common Crawl JSONL shards
    ├── filtered/
    │   ├── wikipedia/          quality filtered
    │   ├── wikipedia_deduped/  quality filtered + deduplicated
    │   ├── code/               quality filtered
    │   ├── code_deduped/       quality filtered + deduplicated
    │   ├── common_crawl/       quality filtered
    │   └── common_crawl_deduped/ quality filtered + deduplicated
    └── curated/
        ├── train.jsonl         final blended dataset
        └── blend_stats.json    source mix breakdown

Usage:
    # Full pipeline
    python curator/scripts/curate.py --target 125m

    # Minimal run — validates pipeline end-to-end with tiny data volumes
    python curator/scripts/curate.py --target mini --mini

    # Individual stages
    python curator/scripts/curate.py --target 125m --stage download
    python curator/scripts/curate.py --target 125m --stage filter
    python curator/scripts/curate.py --target 125m --stage dedup
    python curator/scripts/curate.py --target 125m --stage blend
    python curator/scripts/curate.py --target 125m --stage upload

    # Control parallelism
    python curator/scripts/curate.py --target 125m --workers 8
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from curator.filters.dedup import Deduplicator
from curator.filters.quality import QualityFilter, QualityConfig
from curator.sources.common_crawl import CommonCrawlSource
from curator.sources.code_search_net import CodeSearchNetSource, LANGUAGES
from curator.sources.wikipedia import WikipediaSource
from curator.scripts.upload_s3 import upload_directory, download_prefix, get_bucket_and_prefix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Target configurations ──────────────────────────────────────────────────────
# Token targets per model size — source mix stays constant at 70/20/10.
# mini target is for pipeline validation only — not for training.
#
# Targets are set to give comfortable headroom above Chinchilla compute-optimal:
#   125m — optimal ~2.5B, target 5B  (~2× optimal)
#   350m — optimal ~7B,   target 20B (~3× optimal)
#   1b   — optimal ~20B,  target 50B (~2.5× optimal)
#
# CC segment calibration (empirical, from 125m run):
#   ~6M tokens per segment after filtering and dedup
#   Target CC tokens = total_tokens × 0.70 (SOURCE_MIX)
#   Segments needed  = target_cc_tokens / 6M
#
# For diversity, 350m and 1b spread segments across multiple crawls.
# Each CC crawl has ~90k WARC segments — well above our segment counts.

TARGET_CONFIGS = {
    "mini": {
        "total_tokens":  1_000_000,       # 1M tokens — pipeline validation only
        "cc_segments":   2,               # 2 WARC segments
        "cc_crawls":     ["CC-MAIN-2024-10"],
    },
    "125m": {
        "total_tokens":  5_000_000_000,   # 5B tokens (~2× Chinchilla optimal for 125m)
        "cc_segments":   584,             # 5B × 0.70 / 6M ≈ 584 segments
        "cc_crawls":     ["CC-MAIN-2024-10"],
    },
    "350m": {
        "total_tokens":  20_000_000_000,  # 20B tokens (~3× Chinchilla optimal for 350m)
        "cc_segments":   1_167,           # 20B × 0.70 / 6M ≈ 1167 segments, split across 2 crawls
        "cc_crawls":     ["CC-MAIN-2024-10", "CC-MAIN-2023-50"],
    },
    "1b": {
        "total_tokens":  50_000_000_000,  # 50B tokens (~2.5× Chinchilla optimal for 1b)
        "cc_segments":   1_945,           # 50B × 0.70 / 6M ≈ 1945 segments, split across 3 crawls
        "cc_crawls":     ["CC-MAIN-2024-10", "CC-MAIN-2023-50", "CC-MAIN-2023-40"],
    },
}

# Source mix — fraction of total tokens per source
SOURCE_MIX = {
    "common_crawl": 0.70,
    "wikipedia":    0.20,
    "code":         0.10,
}

# Mini run overrides — passed to source constructors when --mini is set
MINI_OVERRIDES = {
    "wiki_max_docs":    5_000,
    "code_max_docs":    10_000,
    "code_languages":   ["python", "javascript"],   # 2 of 6 languages
}

# Data directories — override with DATA_DIR env var
DATA_DIR     = Path(os.environ.get("DATA_DIR", "data"))
RAW_DIR      = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
CURATED_DIR  = DATA_DIR / "curated"


# ── Helpers ────────────────────────────────────────────────────────────────────

def flatten_datatrove_record(record: dict) -> dict:
    """
    Flatten datatrove's document format back to a flat dict.

    datatrove wraps documents as:
        {"text": "...", "id": "...", "metadata": {"source": ..., "url": ..., ...}}

    We flatten this back to:
        {"text": "...", "source": ..., "url": ..., ...}

    Plain JSONL records (not processed by datatrove) are returned unchanged.
    """
    if "metadata" in record and isinstance(record["metadata"], dict):
        flat = {"text": record.get("text", "")}
        flat.update(record["metadata"])
        # drop datatrove internals
        flat.pop("file_path", None)
        return flat
    return record


# ── Stage 1: Download ──────────────────────────────────────────────────────────

def stage_download(target: str, mini: bool = False) -> None:
    """Download all data sources."""
    cfg = TARGET_CONFIGS[target]
    log.info(f"=== Stage 1: Download (target={target}, mini={mini}) ===")

    # Wikipedia
    log.info("Downloading Wikipedia EN...")
    wiki = WikipediaSource(
        output_dir=RAW_DIR / "wikipedia",
        max_docs=MINI_OVERRIDES["wiki_max_docs"] if mini else None,
    )
    wiki.download()
    log.info(f"Wikipedia stats: {wiki.stats()}")

    # CodeSearchNet
    log.info("Downloading CodeSearchNet...")
    code = CodeSearchNetSource(
        output_dir=RAW_DIR / "code",
        languages=MINI_OVERRIDES["code_languages"] if mini else LANGUAGES,
        max_docs=MINI_OVERRIDES["code_max_docs"] if mini else None,
    )
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

def stage_dedup(workers: int | None = None) -> None:
    """
    Deduplicate filtered data using exact hash + datatrove MinHash LSH.

    Two stages per source:
        1. Exact dedup  — SHA-256 streaming pass, shared cross-source index.
        2. Fuzzy dedup  — datatrove 4-stage disk pipeline.
                          Peak RAM is O(shard_size), not O(corpus_size).
    """
    log.info("=== Stage 3: Deduplication (datatrove MinHash) ===")

    n_workers = workers or max(1, (os.cpu_count() or 4) // 2)
    working_dir = DATA_DIR / "dedup_scratch"

    dedup = Deduplicator(working_dir=working_dir, workers=n_workers)

    sources = ["wikipedia", "code", "common_crawl"]
    for source in sources:
        src_dir = FILTERED_DIR / source
        dst_dir = FILTERED_DIR / f"{source}_deduped"

        shards = list(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"No filtered shards found in {src_dir} — skipping")
            continue

        existing = list(dst_dir.glob("*.jsonl")) if dst_dir.exists() else []
        if existing:
            log.info(f"  {source}: already deduped ({len(existing)} shards) — skipping")
            continue

        dedup.deduplicate_source(
            src_dir=src_dir,
            dst_dir=dst_dir,
            source_name=source,
        )

    log.info(dedup.report())


# ── Stage 4: Blend ─────────────────────────────────────────────────────────────

def stage_blend(target: str, seed: int = 42) -> None:
    """
    Blend sources to the target token ratio and write final train.jsonl.

    Streams each source to hit its target char count, writes per-source
    staging files, then merges and shuffles using a byte-offset index.
    Peak RAM is O(1) — no source is loaded into memory in full at any point.

    Handles both plain JSONL (from filter stage) and datatrove's wrapped
    format {"text": ..., "id": ..., "metadata": {...}}, flattening both
    to a consistent {"text": ..., "source": ..., ...} output format.

    Uses character count as a proxy for token count (4 chars ≈ 1 token).
    """
    log.info(f"=== Stage 4: Blend (target={target}) ===")
    cfg = TARGET_CONFIGS[target]
    total_tokens = cfg["total_tokens"]

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CURATED_DIR / "train.jsonl"

    if output_path.exists():
        log.info("train.jsonl already exists — delete to re-blend")
        return

    rng = random.Random(seed)

    source_dirs = {
        "common_crawl": FILTERED_DIR / "common_crawl_deduped",
        "wikipedia":    FILTERED_DIR / "wikipedia_deduped",
        "code":         FILTERED_DIR / "code_deduped",
    }

    # Target chars per source (4 chars ≈ 1 token)
    target_chars = {
        source: int(total_tokens * fraction * 4)
        for source, fraction in SOURCE_MIX.items()
    }

    # ── Pass 1: stream each source to a per-source staging file ───────────────
    staging_paths = {}
    source_stats = {}

    for source, src_dir in source_dirs.items():
        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"  {source}: no deduped shards found — skipping")
            continue

        staging = CURATED_DIR / f"blend_{source}.jsonl"
        staging_paths[source] = staging

        # Use a distinct variable name to avoid shadowing the outer `target`
        # (the model size string e.g. "125m"). The original code reused `target`
        # here, which caused blend_stats.json to record the last source's char
        # target (an int) instead of the model size string.
        source_char_target = target_chars[source]
        chars = 0
        docs = 0

        with open(staging, "w") as fout:
            for shard in shards:
                if chars >= source_char_target:
                    break
                with open(shard) as fin:
                    for line in fin:
                        record = json.loads(line)

                        # Flatten datatrove format if needed
                        record = flatten_datatrove_record(record)

                        chars += len(record.get("text", ""))
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        docs += 1
                        if chars >= source_char_target:
                            break

        log.info(
            f"  {source}: {docs:,} docs, "
            f"{chars / 1e9:.3f}B chars, "
            f"~{chars // 4 / 1e6:.1f}M tokens"
        )
        source_stats[source] = {"docs": docs, "chars": chars}

    # ── Pass 2: merge staging files into one (interleaved) ────────────────────
    merged_path = CURATED_DIR / "blend_merged.jsonl"
    total_docs = 0
    total_chars = 0

    with open(merged_path, "w") as fout:
        for source, staging in staging_paths.items():
            with open(staging) as fin:
                for line in fin:
                    fout.write(line)
                    total_docs += 1
            staging.unlink()

    log.info(f"Merged {total_docs:,} documents")

    # ── Pass 3: shuffle via byte-offset index ─────────────────────────────────
    log.info("Building line offset index for shuffle...")
    offsets = []
    with open(merged_path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(offset)
            total_chars += len(line) - 1

    log.info(f"Shuffling {len(offsets):,} line offsets...")
    rng.shuffle(offsets)

    log.info(f"Writing shuffled output to {output_path}...")
    with open(merged_path, "rb") as fin, open(output_path, "wb") as fout:
        for offset in offsets:
            fin.seek(offset)
            fout.write(fin.readline())

    merged_path.unlink()

    log.info(
        f"Blend complete — "
        f"{total_docs:,} documents, "
        f"~{total_chars // 4 / 1e9:.2f}B tokens"
    )

    # ── Write blend stats ─────────────────────────────────────────────────────
    # Note: `target` here is the model size string (e.g. "125m"), not a char
    # count. Previously a variable name collision caused this to be written as
    # the last source's char target (an int) instead of the model size string.
    stats_path = CURATED_DIR / "blend_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "target": target,
            "target_tokens": total_tokens,
            "total_documents": total_docs,
            "estimated_tokens": total_chars // 4,
            "source_mix": {
                s: {"docs": v["docs"], "chars": v["chars"]}
                for s, v in source_stats.items()
            },
        }, f, indent=2)
    log.info(f"Blend stats written to {stats_path}")


# ── Stage 5: Upload ────────────────────────────────────────────────────────────

def stage_upload(target: str) -> None:
    """
    Upload curated data to S3 under a versioned path.

    S3 path structure:
        {S3_PREFIX}/{target}/{date}/curated/
        e.g. slm/data/125m/2026-04-02/curated/train.jsonl

    Each run gets its own dated folder per target, so multiple runs
    never overwrite each other. Files within a single day's run are
    overwritten if re-uploaded.
    """
    log.info("=== Stage 5: Upload to S3 ===")
    bucket, prefix = get_bucket_and_prefix()

    date = datetime.now().strftime("%Y-%m-%d")
    dst_prefix = f"{target}/{date}/curated"
    s3_path = f"s3://{bucket}/{prefix}/{dst_prefix}/"

    log.info(f"Uploading to {s3_path}")

    upload_directory(
        src=CURATED_DIR,
        dst_prefix=dst_prefix,
        bucket=bucket,
        prefix=prefix,
        overwrite=True,
    )
    log.info(f"Upload complete → {s3_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

STAGES = ["download", "filter", "dedup", "blend", "upload", "all"]


def main():
    parser = argparse.ArgumentParser(
        description="SLM data curation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal run — validate the pipeline end-to-end quickly
  python curator/scripts/curate.py --target mini --mini

  # Full 125M run
  python curator/scripts/curate.py --target 125m

  # Full 350M run with 16 workers
  python curator/scripts/curate.py --target 350m --workers 16

  # Individual stages
  python curator/scripts/curate.py --target 125m --stage download
  python curator/scripts/curate.py --target 125m --stage dedup --workers 8
        """,
    )
    parser.add_argument(
        "--target",
        choices=list(TARGET_CONFIGS.keys()),
        default="125m",
        help="Model size target — controls data volume. Use 'mini' for pipeline validation.",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Pipeline stage to run. Default: all",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help=(
            "Minimal data volumes for pipeline validation. "
            "Caps Wikipedia at 5k docs, CodeSearchNet at 10k samples (python+js only). "
            "Use with --target mini."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel workers for dedup stage. "
            "Defaults to cpu_count // 2."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for blend stage shuffle. Default: 42",
    )
    args = parser.parse_args()

    if args.mini and args.target != "mini":
        log.warning(
            f"--mini flag is set but --target is '{args.target}'. "
            f"Consider using --target mini for a consistent minimal run."
        )

    log.info(
        f"SLM Curation Pipeline — "
        f"target={args.target}, stage={args.stage}, "
        f"mini={args.mini}, workers={args.workers or 'auto'}"
    )

    if args.stage in ("download", "all"):
        stage_download(args.target, mini=args.mini)

    if args.stage in ("filter", "all"):
        stage_filter()

    if args.stage in ("dedup", "all"):
        stage_dedup(workers=args.workers)

    if args.stage in ("blend", "all"):
        stage_blend(args.target, seed=args.seed)

    if args.stage in ("upload", "all"):
        stage_upload(args.target)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()