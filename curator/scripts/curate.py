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

Blend uses streaming chunk-based shuffle — peak RAM is O(chunk_size),
not O(corpus_size). Chunks are written sequentially, shuffled in memory,
then interleaved for good I/O throughput on any block storage.

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import orjson
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


# ── Worker count ───────────────────────────────────────────────────────────────
# Reserve 2 cores for the OS and datatrove's own threading.
# All parallel stages derive their worker count from this function so there
# is never a hardcoded CPU count anywhere in the pipeline.

def default_workers() -> int:
    """Return a sensible default worker count for the current machine."""
    cpu = os.cpu_count() or 4
    return max(1, cpu - 2)


# ── Target configurations ──────────────────────────────────────────────────────
# Token targets per model size — source mix stays constant at 55/25/20.
# mini target is for pipeline validation only — not for training.
#
# Targets are set to give comfortable headroom above Chinchilla compute-optimal:
#   125m — optimal ~2.5B, target 5B  (~2× optimal)
#   350m — optimal ~7B,   target 15B (~2× optimal)
#   1b   — optimal ~20B,  target 30B (~1.5× optimal)
#
# CC segment calibration (empirical, from 125m run):
#   ~6M tokens per segment after filtering and dedup
#   Target CC tokens = total_tokens × 0.55 (SOURCE_MIX)
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
        "cc_segments":   459,             # 5B × 0.55 / 6M ≈ 459 segments
        "cc_crawls":     ["CC-MAIN-2024-10"],
    },
    "350m": {
        "total_tokens":  15_000_000_000,  # 15B tokens (~2× Chinchilla optimal for 350m)
        "cc_segments":   916,             # 15B × 0.55 / 6M ≈ 916 segments, split across 2 crawls
        "cc_crawls":     ["CC-MAIN-2024-10", "CC-MAIN-2023-50"],
    },
    "1b": {
        "total_tokens":  30_000_000_000,  # 30B tokens (~1.5× Chinchilla optimal for 1b)
        "cc_segments":   1_833,           # 30B × 0.55 / 6M ≈ 1833 segments, split across 3 crawls
        "cc_crawls":     ["CC-MAIN-2024-10", "CC-MAIN-2023-50", "CC-MAIN-2023-40"],
    },
}

# Source mix — fraction of total tokens per source.
# Increased code to 20% (from 10%) for better coding ability.
# Reduced CC to 55% (from 70%) to offset the code increase while
# maintaining Wikipedia's high-quality signal at 25%.
SOURCE_MIX = {
    "common_crawl": 0.55,
    "wikipedia":    0.25,
    "code":         0.20,
}

# Mini run overrides — passed to source constructors when --mini is set
MINI_OVERRIDES = {
    "wiki_max_docs":    5_000,
    "code_max_docs":    10_000,
    "code_languages":   ["python"],
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
        {"text": "...", "id": "...", "source": ..., "url": ..., ...}

    Metadata keys are merged in after top-level keys so that top-level fields
    (text, id) are never silently overwritten by metadata contents.

    Plain JSONL records (not processed by datatrove) are returned unchanged.
    """
    if "metadata" in record and isinstance(record["metadata"], dict):
        # Start with all top-level fields except metadata itself,
        # then merge metadata on top — metadata must not overwrite text/id.
        flat = {k: v for k, v in record.items() if k != "metadata"}
        for k, v in record["metadata"].items():
            if k not in flat:  # never overwrite text, id, or other top-level fields
                flat[k] = v
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

    # CodeSearchNet — Python only
    log.info("Downloading CodeSearchNet (Python)...")
    code = CodeSearchNetSource(
        output_dir=RAW_DIR / "code",
        languages=MINI_OVERRIDES["code_languages"] if mini else ["python"],
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

# FIX 1: Module-level worker state for QualityFilter.
# Each subprocess initializes exactly one QualityFilter (and thus loads the
# fasttext model once). Without this, a new QualityFilter is constructed per
# shard, reloading the fasttext model on every call — O(shards) loads instead
# of O(workers) loads.
_worker_qf: QualityFilter | None = None


def _init_filter_worker() -> None:
    """
    Pool initializer: construct QualityFilter once per subprocess.

    Called once when each worker process starts. The fasttext model is
    loaded here and cached in _worker_qf for the lifetime of the worker.
    """
    global _worker_qf
    _worker_qf = QualityFilter()


def _filter_shard(args: tuple[Path, Path]) -> str:
    """
    Filter a single JSONL shard. Designed to run in a subprocess.

    Uses the module-level _worker_qf initialized by _init_filter_worker,
    so the fasttext model is loaded once per worker, not once per shard.

    Args:
        args: (shard_path, dst_dir) tuple.

    Returns:
        Human-readable report string from QualityFilter.
    """
    shard, dst_dir = args
    out_path = dst_dir / shard.name
    if out_path.exists():
        return f"skip:{shard.name}"

    qf = _worker_qf or QualityFilter()  # fallback if initializer wasn't used
    with open(shard, buffering=8 * 1024 * 1024) as fin, \
         open(out_path, "w", buffering=8 * 1024 * 1024) as fout:
        for line in fin:
            record = orjson.loads(line)
            kept, _ = qf.check(record)
            if kept:
                fout.write(orjson.dumps(record).decode() + "\n")

    return qf.report()


def stage_filter(workers: int | None = None) -> None:
    """
    Apply quality filters to all raw data in parallel.

    All shards across all sources are collected into a single work queue
    and processed by a process pool. Each shard is independent so there
    is no coordination overhead — parallelism is embarrassingly parallel.

    Worker count defaults to cpu_count - 2, leaving headroom for the OS
    and datatrove's threading. Override with --workers.
    """
    n_workers = workers or default_workers()
    log.info(f"=== Stage 2: Quality Filter ({n_workers} workers) ===")

    # Collect all shards across all sources into one flat work list.
    # Processing all sources together maximises worker utilisation —
    # workers don't sit idle waiting for one slow source to finish.
    all_work: list[tuple[Path, Path]] = []
    for source in ["wikipedia", "code", "common_crawl"]:
        src_dir = RAW_DIR / source
        dst_dir = FILTERED_DIR / source
        dst_dir.mkdir(parents=True, exist_ok=True)

        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"No shards found in {src_dir} — skipping")
            continue

        log.info(f"  {source}: {len(shards)} shards queued")
        all_work.extend((shard, dst_dir) for shard in shards)

    if not all_work:
        log.warning("No shards found across any source — skipping filter stage")
        return

    log.info(f"Filtering {len(all_work)} total shards with {n_workers} workers...")
    skipped = processed = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_filter_worker,   # FIX 1: load fasttext once per worker
    ) as executor:
        # chunksize=16 batches work assignments to reduce IPC overhead.
        # Increased from 4 — at 64 vCPUs with thousands of shards, larger
        # batches meaningfully reduce scheduler overhead.
        for report in executor.map(_filter_shard, all_work, chunksize=16):
            if report.startswith("skip:"):
                skipped += 1
            else:
                processed += 1
                log.debug(report)

    log.info(
        f"Filter complete — "
        f"processed: {processed}, skipped (already done): {skipped}"
    )


# ── Stage 3: Deduplicate ───────────────────────────────────────────────────────

def stage_dedup(workers: int | None = None) -> None:
    """
    Deduplicate filtered data using exact hash + datatrove MinHash LSH.

    Two stages per source:
        1. Exact dedup  — SHA-256 streaming pass, shared cross-source index.
        2. Fuzzy dedup  — datatrove 4-stage disk pipeline.
                          Peak RAM is O(shard_size), not O(corpus_size).

    Worker count defaults to cpu_count - 2.
    """
    n_workers = workers or default_workers()
    log.info(f"=== Stage 3: Deduplication (datatrove MinHash, {n_workers} workers) ===")

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

def _write_staging(args: tuple) -> tuple[str, int, int]:
    """
    Stream one source to a per-source staging file. Runs in a subprocess.

    Args:
        args: (source, src_dir, staging_path, source_char_target)

    Returns:
        (source, docs_written, chars_written)
    """
    source, src_dir, staging_path, source_char_target = args
    shards = sorted(Path(src_dir).glob("*.jsonl"))
    chars = docs = 0

    with open(staging_path, "w", buffering=8 * 1024 * 1024) as fout:
        for shard in shards:
            if chars >= source_char_target:
                break
            with open(shard, buffering=8 * 1024 * 1024) as fin:
                for line in fin:
                    record = flatten_datatrove_record(orjson.loads(line))
                    chars += len(record.get("text", ""))
                    fout.write(orjson.dumps(record).decode() + "\n")
                    docs += 1
                    if chars >= source_char_target:
                        break

    return source, docs, chars


def _shuffle_chunked(
    merged_path: Path,
    output_path: Path,
    rng: random.Random,
    chunk_lines: int = 500_000,
) -> int:
    """
    Shuffle a large JSONL file using a chunked approach.

    Replaces the seek-per-line approach which causes millions of random
    disk seeks — worst-case I/O pattern for any block storage device.

    Instead:
        Pass 1 — read sequentially in chunks of chunk_lines, shuffle each
                 chunk in memory, write to a numbered chunk file.
        Pass 2 — shuffle the chunk file order, then concatenate sequentially.

    Both passes are sequential reads/writes — optimal for block storage.
    Peak RAM = chunk_lines × avg_line_size (tunable via chunk_lines param).

    Args:
        merged_path:  Input JSONL file to shuffle.
        output_path:  Output shuffled JSONL file.
        rng:          Seeded random instance for reproducibility.
        chunk_lines:  Lines per chunk. Default 500k — ~400MB at avg 800 bytes/line.

    Returns:
        Total number of lines written.
    """
    chunk_dir = merged_path.parent / "shuffle_chunks"
    chunk_dir.mkdir(exist_ok=True)
    chunk_paths: list[Path] = []
    total_lines = 0

    # Pass 1: write shuffled chunks sequentially
    log.info("Shuffle pass 1/2: writing shuffled chunks...")
    buf: list[str] = []
    chunk_idx = 0

    with open(merged_path, buffering=8 * 1024 * 1024) as fin:
        for line in fin:
            buf.append(line)
            total_lines += 1
            if len(buf) >= chunk_lines:
                rng.shuffle(buf)
                p = chunk_dir / f"chunk_{chunk_idx:06d}.jsonl"
                with open(p, "w", buffering=8 * 1024 * 1024) as fout:
                    fout.writelines(buf)
                chunk_paths.append(p)
                buf = []
                chunk_idx += 1

    # flush remaining
    if buf:
        rng.shuffle(buf)
        p = chunk_dir / f"chunk_{chunk_idx:06d}.jsonl"
        with open(p, "w", buffering=8 * 1024 * 1024) as fout:
            fout.writelines(buf)
        chunk_paths.append(p)

    log.info(f"  Wrote {len(chunk_paths)} chunks ({total_lines:,} lines total)")

    # Pass 2: shuffle chunk order, concatenate sequentially
    log.info("Shuffle pass 2/2: interleaving chunks in random order...")
    rng.shuffle(chunk_paths)

    with open(output_path, "wb") as fout:
        for cp in chunk_paths:
            with open(cp, "rb") as fin:
                while True:
                    block = fin.read(8 * 1024 * 1024)
                    if not block:
                        break
                    fout.write(block)
            cp.unlink()

    chunk_dir.rmdir()
    return total_lines


def stage_blend(target: str, seed: int = 42, workers: int | None = None) -> None:
    """
    Blend sources to the target token ratio and write final train.jsonl.

    Pass 1 — parallel: each source streams to its own staging file
             (3 workers, one per source — they are I/O independent).
    Pass 2 — sequential: merge staging files into one.
    Pass 3 — chunked shuffle: replaces seek-per-line with sequential
             chunk reads/writes for much better block storage throughput.

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

    # ── Pass 1: parallel staging ───────────────────────────────────────────────
    # Three sources are written in parallel — each is I/O independent.
    # We use min(3, cpu_count-2) workers since there are only 3 sources.
    n_blend_workers = min(3, workers or default_workers())

    work = []
    for source, src_dir in source_dirs.items():
        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"  {source}: no deduped shards found — skipping")
            continue
        staging = CURATED_DIR / f"blend_{source}.jsonl"
        # Resume: if staging file already exists, skip re-writing it.
        if staging.exists():
            log.info(f"  {source}: staging file already exists — skipping")
            continue
        work.append((source, str(src_dir), str(staging), target_chars[source]))

    staging_paths: dict[str, Path] = {
        source: CURATED_DIR / f"blend_{source}.jsonl"
        for source in source_dirs
        if (CURATED_DIR / f"blend_{source}.jsonl").exists()
    }
    source_stats: dict[str, dict] = {}

    # FIX: For staging files that already exist (resume path), populate
    # source_stats by counting their contents so blend_stats.json is complete
    # and total_chars is accurate regardless of whether this is a fresh run
    # or a resume.
    for source, staging in staging_paths.items():
        docs = chars = 0
        with open(staging, buffering=8 * 1024 * 1024) as fin:
            for line in fin:
                docs += 1
                try:
                    record = orjson.loads(line)
                    chars += len(record.get("text", ""))
                except Exception:
                    pass
        source_stats[source] = {"docs": docs, "chars": chars}
        log.info(
            f"  {source} (existing staging): {docs:,} docs, "
            f"{chars / 1e9:.3f}B chars, ~{chars // 4 / 1e6:.1f}M tokens"
        )

    if work:
        log.info(
            f"Pass 1/3: streaming {len(work)} sources to staging files "
            f"({n_blend_workers} workers)..."
        )
        with ProcessPoolExecutor(max_workers=n_blend_workers) as executor:
            futures = {executor.submit(_write_staging, w): w[0] for w in work}
            for future in as_completed(futures):
                source, docs, chars = future.result()
                staging_paths[source] = CURATED_DIR / f"blend_{source}.jsonl"
                # Update stats with freshly written values (overrides the
                # pre-scan above which would have seen an empty/missing file).
                source_stats[source] = {"docs": docs, "chars": chars}
                log.info(
                    f"  {source}: {docs:,} docs, "
                    f"{chars / 1e9:.3f}B chars, "
                    f"~{chars // 4 / 1e6:.1f}M tokens"
                )
    else:
        log.info("Pass 1/3: all staging files already exist — skipping")

    # ── Pass 2: merge staging files ───────────────────────────────────────────
    merged_path = CURATED_DIR / "blend_merged.jsonl"

    # FIX 4: count docs correctly — reset counter per source, count lines
    # by reading the staging file rather than counting newlines in raw blocks
    # (which double-counts across block boundaries).
    total_docs = 0
    log.info("Pass 2/3: merging staging files...")
    with open(merged_path, "wb") as fout:
        for source, staging in staging_paths.items():
            source_docs = 0
            with open(staging, "rb") as fin:
                while True:
                    block = fin.read(8 * 1024 * 1024)
                    if not block:
                        break
                    source_docs += block.count(b"\n")
                    fout.write(block)
            total_docs += source_docs
            staging.unlink()
            log.info(f"  {source}: {source_docs:,} docs merged")

    log.info(f"  Total merged: {total_docs:,} documents")

    # ── Pass 3: chunked shuffle ────────────────────────────────────────────────
    log.info("Pass 3/3: shuffling (chunked)...")
    total_lines = _shuffle_chunked(merged_path, output_path, rng)
    merged_path.unlink()

    total_chars = sum(s["chars"] for s in source_stats.values())
    log.info(
        f"Blend complete — "
        f"{total_lines:,} documents, "
        f"~{total_chars // 4 / 1e9:.2f}B tokens"
    )

    # ── Write blend stats ──────────────────────────────────────────────────────
    stats_path = CURATED_DIR / "blend_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "target": target,
            "target_tokens": total_tokens,
            "total_documents": total_lines,
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

  # Full 1B run
  python curator/scripts/curate.py --target 1b

  # Individual stages
  python curator/scripts/curate.py --target 125m --stage download
  python curator/scripts/curate.py --target 125m --stage filter
  python curator/scripts/curate.py --target 125m --stage dedup
  python curator/scripts/curate.py --target 125m --stage blend
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
            "Caps Wikipedia at 5k docs, CodeSearchNet at 10k samples (python only). "
            "Use with --target mini."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel workers for filter, dedup, and blend stages. "
            "Defaults to cpu_count - 2."
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

    n_workers = args.workers or default_workers()
    log.info(
        f"SLM Curation Pipeline — "
        f"target={args.target}, stage={args.stage}, "
        f"mini={args.mini}, workers={n_workers} (cpu_count={os.cpu_count()})"
    )

    if args.stage in ("download", "all"):
        stage_download(args.target, mini=args.mini)

    if args.stage in ("filter", "all"):
        stage_filter(workers=n_workers)

    if args.stage in ("dedup", "all"):
        stage_dedup(workers=n_workers)

    if args.stage in ("blend", "all"):
        stage_blend(args.target, seed=args.seed, workers=n_workers)

    if args.stage in ("upload", "all"):
        stage_upload(args.target)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()