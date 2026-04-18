"""
curator/scripts/curate.py
--------------------------
Main data curation pipeline.

Orchestrates all data sources, quality filters, and deduplication into a
single configurable pipeline. Produces clean JSONL files ready for
tokenizer training and model pretraining.

Pipeline:
    1. Download sources (Wikipedia, CodeSearchNet, Common Crawl)
    2. Apply quality filters
    3. Deduplicate (exact SHA-256 + datatrove disk-based MinHash LSH)
    4. Blend sources to target token ratios
    5. Upload to S3

Blend stage improvements:
    - Pass 1 (parallel): stream each source to a staging file
    - Pass 2 (collapsed): read staging files directly into shuffle chunks
                          — no separate merge pass
    - Pass 3: if total size fits in RAM budget, one-shot in-memory shuffle;
              otherwise chunked disk shuffle

Usage:
    # Full pipeline
    python curator/scripts/curate.py --target 125m

    # Minimal run — validates pipeline end-to-end
    python curator/scripts/curate.py --target mini --mini

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
from curator.scripts.upload_s3 import (
    upload_directory, download_prefix, get_bucket_and_prefix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Worker count ───────────────────────────────────────────────────────────────

def default_workers() -> int:
    cpu = os.cpu_count() or 4
    return max(1, cpu - 2)


# ── Target configurations ──────────────────────────────────────────────────────

TARGET_CONFIGS = {
    "mini": {
        "total_tokens":  1_000_000,
        "cc_segments":   2,
        "cc_crawls":     ["CC-MAIN-2024-10"],
    },
    "125m": {
        "total_tokens":  5_000_000_000,
        "cc_segments":   459,
        "cc_crawls":     ["CC-MAIN-2024-10"],
    },
    "350m": {
        "total_tokens":  15_000_000_000,
        "cc_segments":   916,
        "cc_crawls":     ["CC-MAIN-2024-10", "CC-MAIN-2023-50"],
    },
    "1b": {
        "total_tokens":  30_000_000_000,
        "cc_segments":   1_833,
        "cc_crawls":     ["CC-MAIN-2024-10", "CC-MAIN-2023-50", "CC-MAIN-2023-40"],
    },
}

SOURCE_MIX = {
    "common_crawl": 0.55,
    "wikipedia":    0.25,
    "code":         0.20,
}

MINI_OVERRIDES = {
    "wiki_max_docs":    5_000,
    "code_max_docs":    10_000,
    "code_languages":   ["python"],
}

DATA_DIR     = Path(os.environ.get("DATA_DIR", "data"))
RAW_DIR      = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
CURATED_DIR  = DATA_DIR / "curated"

# In-memory shuffle fast path — if the merged corpus fits in this many GB,
# shuffle in one go instead of the two-pass chunked algorithm.
# Set via env SHUFFLE_RAM_BUDGET_GB; default sized for a 256GB machine.
SHUFFLE_RAM_BUDGET_GB = float(os.environ.get("SHUFFLE_RAM_BUDGET_GB", "64"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def flatten_datatrove_record(record: dict) -> dict:
    """
    Flatten datatrove's document format back to a flat dict.

    datatrove wraps as {"text": ..., "id": ..., "metadata": {...}}.
    We flatten so metadata keys live at the top level, without ever
    overwriting top-level fields (text, id).

    Mutates in place to avoid allocating two dicts per record — at 1b
    scale this runs ~100M+ times in the blend stage.
    """
    md = record.pop("metadata", None)
    if isinstance(md, dict):
        for k, v in md.items():
            if k not in record:
                record[k] = v
        record.pop("file_path", None)
    return record


# ── Stage 1: Download ──────────────────────────────────────────────────────────

def stage_download(target: str, mini: bool = False, workers: int | None = None) -> None:
    cfg = TARGET_CONFIGS[target]
    n_workers = workers or default_workers()
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
        languages=MINI_OVERRIDES["code_languages"] if mini else ["python"],
        max_docs=MINI_OVERRIDES["code_max_docs"] if mini else None,
    )
    code.download()
    log.info(f"CodeSearchNet stats: {code.stats()}")

    # Common Crawl — parallel download + extraction.
    # Extraction workers eat the CPU budget; download threads are lightweight.
    log.info("Downloading Common Crawl...")
    cc = CommonCrawlSource(
        output_dir=RAW_DIR / "common_crawl",
        crawls=cfg["cc_crawls"],
        max_segments=cfg["cc_segments"],
        download_workers=16,
        extract_workers=max(1, n_workers - 4),  # leave a few cores for download + main
    )
    cc.download()
    log.info(f"Common Crawl stats: {cc.stats()}")


# ── Stage 2: Filter ────────────────────────────────────────────────────────────

_worker_qf: QualityFilter | None = None


def _init_filter_worker() -> None:
    """Pool initializer: construct QualityFilter once per subprocess."""
    global _worker_qf
    _worker_qf = QualityFilter()


def _filter_shard(args: tuple[Path, Path]) -> str:
    """Filter a single JSONL shard. Runs in a subprocess."""
    shard, dst_dir = args
    out_path = dst_dir / shard.name
    if out_path.exists():
        return f"skip:{shard.name}"

    qf = _worker_qf or QualityFilter()
    with open(shard, "rb", buffering=8 * 1024 * 1024) as fin, \
         open(out_path, "wb", buffering=8 * 1024 * 1024) as fout:
        for line in fin:
            try:
                record = orjson.loads(line)
            except Exception:
                continue
            kept, _ = qf.check(record)
            if kept:
                fout.write(orjson.dumps(record))
                fout.write(b"\n")

    return qf.report()


def stage_filter(workers: int | None = None) -> None:
    """Apply quality filters to all raw data in parallel."""
    n_workers = workers or default_workers()
    log.info(f"=== Stage 2: Quality Filter ({n_workers} workers) ===")

    all_work: list[tuple[Path, Path]] = []
    for source in ["wikipedia", "code", "common_crawl"]:
        src_dir = RAW_DIR / source
        dst_dir = FILTERED_DIR / source
        dst_dir.mkdir(parents=True, exist_ok=True)

        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"No shards in {src_dir} — skipping")
            continue

        log.info(f"  {source}: {len(shards)} shards queued")
        all_work.extend((shard, dst_dir) for shard in shards)

    if not all_work:
        log.warning("No shards found across any source — skipping filter")
        return

    # Sort largest-first so stragglers don't tail the run
    all_work.sort(key=lambda p: p[0].stat().st_size, reverse=True)

    log.info(f"Filtering {len(all_work)} shards with {n_workers} workers...")
    skipped = processed = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_filter_worker,
    ) as executor:
        for report in executor.map(_filter_shard, all_work, chunksize=16):
            if report.startswith("skip:"):
                skipped += 1
            else:
                processed += 1
                log.debug(report)

    log.info(
        f"Filter complete — processed: {processed}, skipped: {skipped}"
    )


# ── Stage 3: Deduplicate ───────────────────────────────────────────────────────

def stage_dedup(workers: int | None = None) -> None:
    n_workers = workers or default_workers()
    log.info(f"=== Stage 3: Deduplication ({n_workers} workers) ===")

    working_dir = DATA_DIR / "dedup_scratch"
    dedup = Deduplicator(working_dir=working_dir, workers=n_workers)

    for source in ["wikipedia", "code", "common_crawl"]:
        src_dir = FILTERED_DIR / source
        dst_dir = FILTERED_DIR / f"{source}_deduped"

        shards = list(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"No filtered shards in {src_dir} — skipping")
            continue

        if dst_dir.exists() and list(dst_dir.glob("*.jsonl")):
            log.info(f"  {source}: already deduped — skipping")
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

    Reads deduped shards, flattens datatrove records, and writes to a
    staging JSONL until the per-source character target is hit.

    Returns:
        (source, docs_written, chars_written)
    """
    source, src_dir, staging_path, source_char_target = args
    shards = sorted(Path(src_dir).glob("*.jsonl"))
    chars = docs = 0

    with open(staging_path, "wb", buffering=8 * 1024 * 1024) as fout:
        for shard in shards:
            if chars >= source_char_target:
                break
            with open(shard, "rb", buffering=8 * 1024 * 1024) as fin:
                for line in fin:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    record = flatten_datatrove_record(record)
                    text = record.get("text", "")
                    chars += len(text)
                    fout.write(orjson.dumps(record))
                    fout.write(b"\n")
                    docs += 1
                    if chars >= source_char_target:
                        break

    return source, docs, chars


def _shuffle_in_memory(
    staging_paths: dict[str, Path],
    output_path: Path,
    rng: random.Random,
) -> int:
    """
    Fast-path shuffle: read everything into RAM, shuffle once, write once.

    Used when the total staging size fits in SHUFFLE_RAM_BUDGET_GB.
    At 125m (~20 GB) this runs in seconds on a 256GB machine.
    """
    log.info("Shuffle: reading all staging data into memory...")
    lines: list[bytes] = []
    for source, staging in staging_paths.items():
        with open(staging, "rb") as f:
            for line in f:
                lines.append(line)
        staging.unlink()

    log.info(f"  Loaded {len(lines):,} lines — shuffling...")
    rng.shuffle(lines)

    log.info(f"  Writing {len(lines):,} lines to {output_path}...")
    with open(output_path, "wb") as fout:
        fout.writelines(lines)
    return len(lines)


def _shuffle_chunked_from_sources(
    staging_paths: dict[str, Path],
    output_path: Path,
    rng: random.Random,
    chunk_lines: int = 500_000,
) -> int:
    """
    Chunked shuffle collapsing the old 2-pass merge+shuffle into one.

    Pass 1 — read sequentially from all staging files into chunks,
             shuffle each chunk, write to disk. This replaces both the
             old merge pass AND pass 1 of the old shuffle — one full
             write of the corpus instead of two.

    Pass 2 — shuffle chunk order, concatenate sequentially.

    Both passes are sequential reads/writes — optimal for block storage.
    Peak RAM = chunk_lines × avg_line_size.
    """
    chunk_dir = output_path.parent / "shuffle_chunks"
    chunk_dir.mkdir(exist_ok=True)
    chunk_paths: list[Path] = []
    total_lines = 0

    log.info("Shuffle pass 1/2: reading staging files into shuffled chunks...")
    buf: list[bytes] = []
    chunk_idx = 0

    def _flush_chunk():
        nonlocal chunk_idx
        if not buf:
            return
        rng.shuffle(buf)
        p = chunk_dir / f"chunk_{chunk_idx:06d}.jsonl"
        with open(p, "wb", buffering=8 * 1024 * 1024) as fout:
            fout.writelines(buf)
        chunk_paths.append(p)
        buf.clear()
        chunk_idx += 1

    for source, staging in staging_paths.items():
        with open(staging, "rb", buffering=8 * 1024 * 1024) as fin:
            for line in fin:
                buf.append(line)
                total_lines += 1
                if len(buf) >= chunk_lines:
                    _flush_chunk()
        # Free staging once fully read
        staging.unlink()

    _flush_chunk()  # flush any remainder

    log.info(f"  Wrote {len(chunk_paths)} chunks ({total_lines:,} lines total)")

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

    try:
        chunk_dir.rmdir()
    except OSError:
        pass
    return total_lines


def stage_blend(target: str, seed: int = 42, workers: int | None = None) -> None:
    """
    Blend sources to the target token ratio and write final train.jsonl.

    Pass 1 — parallel: each source streams to its own staging file.
    Pass 2 — single-pass shuffle:
             - fits-in-RAM path for small/medium runs
             - chunked disk shuffle for large runs
             Replaces the old (merge → chunked shuffle) two-pass — one
             full write of the corpus instead of two.

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

    target_chars = {
        source: int(total_tokens * fraction * 4)
        for source, fraction in SOURCE_MIX.items()
    }

    # ── Pass 1: parallel staging ───────────────────────────────────────────────
    n_blend_workers = min(3, workers or default_workers())

    work: list[tuple] = []
    staging_paths: dict[str, Path] = {}
    source_stats: dict[str, dict] = {}

    for source, src_dir in source_dirs.items():
        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"  {source}: no deduped shards — skipping")
            continue
        staging = CURATED_DIR / f"blend_{source}.jsonl"
        if staging.exists():
            log.info(f"  {source}: staging file already exists — skipping re-stage")
            # Populate stats from existing staging so blend_stats.json is complete
            docs = chars = 0
            with open(staging, "rb") as fin:
                for line in fin:
                    docs += 1
                    try:
                        record = orjson.loads(line)
                        chars += len(record.get("text", ""))
                    except Exception:
                        pass
            source_stats[source] = {"docs": docs, "chars": chars}
            staging_paths[source] = staging
            log.info(
                f"  {source} (existing): {docs:,} docs, "
                f"{chars / 1e9:.3f}B chars, ~{chars // 4 / 1e6:.1f}M tokens"
            )
            continue
        work.append((source, str(src_dir), str(staging), target_chars[source]))

    if work:
        log.info(
            f"Pass 1/2: staging {len(work)} sources "
            f"({n_blend_workers} workers)..."
        )
        with ProcessPoolExecutor(max_workers=n_blend_workers) as executor:
            futures = {executor.submit(_write_staging, w): w[0] for w in work}
            for future in as_completed(futures):
                source, docs, chars = future.result()
                staging_paths[source] = CURATED_DIR / f"blend_{source}.jsonl"
                source_stats[source] = {"docs": docs, "chars": chars}
                log.info(
                    f"  {source}: {docs:,} docs, "
                    f"{chars / 1e9:.3f}B chars, "
                    f"~{chars // 4 / 1e6:.1f}M tokens"
                )
    else:
        log.info("Pass 1/2: all staging files already exist — skipping")

    if not staging_paths:
        log.error("No staging files written — nothing to blend")
        return

    # ── Pass 2: collapsed shuffle ──────────────────────────────────────────────
    total_staging_bytes = sum(p.stat().st_size for p in staging_paths.values())
    total_staging_gb = total_staging_bytes / 1e9
    log.info(
        f"Pass 2/2: shuffling — staging size {total_staging_gb:.2f} GB, "
        f"RAM budget {SHUFFLE_RAM_BUDGET_GB:.1f} GB"
    )

    if total_staging_gb < SHUFFLE_RAM_BUDGET_GB:
        total_lines = _shuffle_in_memory(staging_paths, output_path, rng)
    else:
        log.info("  Corpus exceeds RAM budget — using chunked disk shuffle")
        total_lines = _shuffle_chunked_from_sources(
            staging_paths, output_path, rng,
        )

    total_chars = sum(s["chars"] for s in source_stats.values())
    log.info(
        f"Blend complete — {total_lines:,} documents, "
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
    log.info(f"Blend stats → {stats_path}")


# ── Stage 5: Upload ────────────────────────────────────────────────────────────

def stage_upload(target: str) -> None:
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
    )
    parser.add_argument(
        "--target",
        choices=list(TARGET_CONFIGS.keys()),
        default="125m",
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers for filter/dedup/blend. Default: cpu_count - 2.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mini and args.target != "mini":
        log.warning(
            f"--mini set but --target is '{args.target}'. "
            f"Consider --target mini."
        )

    n_workers = args.workers or default_workers()
    log.info(
        f"SLM Curation — "
        f"target={args.target}, stage={args.stage}, "
        f"mini={args.mini}, workers={n_workers} (cpu_count={os.cpu_count()})"
    )

    if args.stage in ("download", "all"):
        stage_download(args.target, mini=args.mini, workers=n_workers)
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