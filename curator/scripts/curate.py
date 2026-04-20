"""
curator/scripts/curate.py
--------------------------
Main data curation pipeline.

Orchestrates 10 data sources through quality filtering, deduplication,
blending, and upload. Produces train.jsonl + val.jsonl ready for tokenizer
training and model pretraining. The val split is sampled uniformly from
the shuffled blend output, so it represents the same distribution as train.

Pipeline:
    1. Download sources
    2. Apply quality filters
    3. Deduplicate (exact SHA-256 + datatrove disk-based MinHash LSH)
    4. Blend sources to target token ratios (with cap-and-redistribute)
    5. Upload to S3

Data mix (scale-invariant):
    common_crawl   10%    unlimited (time-bound)
    fineweb        47.5%  15T supply (also overflow sink)
    wikipedia      10%    ~3.7B supply
    pg19           2.5%   ~2.9B supply
    pes2o          5%     ~42B supply
    open_web_math  10%    ~14.7B supply
    stackexchange  5%     ~15B supply
    code           10%    (see code sub-mix below)

Code sub-mix (percentages of the 10% code share):
    stack_v2       50%    (capped — bulk code, 4 langs)
    codesearchnet  35%    (curated function-level, 6 langs)
    stack_smol     10%    (raw code, 30 langs)
    jupyter        4%     (notebook cells)
    conala         1%     (NL-to-code pairs)

Cap-and-redistribute:
    Finite sources (Wikipedia, pg19, etc.) may supply less than their
    character budget allows at large scales. Each source writes up to
    its budget or until its supply is exhausted, whichever comes first.
    The total shortfall is added to FineWeb's budget at the end, which
    acts as the overflow sink. FineWeb has effectively unlimited supply
    (15T tokens) so this always closes the gap.

Blend stage improvements:
    - Pass 1 (parallel): stream each source to a staging file, recording
                         chars written vs target.
    - Pass 2 (sequential): write FineWeb overflow to cover deficit.
    - Pass 3 (shuffled write): if total size fits in RAM budget, one-shot
                               in-memory shuffle; otherwise chunked disk
                               shuffle.

Usage:
    python curator/scripts/curate.py --target 125m
    python curator/scripts/curate.py --target mini --mini
    python curator/scripts/curate.py --target 125m --stage download
"""

import argparse
import json
import logging
import math
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

from curator.constants import CHARS_PER_TOKEN, CC_CHARS_PER_SEGMENT
from curator.filters.dedup import Deduplicator
from curator.filters.quality import QualityFilter

from curator.sources.common_crawl import CommonCrawlSource
from curator.sources.fineweb import FineWebSource
from curator.sources.wikipedia import WikipediaSource
from curator.sources.pg19 import PG19Source
from curator.sources.pes2o import PeS2oSource
from curator.sources.open_web_math import OpenWebMathSource
from curator.sources.stackexchange import StackExchangeSource
from curator.sources.code_search_net import CodeSearchNetSource
from curator.sources.the_stack_smol import StackSmolSource
from curator.sources.the_stack_v2 import StackV2Source
from curator.sources.jupyter import JupyterSource
from curator.sources.conala import ConalaSource

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
#
# cc_segments is computed at runtime from total_tokens × CC_share × CHARS_PER_TOKEN
# / CC_CHARS_PER_SEGMENT — see compute_cc_segments() below. The empirical
# chars-per-segment value (17M) comes from the 125m run; the previous hardcoded
# values assumed 24M which caused the 125m run to undershoot its 5B target.

TARGET_CONFIGS = {
    "mini": {
        "total_tokens":  1_000_000,
        "cc_crawls":     ["CC-MAIN-2024-10"],
    },
    "125m": {
        "total_tokens":  5_000_000_000,
        "cc_crawls":     ["CC-MAIN-2024-10"],
    },
    "350m": {
        "total_tokens":  15_000_000_000,
        "cc_crawls":     ["CC-MAIN-2024-10", "CC-MAIN-2023-50"],
    },
    "1b": {
        "total_tokens":  30_000_000_000,
        "cc_crawls":     ["CC-MAIN-2024-10", "CC-MAIN-2023-50", "CC-MAIN-2023-40"],
    },
}

# Top-level source mix — scale-invariant percentages. All sizes use these.
SOURCE_MIX: dict[str, float] = {
    "common_crawl":   0.10,
    "fineweb":        0.475,
    "wikipedia":      0.10,
    "pg19":           0.025,
    "pes2o":          0.05,
    "open_web_math":  0.10,
    "stackexchange":  0.05,
    "code":           0.10,  # dispatched across CODE_SUB_MIX
}

# Code sub-mix — percentages of the 10% code share (not of total).
# stack_v2 capped at 50% so raw bulk code doesn't dominate the curated
# sources.
CODE_SUB_MIX: dict[str, float] = {
    "stack_v2":       0.50,
    "codesearchnet":  0.35,
    "stack_smol":     0.10,
    "jupyter":        0.04,
    "conala":         0.01,
}

# FineWeb is the overflow sink. When finite sources (Wikipedia, pg19, etc.)
# can't fill their character budget, the deficit is added to FineWeb's
# budget at the end of staging.
OVERFLOW_SINK = "fineweb"

# All non-code source names for iteration. The "code" entry in SOURCE_MIX
# is a share; the actual source names for iteration come from CODE_SUB_MIX.
NON_CODE_SOURCES: list[str] = [s for s in SOURCE_MIX if s != "code"]
CODE_SOURCES: list[str] = list(CODE_SUB_MIX.keys())
ALL_SOURCES: list[str] = NON_CODE_SOURCES + CODE_SOURCES

# ── Mini overrides ─────────────────────────────────────────────────────────────
#
# Mini run exercises every source at small scale to validate the pipeline
# end-to-end before committing to a full run. Caps are rough per-source
# proportions of a 1M-token total.

MINI_OVERRIDES: dict[str, int] = {
    "common_crawl":  2,        # segments, not docs
    "fineweb":       10_000,
    "wikipedia":     5_000,
    "pg19":          50,
    "pes2o":         2_000,
    "open_web_math": 3_000,
    "stackexchange": 2_000,
    "codesearchnet": 5_000,
    "stack_smol":    2_000,
    "stack_v2":      3_000,
    "jupyter":       500,
    "conala":        500,
}

# Data directories
DATA_DIR     = Path(os.environ.get("DATA_DIR", "data"))
RAW_DIR      = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
CURATED_DIR  = DATA_DIR / "curated"

# In-memory shuffle fast path — if the merged corpus fits in this many GB,
# shuffle in one go instead of the chunked disk algorithm.
#
# Default is intentionally conservative: Python list objects carry ~5× the
# on-disk size in RAM due to per-object overhead. A 12 GB disk-size staging
# set occupies roughly 60 GB of process memory once loaded. On a 64 GB
# instance this still fits; on a 256 GB instance we have lots of headroom.
SHUFFLE_RAM_BUDGET_GB = float(os.environ.get("SHUFFLE_RAM_BUDGET_GB", "12"))

# Fraction of blended documents to hold out for validation. The split happens
# at the end of the blend stage (after shuffle), so val is a uniform random
# sample from the same distribution as train. Fixed at 0.5% — at 125m (~5M
# docs) this gives ~25k val docs, plenty for stable perplexity measurement;
# at 1b (~30M docs) it gives ~150k, which is generous. Per-target override
# available via the TARGET_CONFIGS val_fraction key.
VAL_FRACTION = 0.005


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_cc_segments(total_tokens: int) -> int:
    """
    Segments of Common Crawl needed to hit CC's character share.

    Computed from: total_tokens × SOURCE_MIX[cc] × CHARS_PER_TOKEN bytes of
    text, divided by CC_CHARS_PER_SEGMENT bytes produced per segment after
    trafilatura + language filtering.
    """
    cc_share = SOURCE_MIX["common_crawl"]
    target_chars = int(total_tokens * cc_share * CHARS_PER_TOKEN)
    return max(1, math.ceil(target_chars / CC_CHARS_PER_SEGMENT))


def compute_source_char_targets(total_tokens: int) -> dict[str, int]:
    """
    Compute the character budget for each source from the target tokens.

    Returns a dict mapping each source name (all 12: 7 non-code + 5 code)
    to its target character count. Code sources get their share of the
    10% code budget according to CODE_SUB_MIX.
    """
    targets: dict[str, int] = {}
    for source, share in SOURCE_MIX.items():
        if source == "code":
            continue
        targets[source] = int(total_tokens * share * CHARS_PER_TOKEN)

    code_total_chars = int(total_tokens * SOURCE_MIX["code"] * CHARS_PER_TOKEN)
    for code_source, sub_share in CODE_SUB_MIX.items():
        targets[code_source] = int(code_total_chars * sub_share)

    return targets


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

def _build_source(
    name: str,
    mini: bool,
    target: str,
    workers: int,
) -> object:
    """Construct a source instance with mini caps applied when mini=True."""
    raw_dir = RAW_DIR / name
    cap = MINI_OVERRIDES.get(name) if mini else None

    # CC has a different 'cap' semantics: max_segments, and needs crawls + workers.
    if name == "common_crawl":
        cfg = TARGET_CONFIGS[target]
        if mini:
            max_segments = cap
        else:
            max_segments = compute_cc_segments(cfg["total_tokens"])
        return CommonCrawlSource(
            output_dir=raw_dir,
            crawls=cfg["cc_crawls"],
            max_segments=max_segments,
            download_workers=16,
            extract_workers=max(1, workers - 4),
        )

    if name == "fineweb":
        return FineWebSource(output_dir=raw_dir, max_docs=cap)
    if name == "wikipedia":
        return WikipediaSource(output_dir=raw_dir, max_docs=cap)
    if name == "pg19":
        return PG19Source(output_dir=raw_dir, max_docs=cap)
    if name == "pes2o":
        return PeS2oSource(output_dir=raw_dir, max_docs=cap)
    if name == "open_web_math":
        return OpenWebMathSource(output_dir=raw_dir, max_docs=cap)
    if name == "stackexchange":
        return StackExchangeSource(output_dir=raw_dir, max_docs=cap)
    if name == "codesearchnet":
        return CodeSearchNetSource(output_dir=raw_dir, max_docs=cap)
    if name == "stack_smol":
        return StackSmolSource(output_dir=raw_dir, max_docs=cap)
    if name == "stack_v2":
        return StackV2Source(output_dir=raw_dir, max_docs=cap)
    if name == "jupyter":
        return JupyterSource(output_dir=raw_dir, max_docs=cap)
    if name == "conala":
        return ConalaSource(output_dir=raw_dir, max_docs=cap)

    raise ValueError(f"Unknown source: {name}")


def stage_download(target: str, mini: bool = False, workers: int | None = None) -> None:
    """Download every source in SOURCE_MIX + CODE_SUB_MIX."""
    n_workers = workers or default_workers()
    log.info(f"=== Stage 1: Download (target={target}, mini={mini}) ===")

    if not mini:
        cc_segments = compute_cc_segments(TARGET_CONFIGS[target]["total_tokens"])
        log.info(
            f"Common Crawl: computed {cc_segments} segments from "
            f"{TARGET_CONFIGS[target]['total_tokens']:,} tokens × "
            f"{SOURCE_MIX['common_crawl']:.2%} × {CHARS_PER_TOKEN} chars/tok "
            f"÷ {CC_CHARS_PER_SEGMENT:,} chars/segment"
        )

    for name in ALL_SOURCES:
        log.info(f"Downloading {name}...")
        source = _build_source(name, mini=mini, target=target, workers=n_workers)
        source.download()
        try:
            log.info(f"{name} stats: {source.stats()}")
        except Exception as e:
            log.warning(f"{name}: stats() failed — {e}")


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
    for source in ALL_SOURCES:
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

    log.info(f"Filter complete — processed: {processed}, skipped: {skipped}")


# ── Stage 3: Deduplicate ───────────────────────────────────────────────────────

def stage_dedup(workers: int | None = None) -> None:
    n_workers = workers or default_workers()
    log.info(f"=== Stage 3: Deduplication ({n_workers} workers) ===")

    working_dir = DATA_DIR / "dedup_scratch"
    dedup = Deduplicator(working_dir=working_dir, workers=n_workers)

    for source in ALL_SOURCES:
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
    Stream one source's deduped shards to a per-source staging file.
    Runs in a subprocess.

    Writes until the source's character target is hit OR its supply is
    exhausted (whichever comes first). Returns how much was actually
    written so the main process can compute deficits for FineWeb overflow.

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


def _append_overflow(
    args: tuple,
) -> tuple[int, int]:
    """
    Append FineWeb docs to its staging file to cover the total deficit.
    Runs in a subprocess.

    Reads FineWeb deduped shards from where the initial staging pass left
    off (determined by counting chars already in the staging file) and
    appends until `overflow_chars` additional chars have been written.

    Returns: (docs_appended, chars_appended)
    """
    src_dir, staging_path, overflow_chars = args
    if overflow_chars <= 0:
        return 0, 0

    # Count chars already in staging so we can skip those shard contents.
    # Staging files are written in shard-sorted order, so counting chars
    # tells us where to resume.
    already_chars = 0
    with open(staging_path, "rb", buffering=8 * 1024 * 1024) as fin:
        for line in fin:
            try:
                record = orjson.loads(line)
            except Exception:
                continue
            already_chars += len(record.get("text", ""))

    shards = sorted(Path(src_dir).glob("*.jsonl"))
    chars_seen = 0
    chars_appended = 0
    docs_appended = 0
    target_chars = already_chars + overflow_chars

    with open(staging_path, "ab", buffering=8 * 1024 * 1024) as fout:
        for shard in shards:
            if chars_appended >= overflow_chars:
                break
            with open(shard, "rb", buffering=8 * 1024 * 1024) as fin:
                for line in fin:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    record = flatten_datatrove_record(record)
                    text_len = len(record.get("text", ""))
                    chars_seen += text_len
                    # Skip chars already in staging from the initial write
                    if chars_seen <= already_chars:
                        continue
                    fout.write(orjson.dumps(record))
                    fout.write(b"\n")
                    chars_appended += text_len
                    docs_appended += 1
                    if chars_appended >= overflow_chars:
                        break

    return docs_appended, chars_appended


def _shuffle_in_memory(
    staging_paths: dict[str, Path],
    train_path: Path,
    val_path: Path,
    val_fraction: float,
    rng: random.Random,
) -> tuple[int, int]:
    """
    Fast-path shuffle: read everything into RAM, shuffle once, split, write twice.

    After shuffle the order is uniformly random, so taking the first N lines
    as train and the last M lines as val gives an unbiased val sample from
    the same distribution as train.

    Used when the total staging size (scaled by Python object overhead)
    fits in SHUFFLE_RAM_BUDGET_GB.

    Returns:
        (n_train_lines, n_val_lines)
    """
    log.info("Shuffle: reading all staging data into memory...")
    lines: list[bytes] = []
    for source, staging in staging_paths.items():
        with open(staging, "rb") as f:
            for line in f:
                lines.append(line)
        staging.unlink()

    total = len(lines)
    log.info(f"  Loaded {total:,} lines — shuffling...")
    rng.shuffle(lines)

    n_val = max(1, int(total * val_fraction))
    n_train = total - n_val
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    log.info(f"  Writing {n_train:,} lines to {train_path}...")
    with open(train_path, "wb") as fout:
        fout.writelines(train_lines)

    log.info(f"  Writing {n_val:,} lines to {val_path}...")
    with open(val_path, "wb") as fout:
        fout.writelines(val_lines)

    return n_train, n_val


def _shuffle_chunked_from_sources(
    staging_paths: dict[str, Path],
    train_path: Path,
    val_path: Path,
    val_fraction: float,
    rng: random.Random,
    chunk_lines: int = 500_000,
) -> tuple[int, int]:
    """
    Chunked shuffle collapsing merge and shuffle into one pass, then split
    the shuffled output into train.jsonl and val.jsonl.

    Pass 1 — read sequentially from all staging files into chunks,
             shuffle each chunk, write to disk.
    Pass 2 — shuffle chunk order, concatenate sequentially. First
             (1 - val_fraction) of lines go to train.jsonl; the tail
             goes to val.jsonl. Because the chunks themselves are
             shuffled and the chunk order is shuffled, the tail slice
             is a uniform random sample.

    Peak RAM = chunk_lines × avg_line_size.

    Returns:
        (n_train_lines, n_val_lines)
    """
    chunk_dir = train_path.parent / "shuffle_chunks"
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
        staging.unlink()

    _flush_chunk()

    log.info(f"  Wrote {len(chunk_paths)} chunks ({total_lines:,} lines total)")

    n_val = max(1, int(total_lines * val_fraction))
    n_train = total_lines - n_val

    log.info(
        f"Shuffle pass 2/2: interleaving chunks, splitting "
        f"{n_train:,} train / {n_val:,} val..."
    )
    rng.shuffle(chunk_paths)

    written = 0
    train_out = open(train_path, "wb")
    val_out = open(val_path, "wb")
    try:
        for cp in chunk_paths:
            with open(cp, "rb") as fin:
                for line in fin:
                    if written < n_train:
                        train_out.write(line)
                    else:
                        val_out.write(line)
                    written += 1
            cp.unlink()
    finally:
        train_out.close()
        val_out.close()

    try:
        chunk_dir.rmdir()
    except OSError:
        pass
    return n_train, n_val


def stage_blend(target: str, seed: int = 42, workers: int | None = None) -> None:
    """
    Blend sources to the target token ratio and write final train.jsonl + val.jsonl.

    Pass 1 (parallel): each source streams to its own staging file up to
                       its character target or its supply, whichever is
                       smaller. Deficits are recorded per source.

    Pass 2 (sequential): FineWeb appends extra content to cover the total
                         deficit from all supply-constrained sources.

    Pass 3 (shuffle + split): single-pass shuffle followed by a clean
                              (1 - val_fraction) / val_fraction split.
                              Because the shuffle makes order uniformly
                              random, the val slice is an unbiased sample
                              from the same distribution as train.

    Staging files are always rewritten — any existing files from prior
    runs with different mixes would have wrong char counts and are
    removed before staging begins.
    """
    log.info(f"=== Stage 4: Blend (target={target}) ===")
    cfg = TARGET_CONFIGS[target]
    total_tokens = cfg["total_tokens"]
    val_fraction = cfg.get("val_fraction", VAL_FRACTION)

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = CURATED_DIR / "train.jsonl"
    val_path = CURATED_DIR / "val.jsonl"

    if train_path.exists() and val_path.exists():
        log.info("train.jsonl and val.jsonl already exist — delete to re-blend")
        return

    rng = random.Random(seed)

    source_dirs = {
        source: FILTERED_DIR / f"{source}_deduped"
        for source in ALL_SOURCES
    }

    # Initial character targets from the locked mix.
    target_chars = compute_source_char_targets(total_tokens)

    # Remove any stale staging files from prior runs (always re-stage).
    for source in ALL_SOURCES:
        staging = CURATED_DIR / f"blend_{source}.jsonl"
        if staging.exists():
            log.info(f"  {source}: removing stale staging file")
            staging.unlink()

    # ── Pass 1: parallel staging per source ────────────────────────────────────
    n_blend_workers = min(len(ALL_SOURCES), workers or default_workers())

    work: list[tuple] = []
    staging_paths: dict[str, Path] = {}
    source_stats: dict[str, dict] = {}

    for source in ALL_SOURCES:
        src_dir = source_dirs[source]
        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            log.warning(f"  {source}: no deduped shards — skipping")
            continue
        staging = CURATED_DIR / f"blend_{source}.jsonl"
        work.append((source, str(src_dir), str(staging), target_chars[source]))

    if not work:
        log.error("No deduped shards found for any source — nothing to blend")
        return

    log.info(
        f"Pass 1/3: staging {len(work)} sources ({n_blend_workers} workers)..."
    )
    with ProcessPoolExecutor(max_workers=n_blend_workers) as executor:
        futures = {executor.submit(_write_staging, w): w[0] for w in work}
        for future in as_completed(futures):
            source, docs, chars = future.result()
            staging_paths[source] = CURATED_DIR / f"blend_{source}.jsonl"
            source_stats[source] = {
                "docs": docs,
                "chars": chars,
                "target_chars": target_chars[source],
                "deficit": max(0, target_chars[source] - chars),
            }
            deficit_frac = source_stats[source]["deficit"] / max(
                target_chars[source], 1
            )
            flag = " ⚠ short" if deficit_frac > 0.02 else ""
            log.info(
                f"  {source}: {docs:,} docs, "
                f"{chars / 1e9:.3f}B chars "
                f"(target {target_chars[source] / 1e9:.3f}B){flag}"
            )

    # ── Pass 2: FineWeb overflow ───────────────────────────────────────────────
    total_deficit = sum(s["deficit"] for s in source_stats.values())
    overflow_chars = 0
    overflow_docs = 0

    if total_deficit > 0 and OVERFLOW_SINK in staging_paths:
        log.info(
            f"Pass 2/3: FineWeb overflow — covering "
            f"{total_deficit / 1e9:.3f}B character deficit..."
        )
        src_dir = source_dirs[OVERFLOW_SINK]
        staging = staging_paths[OVERFLOW_SINK]
        overflow_docs, overflow_chars = _append_overflow(
            (str(src_dir), str(staging), total_deficit)
        )
        source_stats[OVERFLOW_SINK]["docs"] += overflow_docs
        source_stats[OVERFLOW_SINK]["chars"] += overflow_chars
        source_stats[OVERFLOW_SINK]["overflow_docs"] = overflow_docs
        source_stats[OVERFLOW_SINK]["overflow_chars"] = overflow_chars
        log.info(
            f"  FineWeb overflow: +{overflow_docs:,} docs, "
            f"+{overflow_chars / 1e9:.3f}B chars"
        )
    elif total_deficit > 0:
        log.warning(
            f"Deficit of {total_deficit / 1e9:.3f}B chars but FineWeb "
            f"unavailable — total token count will be below target"
        )

    # ── Pass 3: shuffle + split ────────────────────────────────────────────────
    total_staging_bytes = sum(p.stat().st_size for p in staging_paths.values())
    total_staging_gb = total_staging_bytes / 1e9
    # Python list + bytes object overhead pushes RAM usage to ~5× disk size.
    effective_ram_gb = total_staging_gb * 5
    log.info(
        f"Pass 3/3: shuffling + splitting (val_fraction={val_fraction}) — "
        f"staging on disk {total_staging_gb:.2f} GB, "
        f"effective RAM ~{effective_ram_gb:.2f} GB, "
        f"budget {SHUFFLE_RAM_BUDGET_GB:.1f} GB"
    )

    if effective_ram_gb < SHUFFLE_RAM_BUDGET_GB:
        n_train, n_val = _shuffle_in_memory(
            staging_paths, train_path, val_path, val_fraction, rng,
        )
    else:
        log.info("  Effective RAM exceeds budget — using chunked disk shuffle")
        n_train, n_val = _shuffle_chunked_from_sources(
            staging_paths, train_path, val_path, val_fraction, rng,
        )

    total_lines = n_train + n_val
    total_chars = sum(s["chars"] for s in source_stats.values())
    log.info(
        f"Blend complete — {total_lines:,} documents total "
        f"({n_train:,} train + {n_val:,} val), "
        f"~{total_chars // CHARS_PER_TOKEN / 1e9:.2f}B tokens "
        f"(target {total_tokens / 1e9:.2f}B)"
    )

    # ── Write blend stats ──────────────────────────────────────────────────────
    stats_path = CURATED_DIR / "blend_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "target": target,
            "target_tokens": total_tokens,
            "chars_per_token": CHARS_PER_TOKEN,
            "total_documents": total_lines,
            "train_documents": n_train,
            "val_documents": n_val,
            "val_fraction": val_fraction,
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "source_mix": {
                s: {
                    "docs": v["docs"],
                    "chars": v["chars"],
                    "target_chars": v["target_chars"],
                    "deficit": v["deficit"],
                    **(
                        {
                            "overflow_docs": v["overflow_docs"],
                            "overflow_chars": v["overflow_chars"],
                        }
                        if "overflow_docs" in v
                        else {}
                    ),
                }
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