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

Data mix + token targets are defined in config/data_mix.py and imported
here. Do not add local copies of the source list, percentages, token
targets, CHARS_PER_TOKEN, or CC_CHARS_PER_SEGMENT — those values are
referenced by export.py, notebooks, and tests, and drift between copies
is what this refactor exists to prevent.

Cap-and-redistribute:
    Finite sources (Wikipedia, pg19, etc.) may supply less than their
    character budget allows at large scales. Each source writes up to
    its budget or until its supply is exhausted, whichever comes first.
    The total shortfall is added to OVERFLOW_SINK's budget at the end,
    which acts as the sink. FineWeb (the default sink) has effectively
    unlimited supply (15T tokens) so this always closes the gap.

Per-source download caps:
    Each finite source has a derived `max_docs` cap based on the target
    token budget × the source's share × an inflation factor that absorbs
    filter and dedup losses. This prevents unbounded streaming (which
    bit FineWeb in an earlier run) and keeps download volumes bounded
    per target. See `_AVG_CHARS_PER_DOC`, `_DOWNLOAD_INFLATION`, and
    `_derive_max_docs()` below.

    Buffers are sized to absorb worst-realistic-case filter+dedup
    attrition. Over-buffering wastes disk; under-buffering causes mix
    skew (deficit routes to OVERFLOW_SINK). Erring high is correct.

Blend stage:
    - Pass 1 (parallel): stream each source to a staging file, recording
                         chars written vs target.
    - Pass 2 (sequential): write overflow source's extra content to cover
                           deficit.
    - Pass 3 (shuffle + split): if total size fits in RAM budget, one-shot
                                in-memory shuffle; otherwise weighted-
                                interleave shuffle with reservoir sampling
                                for val.

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

# ── Shared config (single source of truth) ─────────────────────────────────────
# DATA_MIX, CODE_SUBMIX, TARGET_CONFIGS, source lists, and all the locked
# curator constants live in config/data_mix.py. Nothing in this file
# redeclares them.
from config import (
    DATA_MIX,
    CODE_SUBMIX,
    OVERFLOW_SINK,
    NON_CODE_SOURCES,
    CODE_SOURCES,
    ALL_SOURCES,
    TARGET_CONFIGS,
    CHARS_PER_TOKEN,
    CC_CHARS_PER_SEGMENT,
    SHUFFLE_RAM_BUDGET_GB,
    PRETRAIN_VAL_FRACTION,
    MINI_OVERRIDES,
)

from curator.filters.dedup import Deduplicator
from curator.filters.quality import QualityFilter

from curator.sources.common_crawl import CommonCrawlSource
from curator.sources.fineweb import FineWebSource
from curator.sources.wikipedia import WikipediaSource
from curator.sources.pg19 import PG19Source
from curator.sources.pes2o import PeS2oSource
from curator.sources.open_web_math import OpenWebMathSource
from curator.sources.stackexchange import StackExchangeSource
from curator.sources.synthetic_arithmetic import SyntheticArithmeticSource
from curator.sources.code_search_net import CodeSearchNetSource
from curator.sources.stack_smol import StackSmolSource
from curator.sources.stack_v1 import StackV1Source
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


# ── Local share lookups ────────────────────────────────────────────────────────
#
# DATA_MIX stores percentages as floats (10.0, 47.5 ...) for display. The
# curator's math wants shares as fractions (0.10, 0.475 ...). Derive the
# fractional views once here.
_TOP_LEVEL_SHARE: dict[str, float] = {
    name: entry["pct"] / 100.0 for name, entry in DATA_MIX.items()
}
_CODE_SUB_SHARE: dict[str, float] = {
    name: entry["pct"] / 100.0 for name, entry in CODE_SUBMIX.items()
}

# Data directories
DATA_DIR     = Path(os.environ.get("DATA_DIR", "data"))
RAW_DIR      = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
CURATED_DIR  = DATA_DIR / "curated"


# ── Per-source download cap derivation ─────────────────────────────────────────
#
# Translating a char target into a doc cap requires knowing the avg chars/doc
# for each source. Values below are measured from a 125m run (sample
# data/raw/<source>/*.jsonl) — adjust if reality drifts by >2× from these.
# The inflation factor absorbs filter losses (~40% typical), dedup losses
# (~20% typical), and headroom against the avg-chars-per-doc estimate.
#
# Buffer sizing rationale:
#   fineweb       — also OVERFLOW_SINK; needs extra to absorb other deficits
#   wikipedia     — very clean upstream, lower attrition expected
#   pg19          — small absolute count, want safety margin
#   pes2o         — abstracts only (~1.4K chars), supply-bound at 350m+
#   open_web_math — math notation challenges, harsher attrition
#   stackexchange — mostly well-formed, moderate attrition
#   stack_v1      — large files, MinHash dedup is heavy → 5× inflation
#   stack_smol    — small curated subset, lower attrition
#   jupyter       — notebook structure, moderate attrition
#
# Code sub-sources codesearchnet and conala are NOT in the tables below —
# both are supply-bound at 350m+ (codesearchnet ~2M docs upstream, conala
# ~600K). A derived cap would exceed upstream and be a no-op. They stream
# their full corpus; deficit routes to OVERFLOW_SINK like any other shortfall.
#
# Several other sources also become supply-bound at 1b (wikipedia, pg19,
# open_web_math, stack_smol). The cap is still set so that downloads at
# smaller scales remain bounded; at 1b the upstream supply binds first
# and the deficit routes to OVERFLOW_SINK by design.

_AVG_CHARS_PER_DOC: dict[str, int] = {
    "fineweb":       3_000,
    "wikipedia":     5_000,
    "pg19":          400_000,
    "pes2o":         1_400,
    "open_web_math": 8_000,
    "stackexchange": 1_700,
    "synthetic_arithmetic": 1_500,
    "stack_v1":      5_500,
    "stack_smol":    10_000,
    "jupyter":       11_000,
}

_DOWNLOAD_INFLATION: dict[str, float] = {
    "fineweb":       5.0,
    "wikipedia":     3.0,
    "pg19":          5.0,
    "pes2o":         5.0,
    "open_web_math": 5.0,
    "stackexchange": 5.0,
    "stack_v1":      5.0,
    "stack_smol":    5.0,
    "jupyter":       5.0,
}


def _derive_max_docs(name: str, target: str) -> int | None:
    """
    Derive a per-source max_docs cap from the target token budget.

    Returns None for sources we don't cap:
      - common_crawl: has its own segment-based budgeting via compute_cc_segments
      - codesearchnet, conala: supply-bound — upstream has fewer docs than
        even the 1b target needs, so a derived cap would exceed upstream
        and be a no-op.

    Formula for a top-level source (name in DATA_MIX):
        target_chars = total_tokens × _TOP_LEVEL_SHARE[name] × CHARS_PER_TOKEN

    Formula for a code sub-source (name in CODE_SUBMIX):
        target_chars = total_tokens
                     × _TOP_LEVEL_SHARE["code"]
                     × _CODE_SUB_SHARE[name]
                     × CHARS_PER_TOKEN

    Then in both cases:
        max_docs = (target_chars / avg_chars_per_doc) × inflation
    """
    if name not in _AVG_CHARS_PER_DOC:
        return None

    target_tokens = TARGET_CONFIGS[target]["total_tokens"]
    avg_chars     = _AVG_CHARS_PER_DOC[name]
    inflation     = _DOWNLOAD_INFLATION[name]

    if name in _TOP_LEVEL_SHARE:
        share = _TOP_LEVEL_SHARE[name]
    elif name in _CODE_SUB_SHARE:
        share = _TOP_LEVEL_SHARE["code"] * _CODE_SUB_SHARE[name]
    else:
        # Source has chars/doc data but no share — shouldn't happen for
        # well-formed config. Conservative: don't cap.
        return None

    target_chars = target_tokens * share * CHARS_PER_TOKEN
    return int((target_chars / avg_chars) * inflation)


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_cc_segments(total_tokens: int) -> int:
    """
    Segments of Common Crawl needed to hit CC's character share.

    Computed from: total_tokens × DATA_MIX[common_crawl] share × CHARS_PER_TOKEN
    bytes of text, divided by CC_CHARS_PER_SEGMENT bytes produced per segment
    after trafilatura + language filtering.
    """
    cc_share = _TOP_LEVEL_SHARE["common_crawl"]
    target_chars = int(total_tokens * cc_share * CHARS_PER_TOKEN)
    return max(1, math.ceil(target_chars / CC_CHARS_PER_SEGMENT))


def compute_source_char_targets(total_tokens: int) -> dict[str, int]:
    """
    Compute the character budget for each source from the target tokens.

    Returns a dict mapping each concrete source name (7 non-code + 5 code)
    to its target character count. Code sources get their share of the
    10% code budget according to CODE_SUBMIX.
    """
    targets: dict[str, int] = {}
    for source, share in _TOP_LEVEL_SHARE.items():
        if source == "code":
            continue
        targets[source] = int(total_tokens * share * CHARS_PER_TOKEN)

    code_total_chars = int(total_tokens * _TOP_LEVEL_SHARE["code"] * CHARS_PER_TOKEN)
    for code_source, sub_share in _CODE_SUB_SHARE.items():
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

    # Resolve the doc cap:
    #   - mini: from MINI_OVERRIDES (per-source small caps for pipeline testing)
    #   - non-mini: derived from target token budget × share × inflation
    #               (None for sources not in the derivation table — currently
    #                codesearchnet and conala, both supply-bound at 350m+)
    if mini:
        cap = MINI_OVERRIDES.get(name)
    else:
        cap = _derive_max_docs(name, target)
        if cap is not None:
            log.info(
                f"{name} cap derived from {target}: {cap:,} docs "
                f"(avg {_AVG_CHARS_PER_DOC.get(name, 0):,} chars/doc, "
                f"{_DOWNLOAD_INFLATION.get(name, 0)}× inflation)"
            )

    # CC has different 'cap' semantics: max_segments, and needs crawls + workers.
    if name == "common_crawl":
        cfg = TARGET_CONFIGS[target]
        if mini:
            max_segments = MINI_OVERRIDES.get(name)
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
    if name == "synthetic_arithmetic":
        return SyntheticArithmeticSource(output_dir=raw_dir, max_docs=cap)
    if name == "codesearchnet":
        return CodeSearchNetSource(output_dir=raw_dir, max_docs=cap)
    if name == "stack_smol":
        return StackSmolSource(output_dir=raw_dir, max_docs=cap)
    if name == "stack_v1":
        return StackV1Source(output_dir=raw_dir, max_docs=cap)
    if name == "jupyter":
        return JupyterSource(output_dir=raw_dir, max_docs=cap)
    if name == "conala":
        return ConalaSource(output_dir=raw_dir, max_docs=cap)

    raise ValueError(f"Unknown source: {name}")


def stage_download(target: str, mini: bool = False, workers: int | None = None) -> None:
    """Download every source in DATA_MIX + CODE_SUBMIX."""
    n_workers = workers or default_workers()
    log.info(f"=== Stage 1: Download (target={target}, mini={mini}) ===")

    if not mini:
        cc_segments = compute_cc_segments(TARGET_CONFIGS[target]["total_tokens"])
        log.info(
            f"Common Crawl: computed {cc_segments} segments from "
            f"{TARGET_CONFIGS[target]['total_tokens']:,} tokens × "
            f"{_TOP_LEVEL_SHARE['common_crawl']:.2%} × {CHARS_PER_TOKEN} chars/tok "
            f"÷ {CC_CHARS_PER_SEGMENT:,} chars/segment"
        )

    for name in ALL_SOURCES:
        log.info(f"Downloading {name}...")
        source = _build_source(name, mini=mini, target=target, workers=n_workers)
        try:
            source.download()
        except Exception:
            log.exception(f"{name}: download failed — continuing with remaining sources")
            continue
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
    parse_errors = 0
    with open(shard, "rb", buffering=8 * 1024 * 1024) as fin, \
         open(out_path, "wb", buffering=8 * 1024 * 1024) as fout:
        for line in fin:
            try:
                record = orjson.loads(line)
            except Exception:
                parse_errors += 1
                continue
            kept, _ = qf.check(record)
            if kept:
                fout.write(orjson.dumps(record))
                fout.write(b"\n")

    report = qf.report()
    if parse_errors:
        report = f"{report} | parse_errors={parse_errors} in {shard.name}"
    return report


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
    written so the main process can compute deficits for overflow.

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


def _append_overflow(args: tuple) -> tuple[int, int]:
    """
    Append overflow-source docs to its staging file to cover the total deficit.
    Runs in a subprocess.

    Reads OVERFLOW_SINK deduped shards from where the initial staging pass
    left off (determined by counting chars already in the staging file)
    and appends until `overflow_chars` additional chars have been written.

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
) -> tuple[int, int, dict[str, int]]:
    """
    Fast-path shuffle: read everything into RAM, shuffle once, split, write twice.

    After shuffle the order is uniformly random across all sources, so
    taking the first N lines as train and the last M lines as val gives
    an unbiased val sample from the same distribution as train.

    Used when the total staging size (scaled by Python object overhead)
    fits in SHUFFLE_RAM_BUDGET_GB.

    Returns:
        (n_train_lines, n_val_lines, val_source_counts)
    """
    log.info("Shuffle: reading all staging data into memory...")
    # Track source per line by maintaining a parallel list of (line, source).
    # Cheap (~2× memory of the line itself for a short string) and avoids
    # a post-write scan of val.jsonl that depends on the `source` field.
    pairs: list[tuple[bytes, str]] = []
    for source, staging in staging_paths.items():
        with open(staging, "rb") as f:
            for line in f:
                pairs.append((line, source))
        staging.unlink()

    total = len(pairs)
    log.info(f"  Loaded {total:,} lines — shuffling...")
    rng.shuffle(pairs)

    n_val = max(1, int(total * val_fraction))
    n_train = total - n_val
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    del pairs

    val_source_counts: dict[str, int] = {}
    for _, src in val_pairs:
        val_source_counts[src] = val_source_counts.get(src, 0) + 1

    log.info(f"  Writing {n_train:,} lines to {train_path}...")
    with open(train_path, "wb") as fout:
        fout.writelines(line for line, _ in train_pairs)

    log.info(f"  Writing {n_val:,} lines to {val_path}...")
    with open(val_path, "wb") as fout:
        fout.writelines(line for line, _ in val_pairs)

    return n_train, n_val, val_source_counts


def _shuffle_chunked_from_sources(
    staging_paths: dict[str, Path],
    train_path: Path,
    val_path: Path,
    val_fraction: float,
    rng: random.Random,
    total_lines: int,
    source_doc_counts: dict[str, int],
    chunk_lines: int = 500_000,
) -> tuple[int, int, dict[str, int]]:
    """
    Single-pass shuffle: weighted-interleave reads + reservoir sample for val.

    Two bugs the previous tail-slice version had:

      Bug 1 — train source-purity. Staging files were read sequentially
              (source-by-source). Chunks of chunk_lines fill source-by-source
              too: the first chunks were a mix of small early sources, and
              late chunks were 100% fineweb (since fineweb alone is more
              than 7 chunks' worth at 125m). Shuffling chunk *order* doesn't
              homogenize chunks whose contents are already source-pure —
              the resulting train.jsonl had 500k-line source-contiguous
              regions, e.g. the first 100k lines were all fineweb.

      Bug 2 — val tail-slice bias. Taking the last n_val lines of the
              concatenated chunks meant val composition was whatever
              chunk(s) happened to land at the tail of the chunk-order
              shuffle, not a uniform sample of the corpus.

    Fix:

      Pass 1 — weighted-interleave reads. Open all staging files at once.
               At each step pick a source with probability proportional to
               its remaining line count, read one line from that source.
               This produces a stream where any window's source mix matches
               the global mix.

               Each line then enters the val reservoir or the train chunk
               buffer. Reservoir uses Vitter's Algorithm R: every line in
               the corpus has equal probability n_val/total of landing in
               val regardless of its source or arrival position.

               When the train buffer hits chunk_lines, shuffle and write
               the chunk to disk. Each chunk now contains a representative
               source mix because the input stream was already mixed.

      Pass 2 — shuffle the chunk order, concatenate to train.jsonl.
               Reservoir → val.jsonl directly.

    Memory:
      train_buf:     chunk_lines × avg_line_size  (default 500k × ~1KB ≈ 500MB)
      val_reservoir: n_val × avg_line_size        (~tens of MB at 125m–1b)
      file handles:  one per source (≤ 12)

    Args:
        total_lines: Sum of doc counts across all staging files (post-overflow).
        source_doc_counts: Per-source doc count for weighting. Sources with
                           a staging file but missing from this dict default
                           to 0 weight (clamped to ≥1 below so the source
                           still gets drained).

    Returns:
        (n_train_actual, n_val_actual, val_source_counts)
    """
    n_val = max(1, int(total_lines * val_fraction))

    chunk_dir = train_path.parent / "shuffle_chunks"
    chunk_dir.mkdir(exist_ok=True)
    chunk_paths: list[Path] = []

    log.info(
        f"Shuffle pass 1/2: weighted-interleave streaming "
        f"({len(staging_paths)} sources, total {total_lines:,} lines), "
        f"reservoir-sampling {n_val:,} for val..."
    )

    # Open all staging files. We close + unlink each as it's exhausted.
    handles: dict[str, "object"] = {}
    remaining: dict[str, int] = {}
    for source, path in staging_paths.items():
        handles[source] = open(path, "rb", buffering=8 * 1024 * 1024)
        remaining[source] = source_doc_counts.get(source, 0)

    val_reservoir: list[bytes] = []
    val_source_reservoir: list[str] = []  # parallel array of sources for val
    train_buf: list[bytes] = []
    chunk_idx = 0
    seen = 0

    def _flush_chunk():
        nonlocal chunk_idx
        if not train_buf:
            return
        rng.shuffle(train_buf)
        p = chunk_dir / f"chunk_{chunk_idx:06d}.jsonl"
        with open(p, "wb", buffering=8 * 1024 * 1024) as fout:
            fout.writelines(train_buf)
        chunk_paths.append(p)
        train_buf.clear()
        chunk_idx += 1

    active = list(handles.keys())

    while active:
        # Weighted-random source pick. Clamp weight to ≥1 so a source
        # whose remaining count is undercounted still gets drained.
        weights = [max(1, remaining[s]) for s in active]
        chosen = rng.choices(active, weights=weights, k=1)[0]

        line = handles[chosen].readline()
        if not line:
            # Source exhausted — close handle, drop from active list.
            handles[chosen].close()
            try:
                staging_paths[chosen].unlink()
            except FileNotFoundError:
                pass
            active.remove(chosen)
            continue

        remaining[chosen] = max(0, remaining[chosen] - 1)
        seen += 1

        # Reservoir sampling (Vitter Algorithm R).
        if seen <= n_val:
            val_reservoir.append(line)
            val_source_reservoir.append(chosen)
        else:
            j = rng.randint(1, seen)
            if j <= n_val:
                displaced_line = val_reservoir[j - 1]
                val_reservoir[j - 1] = line
                val_source_reservoir[j - 1] = chosen
                train_buf.append(displaced_line)
            else:
                train_buf.append(line)

        if len(train_buf) >= chunk_lines:
            _flush_chunk()

    _flush_chunk()

    if seen != total_lines:
        log.warning(
            f"Reservoir saw {seen:,} lines but expected {total_lines:,} — "
            f"val sampling fraction will drift slightly from {val_fraction:.4f}"
        )

    val_source_counts: dict[str, int] = {}
    for s in val_source_reservoir:
        val_source_counts[s] = val_source_counts.get(s, 0) + 1

    log.info(
        f"  Wrote {len(chunk_paths)} train chunks, "
        f"reservoir holds {len(val_reservoir):,} val lines"
    )

    # Shuffle reservoir order so val.jsonl ordering doesn't carry
    # arrival-time signal. Sample membership is already uniform; this
    # just randomizes within-file row order.
    paired = list(zip(val_reservoir, val_source_reservoir))
    rng.shuffle(paired)
    val_reservoir = [line for line, _ in paired]
    del paired, val_source_reservoir

    log.info(
        f"Shuffle pass 2/2: writing {len(val_reservoir):,} val lines + "
        f"concatenating shuffled train chunks..."
    )

    with open(val_path, "wb", buffering=8 * 1024 * 1024) as fout:
        fout.writelines(val_reservoir)
    n_val_actual = len(val_reservoir)
    val_reservoir.clear()  # free RAM before train write

    rng.shuffle(chunk_paths)
    n_train_actual = 0
    with open(train_path, "wb") as train_out:
        for cp in chunk_paths:
            with open(cp, "rb") as fin:
                for line in fin:
                    train_out.write(line)
                    n_train_actual += 1
            cp.unlink()

    try:
        chunk_dir.rmdir()
    except OSError:
        pass

    return n_train_actual, n_val_actual, val_source_counts


def stage_blend(target: str, seed: int = 42, workers: int | None = None) -> None:
    """
    Blend sources to the target token ratio and write final train.jsonl + val.jsonl.

    Pass 1 (parallel): each source streams to its own staging file up to
                       its character target or its supply, whichever is
                       smaller. Deficits are recorded per source.

    Pass 2 (sequential): OVERFLOW_SINK appends extra content to cover the
                         total deficit from all supply-constrained sources.

    Pass 3 (shuffle + split): if effective RAM is below budget, one-shot
                              in-memory shuffle. Otherwise weighted-
                              interleave streaming with reservoir sampling
                              for val. Both paths produce a globally-mixed
                              train and a uniform-sample val.

    Staging files are always rewritten — any existing files from prior
    runs with different mixes would have wrong char counts and are
    removed before staging begins.
    """
    log.info(f"=== Stage 4: Blend (target={target}) ===")
    cfg = TARGET_CONFIGS[target]
    total_tokens = cfg["total_tokens"]
    val_fraction = cfg.get("val_fraction", PRETRAIN_VAL_FRACTION)

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

    # ── Pass 2: overflow ───────────────────────────────────────────────────────
    total_deficit = sum(s["deficit"] for s in source_stats.values())
    overflow_chars = 0
    overflow_docs = 0

    if total_deficit > 0 and OVERFLOW_SINK in staging_paths:
        log.info(
            f"Pass 2/3: {OVERFLOW_SINK} overflow — covering "
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
            f"  {OVERFLOW_SINK} overflow: +{overflow_docs:,} docs, "
            f"+{overflow_chars / 1e9:.3f}B chars"
        )
    elif total_deficit > 0:
        log.warning(
            f"Deficit of {total_deficit / 1e9:.3f}B chars but {OVERFLOW_SINK} "
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
        n_train, n_val, val_source_counts = _shuffle_in_memory(
            staging_paths, train_path, val_path, val_fraction, rng,
        )
    else:
        log.info("  Effective RAM exceeds budget — using chunked disk shuffle")
        # Weighted-interleave needs total_lines + per-source counts up
        # front. Both come from source_stats, finalized after pass 2.
        source_doc_counts = {
            s: source_stats[s]["docs"]
            for s in staging_paths.keys()
            if s in source_stats
        }
        total_lines_calc = sum(source_doc_counts.values())
        n_train, n_val, val_source_counts = _shuffle_chunked_from_sources(
            staging_paths, train_path, val_path, val_fraction, rng,
            total_lines=total_lines_calc,
            source_doc_counts=source_doc_counts,
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
                    "val_docs": val_source_counts.get(s, 0),
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