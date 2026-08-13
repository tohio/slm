"""
curator/scripts/curate.py
--------------------------
Main data curation pipeline.

Orchestrates configured data sources through quality filtering, deduplication,
and blending. Produces train.jsonl + val.jsonl ready for tokenizer
training and model pretraining. The val split is sampled uniformly from
the shuffled blend output, so it represents the same distribution as train.

Pipeline:
    1. Download sources
    2. Apply quality filters
    3. Deduplicate (exact SHA-256 + datatrove disk-based MinHash LSH)
    4. Blend sources to target token ratios (with cap-and-redistribute)
    Artifact transfer is a separate RUN_ID-scoped command.

Data mix + token targets are defined in config/data_mix.py and imported
here. Do not add local copies of the source list, percentages, token
targets, CHARS_PER_TOKEN, or CC_CHARS_PER_SEGMENT — those values are
referenced by export.py, notebooks, and tests, and drift between copies
is what this refactor exists to prevent.

Cap-and-redistribute:
    Finite sources may supply less than their character budget allows at
    large scales. Each source writes up to its budget or until its supply is
    exhausted, whichever comes first. Local synthetic source deficits route
    first to Nemotron Specialized, then FineWeb-Edu, then FineWeb. General
    source deficits route first to FineWeb-Edu, then FineWeb. FineWeb remains
    the final broad-web fallback via OVERFLOW_SINK.

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
    - Pass 2 (sequential): route deficits through source-aware overflow
                                chains.
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
import logging
import math
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    SYNTHETIC_SOURCES,
    FILTER_SOURCE_FAMILIES,
    PROSE_HEURISTIC_SKIP_SOURCES,
    DEDUP_PRIORITY,
    TARGET_CONFIGS,
    CHARS_PER_TOKEN,
    CC_CHARS_PER_SEGMENT,
    SHUFFLE_RAM_BUDGET_GB,
    PRETRAIN_VAL_FRACTION,
    MINI_OVERRIDES,
    SUPPLEMENTAL_CHAR_CAPS,
    source_filter_family,
    benchmark_decontamination_contract,
)

from config.data_mix import (
    SYNTHETIC_AVG_CHARS_PER_DOC,
    SYNTHETIC_DOC_INFLATION,
)

from config.paths import (
    data_run_dir, raw_dir, filtered_dir, dedup_scratch_dir, curated_dir,
)

from curator.filters.dedup import (
    MINHASH_CONTRACT,
    MINHASH_LSH_CROSSOVER,
    Deduplicator,
)
from curator.filters.overlap import audit_exact_split_overlap
from curator.filters.benchmark_contamination import (
    BenchmarkContaminationAuditor,
    build_benchmark_index,
)
from curator.filters.near_overlap import (
    JsonlByteRangeReader,
    NearOverlapReporter,
    audit_minhash_split_overlap,
    build_cross_index_removals,
)
from curator.filters.sensitive_content import (
    SENSITIVE_CONTENT_CONTRACT,
    SensitiveContentAuditor,
)
from curator.filters.quality import QualityFilter, require_fasttext_model
from curator.filters.segments import segment_long_document
from curator.state import (
    MANIFEST_NAME,
    atomic_write_json,
    code_fingerprint,
    file_snapshot,
    manifest_matches,
    manifest_outputs_match,
    stable_digest,
    tree_signature,
    write_manifest,
)

from curator.sources.common_crawl import CommonCrawlSource
from curator.sources.hf import resolve_dataset_revision
from curator.sources.fineweb import FineWebSource
from curator.sources.fineweb_edu import FineWebEduSource
from curator.sources.wikipedia import WikipediaSource
from curator.sources.pg19 import PG19Source
from curator.sources.pes2o import PeS2oSource
from curator.sources.stackexchange import StackExchangeSource
from curator.sources.hf_synthetic import (
    SyntheticArithmeticSource,
    SyntheticTaskCodeSource,
    EducationalQAMCQMathSource,
    EducationalQAMCQGeneralSource,
    FactualRestraintSource,
)
from curator.sources.code_search_net import CodeSearchNetSource
from curator.sources.stack_smol import StackSmolSource
from curator.sources.stack_v1 import StackV1Source
from curator.sources.jupyter import JupyterSource
from curator.sources.conala import ConalaSource
from curator.sources.nemotron_cc_math import NemotronCCMathSource
from curator.sources.nemotron_specialized import NemotronSpecializedSource

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
# DATA_MIX stores percentages as floats (for example 5.0, 26.5, 15.0) for display. The
# curator's math wants shares as fractions (0.10, 0.475 ...). Derive the
# fractional views once here.
_TOP_LEVEL_SHARE: dict[str, float] = {
    name: entry["pct"] / 100.0 for name, entry in DATA_MIX.items()
}
_CODE_SUB_SHARE: dict[str, float] = {
    name: entry["pct"] / 100.0 for name, entry in CODE_SUBMIX.items()
}

# Data directories are target-scoped. DATA_DIR remains the base root
# (default: data); curation writes under data/runs/<target>/...
DATA_DIR     = data_run_dir("125m")
RAW_DIR      = raw_dir("125m")
FILTERED_DIR = filtered_dir("125m")
DEDUP_SCRATCH_DIR = dedup_scratch_dir("125m")
CURATED_DIR  = curated_dir("125m")


def configure_data_dirs(target: str) -> None:
    """Configure target-scoped curation artifact directories."""
    global DATA_DIR, RAW_DIR, FILTERED_DIR, DEDUP_SCRATCH_DIR, CURATED_DIR
    DATA_DIR = data_run_dir(target)
    RAW_DIR = raw_dir(target)
    FILTERED_DIR = filtered_dir(target)
    DEDUP_SCRATCH_DIR = dedup_scratch_dir(target)
    CURATED_DIR = curated_dir(target)

# Generated/template-like sources are intentionally dense and repetitive.
# Fuzzy MinHash dedup collapses them aggressively, which destroys the intended
# amplified signal. These sources still run exact dedup, but bypass fuzzy
# MinHash so near-duplicate templates remain available as training signal.
SKIP_FUZZY_DEDUP_SOURCES = SYNTHETIC_SOURCES

# FineWeb applies MinHash independently to each Common Crawl dump. Exact
# normalized dedup remains cross-source and cross-dump; only fuzzy candidate
# clustering is partitioned here.
FUZZY_DEDUP_PARTITION_FIELDS = {
    "common_crawl": "crawl",
}

# Synthetic sources are externally generated and published to Hugging Face.
# They are controlled behavior supplements, not bulk reservoirs. If they
# underfill, use the scalable specialized synthetic source first, then cleaner
# educational web text, and only then the broad FineWeb fallback.
SYNTHETIC_OVERFLOW_CHAIN = (
    "nemotron_specialized",
    "fineweb_edu",
    OVERFLOW_SINK,
)

DEFAULT_OVERFLOW_CHAIN = (
    "fineweb_edu",
    OVERFLOW_SINK,
)


# ── Source selection / source-scoped runs ──────────────────────────────────────

def _resolve_sources(source_arg: str | None) -> list[str]:
    """Resolve a comma-separated --sources value into concrete source names."""
    if not source_arg:
        return list(ALL_SOURCES)

    requested: list[str] = []
    seen: set[str] = set()
    for raw in source_arg.split(","):
        name = raw.strip()
        if not name or name in seen:
            continue
        requested.append(name)
        seen.add(name)

    unknown = sorted(set(requested) - set(ALL_SOURCES))
    if unknown:
        valid = ", ".join(ALL_SOURCES)
        raise ValueError(
            f"Unknown source(s): {', '.join(unknown)}. Valid sources: {valid}"
        )
    if not requested:
        raise ValueError("--sources was provided but no source names were parsed")
    return requested


def _count_jsonl_dir(path: Path) -> tuple[int, int, int]:
    """Return (files, docs, chars) for JSONL shards in a directory."""
    files = sorted(path.glob("*.jsonl"))
    docs = 0
    chars = 0

    for shard in files:
        with open(shard, "rb", buffering=8 * 1024 * 1024) as f:
            for line in f:
                try:
                    row = orjson.loads(line)
                except Exception:
                    continue
                docs += 1
                chars += len(row.get("text", ""))

    return len(files), docs, chars


def stage_source_stats(sources: list[str] | None = None) -> None:
    """Report raw/filter/dedup capacity stats for selected sources."""
    selected_sources = sources or list(ALL_SOURCES)
    log.info("=== Source capacity stats ===")

    for source in selected_sources:
        log.info("=" * 80)
        log.info(source)

        rows: dict[str, tuple[int, int, int]] = {}
        for label, path in {
            "raw": RAW_DIR / source,
            "filtered": FILTERED_DIR / source,
            "deduped": FILTERED_DIR / f"{source}_deduped",
        }.items():
            files, docs, chars = _count_jsonl_dir(path)
            rows[label] = (files, docs, chars)
            avg = chars // max(docs, 1)
            log.info(
                f"  {label:8s} files={files:5,} "
                f"docs={docs:12,} chars={chars:15,} avg_chars_doc={avg:,}"
            )

        raw_docs, raw_chars = rows["raw"][1], rows["raw"][2]
        filt_docs, filt_chars = rows["filtered"][1], rows["filtered"][2]
        ded_docs, ded_chars = rows["deduped"][1], rows["deduped"][2]

        log.info(
            f"  filter_doc_retention={filt_docs / max(raw_docs, 1) * 100:8.2f}% "
            f"dedup_doc_retention={ded_docs / max(filt_docs, 1) * 100:8.2f}% "
            f"total_char_retention={ded_chars / max(raw_chars, 1) * 100:8.2f}%"
        )


# ── Per-source download cap derivation ─────────────────────────────────────────
#
# Translating a char target into a doc cap requires knowing the avg chars/doc
# for each source. Values below are measured from a 125m run (sample
# data/runs/<size>/raw/<source>/*.jsonl) — adjust if reality drifts by >2× from these.
# The inflation factor absorbs filter losses (~40% typical), dedup losses
# (~20% typical), and headroom against the avg-chars-per-doc estimate.
#
# Buffer sizing rationale:
#   fineweb       — also OVERFLOW_SINK; needs extra to absorb other deficits
#   wikipedia     — very clean upstream, lower attrition expected
#   pg19          — char-capped in _derive_max_chars; docs cap is safety only
#   pes2o         — abstracts only (~1.4K chars), supply-bound at 350m+
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
# stack_smol). The cap is still set so that downloads at
# smaller scales remain bounded; at 1b the upstream supply binds first
# and the deficit routes to OVERFLOW_SINK by design.

_AVG_CHARS_PER_DOC: dict[str, int] = {
    "fineweb":       3_000,
    "fineweb_edu":   3_000,
    "wikipedia":     5_000,
    "pg19":          400_000,
    "pes2o":         1_400,
    "nemotron_cc_math": 4_000,
    "nemotron_specialized": 2_200,
    "stackexchange": 1_700,
    "stack_v1":      5_500,
    "stack_smol":    10_000,
    "jupyter":       11_000,
}

_DOWNLOAD_INFLATION: dict[str, float] = {
    "fineweb":       5.0,
    "fineweb_edu":   5.0,
    "wikipedia":     3.0,
    "pg19":          1.5,
    "pes2o":         5.0,
    "nemotron_cc_math": 2.0,
    "nemotron_specialized": 1.6,
    "stackexchange": 5.0,
    "stack_v1":      5.0,
    "stack_smol":    5.0,
    "jupyter":       5.0,
}



def _source_target_chars(name: str, target: str) -> int | None:
    """Return target chars for a concrete source, applying supplemental caps."""
    target_tokens = TARGET_CONFIGS[target]["corpus_tokens"]
    if name in _TOP_LEVEL_SHARE:
        share = _TOP_LEVEL_SHARE[name]
    elif name in _CODE_SUB_SHARE:
        share = _TOP_LEVEL_SHARE["code"] * _CODE_SUB_SHARE[name]
    else:
        return None
    target_chars = int(target_tokens * share * CHARS_PER_TOKEN)
    cap = SUPPLEMENTAL_CHAR_CAPS.get(name, {}).get(target)
    if cap is not None:
        return min(target_chars, cap)
    return target_chars

def _derive_max_docs(name: str, target: str) -> int | None:
    """
    Derive a per-source max_docs cap from the target token budget.

    Returns None for sources we don't cap:
      - common_crawl: has its own segment-based budgeting via compute_cc_segments
      - codesearchnet, conala: supply-bound — upstream has fewer docs than
        even the 1b target needs, so a derived cap would exceed upstream
        and be a no-op.

    Formula for a top-level source (name in DATA_MIX):
        target_chars = corpus_tokens × _TOP_LEVEL_SHARE[name] × CHARS_PER_TOKEN

    Formula for a code sub-source (name in CODE_SUBMIX):
        target_chars = total_tokens
                     × _TOP_LEVEL_SHARE["code"]
                     × _CODE_SUB_SHARE[name]
                     × CHARS_PER_TOKEN

    Then in both cases:
        max_docs = (target_chars / avg_chars_per_doc) × inflation
    """
    if name in SYNTHETIC_SOURCES:
        avg_chars = SYNTHETIC_AVG_CHARS_PER_DOC[name]
        inflation = SYNTHETIC_DOC_INFLATION
    elif name in _AVG_CHARS_PER_DOC:
        avg_chars = _AVG_CHARS_PER_DOC[name]
        inflation = _DOWNLOAD_INFLATION[name]
    else:
        return None

    target_chars = _source_target_chars(name, target)
    if target_chars is None:
        return None
    return max(1, int((target_chars / avg_chars) * inflation))


# ── Helpers ────────────────────────────────────────────────────────────────────


def _derive_max_chars(name: str, target: str) -> int | None:
    """
    Derive a per-source raw character cap from the target token budget.

    PG-19 books are very large. A docs-only cap can download several times
    more text than blend can use, so PG-19 gets a character cap with a
    modest buffer. Other sources continue using max_docs caps.
    """
    target_chars = _source_target_chars(name, target)
    if target_chars is None:
        return None

    if name == "pg19":
        return int(target_chars * 1.30)

    if name in SYNTHETIC_SOURCES:
        return int(target_chars * 2.00)

    return None

def compute_cc_segments(total_tokens: int) -> int:
    """
    Segments of Common Crawl needed to hit CC's character share.

    Computed from: corpus_tokens × DATA_MIX[common_crawl] share × CHARS_PER_TOKEN
    bytes of text, divided by CC_CHARS_PER_SEGMENT bytes produced per segment
    after trafilatura + language filtering.
    """
    cc_share = _TOP_LEVEL_SHARE["common_crawl"]
    target_chars = int(total_tokens * cc_share * CHARS_PER_TOKEN)
    return max(1, math.ceil(target_chars / CC_CHARS_PER_SEGMENT))


def compute_source_char_targets(target: str) -> dict[str, int]:
    """Compute complete character budgets with fixed-supply caps redistributed.

    Supplemental caps reduce a source's nominal share, but they must not reduce
    the corpus target. The capped amount is reassigned to the configured
    scalable overflow sink before any source is staged.
    """
    targets: dict[str, int] = {}
    uncapped_total = 0
    for source in _TOP_LEVEL_SHARE:
        if source == "code":
            continue
        uncapped_total += int(
            TARGET_CONFIGS[target]["corpus_tokens"]
            * _TOP_LEVEL_SHARE[source]
            * CHARS_PER_TOKEN
        )
        chars = _source_target_chars(source, target)
        if chars is not None:
            targets[source] = chars
    for code_source in _CODE_SUB_SHARE:
        uncapped_total += int(
            TARGET_CONFIGS[target]["corpus_tokens"]
            * _TOP_LEVEL_SHARE["code"]
            * _CODE_SUB_SHARE[code_source]
            * CHARS_PER_TOKEN
        )
        chars = _source_target_chars(code_source, target)
        if chars is not None:
            targets[code_source] = chars

    cap_shortfall = uncapped_total - sum(targets.values())
    if cap_shortfall < 0:
        raise RuntimeError(
            f"Source targets exceed uncapped target by {-cap_shortfall:,} chars"
        )
    targets[OVERFLOW_SINK] += cap_shortfall
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
    raw_root: Path | None = None,
) -> object:
    """Construct a source instance with mini caps applied when mini=True."""
    raw_dir = (raw_root or RAW_DIR) / name

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
            if name in SYNTHETIC_SOURCES:
                avg_for_log = SYNTHETIC_AVG_CHARS_PER_DOC.get(name, 0)
                inflation_for_log = SYNTHETIC_DOC_INFLATION
            else:
                avg_for_log = _AVG_CHARS_PER_DOC.get(name, 0)
                inflation_for_log = _DOWNLOAD_INFLATION.get(name, 0)

            log.info(
                f"{name} cap derived from {target}: {cap:,} docs "
                f"(avg {avg_for_log:,} chars/doc, "
                f"{inflation_for_log}× inflation)"
            )

    # CC has different 'cap' semantics: max_segments, and needs crawls + workers.
    if name == "common_crawl":
        cfg = TARGET_CONFIGS[target]
        if mini:
            max_segments = MINI_OVERRIDES.get(name)
        else:
            max_segments = compute_cc_segments(cfg["corpus_tokens"])

        # Common Crawl is a two-stage pipeline:
        #   - download_workers are network-bound HTTPS WARC reads
        #   - extract_workers are CPU-bound WARC parsing + trafilatura + fastText
        # Derive both from the global worker count so CC scales with the same
        # CPU-aware policy as the rest of curation, while keeping sane caps for
        # very large machines and safe behavior on small 2–8 vCPU test boxes.
        cc_download_workers = min(32, max(1, math.ceil(workers / 3)))
        cc_extract_workers = max(1, workers - cc_download_workers)
        cc_in_flight = min(64, max(2, cc_download_workers * 2))

        # Full runs have enough WARC segments to benefit from the dynamic
        # worker split above. Mini runs usually process only a few segments,
        # so clamp worker pools to the amount of actual segment work available.
        # This avoids spawning dozens of idle workers for `curate-mini` while
        # preserving the higher-throughput defaults for full curation.
        if max_segments is not None:
            cc_download_workers = min(
                cc_download_workers,
                max(1, max_segments),
            )
            cc_extract_workers = min(
                cc_extract_workers,
                max(1, max_segments * 2),
            )
            cc_in_flight = min(
                cc_in_flight,
                max(2, max_segments * 2),
            )

        log.info(
            "Common Crawl worker config: "
            f"download_workers={cc_download_workers}, "
            f"extract_workers={cc_extract_workers}, "
            f"in_flight={cc_in_flight}"
        )

        return CommonCrawlSource(
            output_dir=raw_dir,
            crawls=cfg["cc_crawls"],
            max_segments=max_segments,
            download_workers=cc_download_workers,
            extract_workers=cc_extract_workers,
            in_flight=cc_in_flight,
        )

    if name == "fineweb":
        return FineWebSource(output_dir=raw_dir, max_docs=cap)
    if name == "fineweb_edu":
        # The 10BT sample is too small to absorb filtering/dedup headroom at
        # 350m and 1b. Keep small runs cheap and use the 100BT sample where
        # the requested retained share exceeds the small sample's safe range.
        edu_config = (
            "sample-100BT" if target in {"350m", "1b"} and not mini
            else "sample-10BT"
        )
        return FineWebEduSource(
            output_dir=raw_dir,
            config=edu_config,
            max_docs=cap,
        )
    if name == "wikipedia":
        return WikipediaSource(output_dir=raw_dir, max_docs=cap)
    if name == "pg19":
        char_cap = None if mini else _derive_max_chars(name, target)
        if char_cap is not None:
            log.info(
                f"pg19 char cap derived from {target}: {char_cap:,} chars "
                f"(target share with 1.30× buffer)"
            )
        return PG19Source(output_dir=raw_dir, max_docs=cap, max_chars=char_cap)
    if name == "pes2o":
        return PeS2oSource(output_dir=raw_dir, max_docs=cap)
    if name == "nemotron_cc_math":
        return NemotronCCMathSource(output_dir=raw_dir, max_docs=cap)

    if name == "nemotron_specialized":
        return NemotronSpecializedSource(output_dir=raw_dir, max_docs=cap)

    if name == "stackexchange":
        return StackExchangeSource(output_dir=raw_dir, max_docs=cap)
    if name == "synthetic_arithmetic":
        return SyntheticArithmeticSource(output_dir=raw_dir, max_docs=cap, max_chars=_derive_max_chars(name, target))
    if name == "synthetic_task_code":
        return SyntheticTaskCodeSource(output_dir=raw_dir, max_docs=cap, max_chars=_derive_max_chars(name, target))
    if name == "educational_qa_mcq_math":
        return EducationalQAMCQMathSource(output_dir=raw_dir, max_docs=cap, max_chars=_derive_max_chars(name, target))
    if name == "educational_qa_mcq_general":
        return EducationalQAMCQGeneralSource(output_dir=raw_dir, max_docs=cap, max_chars=_derive_max_chars(name, target))
    if name == "factual_restraint":
        return FactualRestraintSource(output_dir=raw_dir, max_docs=cap, max_chars=_derive_max_chars(name, target))
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


def stage_download(
    target: str,
    mini: bool = False,
    workers: int | None = None,
    sources: list[str] | None = None,
    force: bool = False,
) -> None:
    """Download each source into an isolated staging directory.

    A raw source is reusable only when its completion manifest matches the
    source contract. Downloads never append to an unverified final directory.
    This intentionally restarts an interrupted source rather than guessing a
    filtered-stream cursor from the number of output shards.
    """
    n_workers = workers or default_workers()
    log.info(f"=== Stage 1: Download (target={target}, mini={mini}) ===")

    selected_sources = sources or list(ALL_SOURCES)

    if not mini and "common_crawl" in selected_sources:
        cc_segments = compute_cc_segments(TARGET_CONFIGS[target]["corpus_tokens"])
        log.info(
            f"Common Crawl: computed {cc_segments} segments from "
            f"{TARGET_CONFIGS[target]['corpus_tokens']:,} tokens × "
            f"{_TOP_LEVEL_SHARE['common_crawl']:.2%} × {CHARS_PER_TOKEN} chars/tok "
            f"÷ {CC_CHARS_PER_SEGMENT:,} chars/segment"
        )

    for name in selected_sources:
        final_dir = RAW_DIR / name
        partial_root = RAW_DIR / ".partial"
        partial_dir = partial_root / name

        contract_source = _build_source(
            name,
            mini=mini,
            target=target,
            workers=n_workers,
            raw_root=RAW_DIR,
        )
        dataset_name = (
            getattr(contract_source, "DATASET_NAME", None)
            or getattr(contract_source, "HF_REPO", None)
        )
        dataset_revision = (
            resolve_dataset_revision(dataset_name) if dataset_name else None
        )
        contract = {
            "source": name,
            "target": target,
            "mini": mini,
            "implementation": (
                f"{contract_source.__class__.__module__}."
                f"{contract_source.__class__.__qualname__}"
            ),
            "implementation_sha256": code_fingerprint(
                contract_source.__class__,
                resolve_dataset_revision,
            ),
            "settings": {
                key: value
                for key, value in vars(contract_source).items()
                if key not in {"output_dir", "tmp_dir"}
            },
            "dataset": dataset_name,
            "dataset_revision": dataset_revision,
            "dataset_config": getattr(
                contract_source,
                "DATASET_CONFIG",
                getattr(contract_source, "CONFIG_NAME", None),
            ),
        }

        if manifest_matches(
            final_dir,
            stage="download",
            contract=contract,
            input_signature=None,
        ):
            log.info(f"{name}: verified raw manifest matches — reusing")
            continue

        if final_dir.exists() and any(final_dir.iterdir()) and not force:
            raise RuntimeError(
                f"{name}: raw output exists without a matching completion "
                f"manifest: {final_dir}. Re-run with --force to rebuild this "
                f"source safely; existing data is left untouched."
            )

        if partial_dir.exists():
            log.warning(f"{name}: removing incomplete staging directory {partial_dir}")
            shutil.rmtree(partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Downloading {name} into isolated staging...")
        source = _build_source(
            name,
            mini=mini,
            target=target,
            workers=n_workers,
            raw_root=partial_root,
        )
        output_files = source.download()
        shards = sorted(partial_dir.glob("*.jsonl"))
        if not shards:
            raise RuntimeError(
                f"{name}: download returned without producing JSONL shards "
                f"(reported {len(output_files or [])} output files)"
            )

        write_manifest(
            partial_dir,
            stage="download",
            contract=contract,
            input_signature=None,
        )

        backup_dir = RAW_DIR / f".{name}.previous"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if final_dir.exists():
            final_dir.replace(backup_dir)
        try:
            partial_dir.replace(final_dir)
        except Exception:
            if backup_dir.exists() and not final_dir.exists():
                backup_dir.replace(final_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


# ── Stage 2: Filter ────────────────────────────────────────────────────────────

_worker_qf: QualityFilter | None = None


def _init_filter_worker() -> None:
    """Pool initializer: construct QualityFilter once per subprocess."""
    global _worker_qf
    _worker_qf = QualityFilter()


def _filter_shard(args: tuple[Path, Path, str]) -> tuple[str, str, dict]:
    """Filter a single JSONL shard. Runs in a subprocess."""
    shard, dst_dir, source = args
    out_path = dst_dir / shard.name
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")

    qf = _worker_qf or QualityFilter()
    qf.reset_stats()
    parse_errors = 0
    input_documents = 0
    segmented_input_documents = 0
    produced_segments = 0
    try:
        with open(shard, "rb", buffering=8 * 1024 * 1024) as fin, \
             open(tmp_path, "wb", buffering=8 * 1024 * 1024) as fout:
            for line in fin:
                try:
                    record = orjson.loads(line)
                except Exception:
                    parse_errors += 1
                    continue
                input_documents += 1
                records = segment_long_document(
                    record,
                    eligible_sources=qf.config.long_document_segment_sources,
                    max_chars=qf.config.max_chars,
                    min_chars=qf.config.min_chars,
                )
                produced_segments += len(records)
                segmented_input_documents += int(len(records) > 1)
                for candidate in records:
                    kept, _ = qf.check(candidate, expected_source=source)
                    if kept:
                        fout.write(orjson.dumps(candidate))
                        fout.write(b"\n")
            fout.flush()
            os.fsync(fout.fileno())
        if parse_errors:
            raise RuntimeError(
                f"{shard}: encountered {parse_errors:,} invalid JSONL records"
            )
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    stats = qf.stats_snapshot()
    if stats["total"] != produced_segments:
        raise RuntimeError(f"{shard}: segmentation/filter count mismatch")
    stats.update({
        "input_documents": input_documents,
        "segmented_input_documents": segmented_input_documents,
        "produced_segments": produced_segments,
    })
    return source, shard.name, stats


def _merge_filter_stats(aggregate: dict, shard_stats: dict) -> None:
    """Merge one worker's filter counts into a per-source audit summary."""
    aggregate["shards"] += 1
    for field in (
        "input_documents",
        "segmented_input_documents",
        "produced_segments",
    ):
        aggregate[field] += int(shard_stats[field])
    for field in ("total", "kept", "rejected", "fasttext_prediction_errors"):
        aggregate[field] += int(shard_stats[field])
    for reason, count in shard_stats["rejection_reasons"].items():
        aggregate["rejection_reasons"][reason] = (
            aggregate["rejection_reasons"].get(reason, 0) + int(count)
        )


def stage_filter(workers: int | None = None, sources: list[str] | None = None) -> None:
    """Apply quality filters to all raw data in parallel."""
    n_workers = workers or default_workers()
    log.info(f"=== Stage 2: Quality Filter ({n_workers} workers) ===")

    selected_sources = sources or list(ALL_SOURCES)

    fasttext_path = Path(os.environ.get("DATA_DIR", "data")) / "models" / "lid.176.ftz"
    prose_sources = [
        source for source in selected_sources
        if source not in PROSE_HEURISTIC_SKIP_SOURCES
    ]
    if prose_sources and not fasttext_path.exists():
        raise RuntimeError(
            f"FastText language model is required for prose filtering but "
            f"was not found at {fasttext_path}. Run "
            f"'make download-fasttext-model' before filtering."
        )
    if prose_sources:
        require_fasttext_model()

    filter_contract = {
        "implementation_sha256": code_fingerprint(
            QualityFilter,
            segment_long_document,
        ),
        "audit_schema_version": 2,
        "source_families": FILTER_SOURCE_FAMILIES,
        "quality_config": vars(QualityFilter().config),
        "fasttext_model": (
            file_snapshot([fasttext_path], root=fasttext_path.parent)[0]
            if fasttext_path.exists()
            else None
        ),
    }
    all_work: list[tuple[Path, Path, str]] = []
    pending_sources: dict[str, tuple[Path, str]] = {}
    filter_stats: dict[str, dict] = {}
    for source in selected_sources:
        src_dir = RAW_DIR / source
        dst_dir = FILTERED_DIR / source

        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            raise RuntimeError(f"Required raw source has no shards: {src_dir}")
        if not manifest_outputs_match(src_dir):
            raise RuntimeError(
                f"Required raw source is not manifest-complete: {src_dir}"
            )
        input_signature = tree_signature(src_dir)
        source_contract = {**filter_contract, "source": source}
        if manifest_matches(
            dst_dir,
            stage="filter",
            contract=source_contract,
            input_signature=input_signature,
        ):
            log.info(f"  {source}: verified filtered manifest matches — reusing")
            continue

        if dst_dir.exists():
            log.warning(f"  {source}: replacing stale/incomplete filtered output")
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"  {source}: {len(shards)} shards queued")
        all_work.extend((shard, dst_dir, source) for shard in shards)
        pending_sources[source] = (dst_dir, input_signature)
        filter_stats[source] = {
            "schema_version": 2,
            "stage": "filter",
            "source": source,
            "source_family": source_filter_family(source),
            "shards": 0,
            "input_documents": 0,
            "segmented_input_documents": 0,
            "produced_segments": 0,
            "total": 0,
            "kept": 0,
            "rejected": 0,
            "rejection_reasons": {},
            "fasttext_prediction_errors": 0,
        }

    if not all_work:
        log.info("All selected sources have verified filtered outputs")
        return

    # Sort largest-first so stragglers don't tail the run
    all_work.sort(key=lambda p: p[0].stat().st_size, reverse=True)

    log.info(f"Filtering {len(all_work)} shards with {n_workers} workers...")
    processed = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_filter_worker,
    ) as executor:
        for source, shard_name, shard_stats in executor.map(
            _filter_shard, all_work, chunksize=16
        ):
            processed += 1
            _merge_filter_stats(filter_stats[source], shard_stats)
            log.debug(
                f"Filtered {source}/{shard_name}: "
                f"kept={shard_stats['kept']:,}/{shard_stats['total']:,}"
            )

    for source, (dst_dir, input_signature) in pending_sources.items():
        stats = filter_stats[source]
        if stats["total"] != stats["kept"] + stats["rejected"]:
            raise RuntimeError(f"{source}: inconsistent aggregate filter counts")
        if stats["total"] != stats["produced_segments"]:
            raise RuntimeError(f"{source}: inconsistent segmentation counts")
        stats["rejection_reasons"] = dict(
            sorted(stats["rejection_reasons"].items())
        )
        write_manifest(
            dst_dir,
            stage="filter",
            contract={**filter_contract, "source": source},
            input_signature=input_signature,
            metadata={"audit": stats},
        )

    log.info(f"Filter complete — processed: {processed}")


# ── Stage 3: Deduplicate ───────────────────────────────────────────────────────

def stage_dedup(workers: int | None = None, sources: list[str] | None = None) -> None:
    n_workers = workers or default_workers()
    log.info(f"=== Stage 3: Deduplication ({n_workers} workers) ===")

    working_dir = DEDUP_SCRATCH_DIR
    dedup = Deduplicator(working_dir=working_dir, workers=n_workers)

    selected_set = set(sources or ALL_SOURCES)
    if set(DEDUP_PRIORITY) != set(ALL_SOURCES):
        missing = sorted(set(ALL_SOURCES) - set(DEDUP_PRIORITY))
        unknown = sorted(set(DEDUP_PRIORITY) - set(ALL_SOURCES))
        raise RuntimeError(
            f"DEDUP_PRIORITY drifted from ALL_SOURCES; "
            f"missing={missing}, unknown={unknown}"
        )
    selected_sources = [
        source for source in DEDUP_PRIORITY if source in selected_set
    ]

    prior_dedup_signatures: list[dict[str, str]] = []
    for source in selected_sources:
        src_dir = FILTERED_DIR / source
        dst_dir = FILTERED_DIR / f"{source}_deduped"

        shards = sorted(src_dir.glob("*.jsonl"))
        if not shards:
            raise RuntimeError(f"Required filtered source has no shards: {src_dir}")
        if not manifest_outputs_match(src_dir):
            raise RuntimeError(
                f"Required filtered source is not manifest-complete: {src_dir}"
            )

        input_signature = tree_signature(src_dir)
        contract = {
            "source": source,
            "implementation_sha256": code_fingerprint(Deduplicator),
            "audit_schema_version": 1,
            "exact_hash": "sha256-normalized-prefix-128",
            "cross_source_priority": DEDUP_PRIORITY,
            "prior_exact_inputs": prior_dedup_signatures,
            "fuzzy_enabled": source not in SKIP_FUZZY_DEDUP_SOURCES,
            "fuzzy_partition_field": FUZZY_DEDUP_PARTITION_FIELDS.get(source),
            "minhash": {
                **MINHASH_CONTRACT,
                "lsh_probability_50pct": MINHASH_LSH_CROSSOVER,
            },
        }

        if manifest_matches(
            dst_dir,
            stage="dedup",
            contract=contract,
            input_signature=input_signature,
        ):
            indexed = dedup.index_source_input(src_dir)
            log.info(
                f"  {source}: verified dedup manifest matches — reusing and "
                f"indexed {indexed:,} retained documents"
            )
            prior_dedup_signatures.append(
                {
                    "source": source,
                    "input_signature": input_signature,
                }
            )
            continue

        if dst_dir.exists():
            log.warning(f"  {source}: replacing stale/incomplete dedup output")
            shutil.rmtree(dst_dir)
        scratch_dir = working_dir / source
        if scratch_dir.exists():
            log.warning(f"  {source}: removing stale dedup scratch")
            shutil.rmtree(scratch_dir)

        if source in SKIP_FUZZY_DEDUP_SOURCES:
            log.info(
                f"  {source}: running exact dedup; skipping fuzzy MinHash dedup"
            )
            dedup.deduplicate_source_exact_only(
                src_dir=src_dir,
                dst_dir=dst_dir,
                source_name=source,
            )
        elif source in FUZZY_DEDUP_PARTITION_FIELDS:
            dedup.deduplicate_source_by_partition(
                src_dir=src_dir,
                dst_dir=dst_dir,
                source_name=source,
                partition_field=FUZZY_DEDUP_PARTITION_FIELDS[source],
            )
        else:
            dedup.deduplicate_source(
                src_dir=src_dir,
                dst_dir=dst_dir,
                source_name=source,
            )

        write_manifest(
            dst_dir,
            stage="dedup",
            contract=contract,
            input_signature=input_signature,
            metadata={
                "audit": {
                    "schema_version": 1,
                    "stage": "dedup",
                    **dedup.stats_for(source),
                }
            },
        )
        prior_dedup_signatures.append(
            {
                "source": source,
                "input_signature": input_signature,
            }
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
                    except Exception as exc:
                        raise RuntimeError(
                            f"Invalid deduplicated JSONL record in {shard}"
                        ) from exc
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
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid blend staging JSONL record in {staging_path}"
                ) from exc
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
                    except Exception as exc:
                        raise RuntimeError(
                            f"Invalid deduplicated JSONL record in {shard}"
                        ) from exc
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


def _append_overflow_to_source(
    overflow_source: str,
    requested_chars: int,
    source_dirs: dict[str, Path],
    staging_paths: dict[str, Path],
    source_stats: dict[str, dict],
) -> tuple[int, int]:
    # Append additional docs from one overflow source into its staging file.
    if requested_chars <= 0:
        return 0, 0

    if overflow_source not in staging_paths:
        log.warning(
            f"Overflow source {overflow_source!r} unavailable; "
            f"cannot cover {requested_chars / 1e9:.3f}B chars"
        )
        return 0, 0

    src_dir = source_dirs.get(overflow_source)
    if src_dir is None:
        log.warning(
            f"Overflow source {overflow_source!r} has no source directory; "
            f"cannot cover {requested_chars / 1e9:.3f}B chars"
        )
        return 0, 0

    docs, chars = _append_overflow(
        (str(src_dir), str(staging_paths[overflow_source]), requested_chars)
    )

    source_stats[overflow_source]["docs"] += docs
    source_stats[overflow_source]["chars"] += chars
    source_stats[overflow_source]["overflow_docs"] = (
        source_stats[overflow_source].get("overflow_docs", 0) + docs
    )
    source_stats[overflow_source]["overflow_chars"] = (
        source_stats[overflow_source].get("overflow_chars", 0) + chars
    )

    log.info(
        f"  {overflow_source} overflow: +{docs:,} docs, "
        f"+{chars / 1e9:.3f}B chars "
        f"(requested {requested_chars / 1e9:.3f}B)"
    )
    return docs, chars


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
    interleave_block_lines: int = 1_024,
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
               its remaining line count, then read a small block from that
               source. The train chunk is shuffled before it is written, so
               1,024-line scheduling blocks preserve mixed chunks while
               avoiding one Python weighted-choice call per document.

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
        block_size = min(
            interleave_block_lines,
            max(1, remaining[chosen]),
        )
        exhausted = False
        for _ in range(block_size):
            line = handles[chosen].readline()
            if not line:
                exhausted = True
                break

            remaining[chosen] = max(0, remaining[chosen] - 1)
            seen += 1

            # Reservoir sampling (Vitter Algorithm R) remains per-document,
            # so every record has exactly n_val / total_lines inclusion
            # probability regardless of scheduling block size.
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

        if exhausted:
            handles[chosen].close()
            try:
                staging_paths[chosen].unlink()
            except FileNotFoundError:
                pass
            active.remove(chosen)

    _flush_chunk()

    if seen != total_lines:
        raise RuntimeError(
            f"Blend staging line-count mismatch: read {seen:,}, expected "
            f"{total_lines:,}. Refusing to emit a biased validation split."
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

    Pass 2 (sequential): deficits route through source-aware overflow chains.
                         Local synthetic deficits use Nemotron Specialized
                         first; general deficits use FineWeb-Edu first.

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
    total_tokens = cfg["corpus_tokens"]
    val_fraction = cfg.get("val_fraction", PRETRAIN_VAL_FRACTION)

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = CURATED_DIR / "train.jsonl"
    val_path = CURATED_DIR / "val.jsonl"

    source_dirs = {
        source: FILTERED_DIR / f"{source}_deduped"
        for source in ALL_SOURCES
    }

    # Initial character targets from the locked mix.
    target_chars = compute_source_char_targets(target)
    missing_sources = [
        source
        for source, source_dir in source_dirs.items()
        if not list(source_dir.glob("*.jsonl"))
        or not manifest_outputs_match(source_dir)
    ]
    if missing_sources:
        raise RuntimeError(
            "Cannot blend without complete deduplicated outputs for every "
            f"configured source: {', '.join(missing_sources)}"
        )

    input_signature = stable_digest(
        {
            source: tree_signature(source_dir)
            for source, source_dir in sorted(source_dirs.items())
        }
    )
    blend_contract = {
        "target": target,
        "implementation_sha256": code_fingerprint(
            stage_blend,
            audit_exact_split_overlap,
            build_benchmark_index,
            BenchmarkContaminationAuditor,
            audit_minhash_split_overlap,
            build_cross_index_removals,
            JsonlByteRangeReader,
            NearOverlapReporter,
            SensitiveContentAuditor,
        ),
        "target_tokens": total_tokens,
        "seed": seed,
        "val_fraction": val_fraction,
        "chars_per_token_estimate": CHARS_PER_TOKEN,
        "source_targets_chars": target_chars,
        "data_mix": DATA_MIX,
        "code_submix": CODE_SUBMIX,
        "overflow_sink": OVERFLOW_SINK,
        "benchmark_decontamination": benchmark_decontamination_contract(),
        "near_overlap": {
            "algorithm": "datatrove_minhash_validation_index_lsh",
            "minhash": {
                **MINHASH_CONTRACT,
                "lsh_probability_50pct": MINHASH_LSH_CROSSOVER,
            },
        },
        "sensitive_content": SENSITIVE_CONTENT_CONTRACT,
    }
    if manifest_matches(
        CURATED_DIR,
        stage="blend",
        contract=blend_contract,
        input_signature=input_signature,
        output_pattern="*.json*",
    ):
        log.info("Verified blend manifest matches inputs/configuration — reusing")
        return

    # Resolve the small, pinned benchmark inputs before replacing any existing
    # blend output. Network/cache failure therefore cannot destroy a previously
    # completed corpus. The index remains in memory only for this stage.
    benchmark_index = build_benchmark_index()

    for stale in (
        train_path,
        val_path,
        CURATED_DIR / "blend_stats.json",
        CURATED_DIR / "exact_overlap_report.json",
        CURATED_DIR / "benchmark_contamination_report.json",
        CURATED_DIR / "near_overlap_report.json",
        CURATED_DIR / "sensitive_content_report.json",
        CURATED_DIR / MANIFEST_NAME,
    ):
        stale.unlink(missing_ok=True)

    rng = random.Random(seed)

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
            raise RuntimeError(f"  {source}: no deduped shards")
        staging = CURATED_DIR / f"blend_{source}.jsonl"
        work.append((source, str(src_dir), str(staging), target_chars[source]))

    if not work:
        raise RuntimeError("No deduped shards found for any source")

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
                "initial_deficit": max(0, target_chars[source] - chars),
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

    # Futures complete nondeterministically. Normalize mapping order before
    # overflow and seeded shuffling so identical inputs produce identical
    # outputs.
    staging_paths = {
        source: staging_paths[source] for source in ALL_SOURCES
    }
    source_stats = {
        source: source_stats[source] for source in ALL_SOURCES
    }

    # ── Pass 2: source-aware overflow ─────────────────────────────────────────
    overflow_chains = {
        source: SYNTHETIC_OVERFLOW_CHAIN for source in SYNTHETIC_SOURCES
    }
    for source in CODE_SOURCES:
        overflow_chains[source] = ("stack_v1", "fineweb_edu", OVERFLOW_SINK)

    total_deficit = sum(s["deficit"] for s in source_stats.values())
    if total_deficit > 0:
        log.info(
            f"Pass 2/3: source-aware overflow — covering "
            f"{total_deficit / 1e9:.3f}B character deficit..."
        )

    for source in ALL_SOURCES:
        stats = source_stats[source]
        remaining_deficit = stats.get("deficit", 0)
        if remaining_deficit <= 0:
            continue

        chain = overflow_chains.get(source, DEFAULT_OVERFLOW_CHAIN)
        log.info(
            f"  {source}: routing {remaining_deficit / 1e9:.3f}B "
            f"deficit via {' -> '.join(chain)}"
        )
        for overflow_source in chain:
            if remaining_deficit <= 0:
                break
            if overflow_source == source:
                continue
            docs, chars = _append_overflow_to_source(
                overflow_source=overflow_source,
                requested_chars=remaining_deficit,
                source_dirs=source_dirs,
                staging_paths=staging_paths,
                source_stats=source_stats,
            )
            remaining_deficit = max(0, remaining_deficit - chars)

        if remaining_deficit > 0:
            log.error(
                f"  {source}: unresolved deficit after overflow: "
                f"{remaining_deficit / 1e9:.3f}B chars"
            )
        stats["deficit"] = remaining_deficit

    unresolved_deficit = sum(
        stats.get("deficit", 0) for stats in source_stats.values()
    )
    if unresolved_deficit:
        raise RuntimeError(
            f"Blend cannot satisfy the configured corpus target; "
            f"{unresolved_deficit:,} characters remain unresolved"
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

    train_tmp = train_path.with_name(f".{train_path.name}.{os.getpid()}.tmp")
    val_tmp = val_path.with_name(f".{val_path.name}.{os.getpid()}.tmp")
    try:
        if effective_ram_gb < SHUFFLE_RAM_BUDGET_GB:
            n_train, n_val, val_source_counts = _shuffle_in_memory(
                staging_paths, train_tmp, val_tmp, val_fraction, rng,
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
                staging_paths, train_tmp, val_tmp, val_fraction, rng,
                total_lines=total_lines_calc,
                source_doc_counts=source_doc_counts,
            )
        train_tmp.replace(train_path)
        val_tmp.replace(val_path)
    except Exception:
        train_tmp.unlink(missing_ok=True)
        val_tmp.unlink(missing_ok=True)
        raise

    total_lines = n_train + n_val
    total_chars = sum(s["chars"] for s in source_stats.values())
    log.info(
        f"Blend complete — {total_lines:,} documents total "
        f"({n_train:,} train + {n_val:,} val), "
        f"~{total_chars // CHARS_PER_TOKEN / 1e9:.2f}B tokens "
        f"(target {total_tokens / 1e9:.2f}B)"
    )

    # Full-corpus exact split audit. Use the same normalized hash contract as
    # deduplication, retain only the much smaller validation hash set in RAM,
    # and persist the report even when the gate fails.
    benchmark_auditor = BenchmarkContaminationAuditor(benchmark_index)
    sensitive_content_auditor = SensitiveContentAuditor()

    def observe_finalized_record(
        split: str, line_number: int, record: dict
    ) -> None:
        benchmark_auditor.observe(split, line_number, record)
        sensitive_content_auditor.observe(split, line_number, record)

    overlap_report = audit_exact_split_overlap(
        train_path,
        val_path,
        record_observer=observe_finalized_record,
    )
    overlap_report["expected_train_documents"] = n_train
    overlap_report["expected_validation_documents"] = n_val
    overlap_report["split_counts_match"] = (
        overlap_report["train_documents"] == n_train
        and overlap_report["validation_documents"] == n_val
    )
    overlap_report["passed"] = (
        overlap_report["passed"] and overlap_report["split_counts_match"]
    )
    overlap_report_path = CURATED_DIR / "exact_overlap_report.json"
    atomic_write_json(overlap_report_path, overlap_report)

    benchmark_report = benchmark_auditor.report({
        "train": overlap_report["train_documents"],
        "validation": overlap_report["validation_documents"],
    })
    benchmark_report["split_counts_match"] = overlap_report["split_counts_match"]
    benchmark_report["passed"] = (
        benchmark_report["passed"] and benchmark_report["split_counts_match"]
    )
    benchmark_report_path = CURATED_DIR / "benchmark_contamination_report.json"
    atomic_write_json(benchmark_report_path, benchmark_report)

    sensitive_content_report = sensitive_content_auditor.report({
        "train": overlap_report["train_documents"],
        "validation": overlap_report["validation_documents"],
    })
    sensitive_content_report_path = (
        CURATED_DIR / "sensitive_content_report.json"
    )
    atomic_write_json(
        sensitive_content_report_path, sensitive_content_report
    )

    # The shared scan is complete, so persist all reports before enforcing
    # scan-integrity and contamination gates. One failure must not hide the
    # completed results of the other observers.
    if not overlap_report["passed"]:
        raise RuntimeError(
            "Exact split-overlap gate failed: "
            f"validation_duplicate_hashes="
            f"{overlap_report['validation_duplicate_hashes']:,}, "
            f"train_validation_overlap_hashes="
            f"{overlap_report['train_validation_overlap_hashes']:,}. "
            f"split_counts_match={overlap_report['split_counts_match']}. "
            f"See {overlap_report_path}; benchmark audit: "
            f"{benchmark_report_path}; sensitive-content audit: "
            f"{sensitive_content_report_path}."
        )
    if not benchmark_report["passed"]:
        raise RuntimeError(
            "Benchmark-contamination gate failed: "
            f"matched_documents={benchmark_report['matched_documents']:,}, "
            f"matched_unique_queries="
            f"{benchmark_report['matched_unique_queries']:,}, "
            f"ngram_matched_documents="
            f"{benchmark_report['ngram_matched_documents']:,}, "
            f"split_counts_match={benchmark_report['split_counts_match']}. "
            f"See {benchmark_report_path}."
        )
    if not sensitive_content_report["passed"]:
        raise RuntimeError(
            "Sensitive-content audit failed to scan every finalized record: "
            f"split_counts_match="
            f"{sensitive_content_report['split_counts_match']}. "
            f"See {sensitive_content_report_path}."
        )

    near_overlap_working = DEDUP_SCRATCH_DIR / "split_near_overlap"
    near_overlap_report = audit_minhash_split_overlap(
        train_path,
        val_path,
        near_overlap_working,
        workers=workers,
    )
    near_overlap_report["validation_documents"] = overlap_report[
        "validation_documents"
    ]
    near_overlap_report["expected_train_documents"] = n_train
    near_overlap_report["expected_validation_documents"] = n_val
    near_overlap_report["split_counts_match"] = (
        near_overlap_report["train_documents"] == n_train
        and near_overlap_report["validation_documents"] == n_val
    )
    near_overlap_report["passed"] = (
        near_overlap_report["passed"]
        and near_overlap_report["split_counts_match"]
    )
    near_overlap_report_path = CURATED_DIR / "near_overlap_report.json"
    atomic_write_json(near_overlap_report_path, near_overlap_report)
    shutil.rmtree(near_overlap_working, ignore_errors=True)
    if not near_overlap_report["passed"]:
        raise RuntimeError(
            "Train/validation MinHash near-overlap gate failed: "
            f"matched_train_documents="
            f"{near_overlap_report['matched_train_documents']:,}, "
            f"split_counts_match={near_overlap_report['split_counts_match']}. "
            f"See {near_overlap_report_path}."
        )

    # ── Write blend stats ──────────────────────────────────────────────────────
    stats_path = CURATED_DIR / "blend_stats.json"
    atomic_write_json(
        stats_path,
        {
            "target": target,
            "target_tokens": total_tokens,
            "chars_per_token": CHARS_PER_TOKEN,
            "total_documents": total_lines,
            "train_documents": n_train,
            "val_documents": n_val,
            "val_fraction": val_fraction,
            "estimated_tokens_from_chars": total_chars // CHARS_PER_TOKEN,
            "exact_overlap_audit": overlap_report,
            "benchmark_contamination_audit": benchmark_report,
            "sensitive_content_audit": sensitive_content_report,
            "near_overlap_audit": near_overlap_report,
            "token_count_status": (
                "estimate_only; authoritative realized token counts are "
                "written by pretrain/data/tokenize_data.py"
            ),
            "source_mix": {
                s: {
                    "docs": v["docs"],
                    "chars": v["chars"],
                    "target_chars": v["target_chars"],
                    "initial_deficit": v["initial_deficit"],
                    "unresolved_deficit": v["deficit"],
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
        },
    )
    write_manifest(
        CURATED_DIR,
        stage="blend",
        contract=blend_contract,
        input_signature=input_signature,
        output_pattern="*.json*",
    )
    log.info(f"Blend stats → {stats_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

STAGES = ["download", "filter", "dedup", "blend", "stats", "all"]


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
        "--sources",
        default=None,
        help=(
            "Comma-separated concrete source names for source-scoped capacity "
            "runs. Example: --sources nemotron_cc_math,nemotron_specialized. "
            "When used with --stage all, only download/filter/dedup/stats run; "
            "blend is intentionally skipped."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers for filter/dedup/blend. Default: cpu_count - 2.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow download stage to replace raw source directories whose "
            "completion manifest is missing or stale. Replacement occurs "
            "only after a new staged download completes."
        ),
    )
    args = parser.parse_args()

    if args.mini and args.target != "mini":
        log.warning(
            f"--mini set but --target is '{args.target}'. "
            f"Consider --target mini."
        )

    configure_data_dirs(args.target)

    try:
        selected_sources = _resolve_sources(args.sources)
    except ValueError as exc:
        parser.error(str(exc))

    source_scoped = args.sources is not None

    if source_scoped and args.stage == "blend":
        parser.error("--sources is for source capacity runs; do not use it with blend")

    n_workers = args.workers or default_workers()
    log.info(
        f"SLM Curation — "
        f"target={args.target}, stage={args.stage}, "
        f"mini={args.mini}, workers={n_workers} (cpu_count={os.cpu_count()}), "
        f"data_dir={DATA_DIR}, "
        f"sources={','.join(selected_sources) if source_scoped else 'all'}"
    )

    if args.stage in ("download", "all"):
        stage_download(
            args.target,
            mini=args.mini,
            workers=n_workers,
            sources=selected_sources,
            force=args.force,
        )
    if args.stage in ("filter", "all"):
        stage_filter(workers=n_workers, sources=selected_sources)
    if args.stage in ("dedup", "all"):
        stage_dedup(workers=n_workers, sources=selected_sources)

    if source_scoped:
        if args.stage in ("all", "stats"):
            stage_source_stats(selected_sources)
        if args.stage == "all":
            log.info(
                "Source-scoped --stage all complete after download/filter/dedup/stats; "
                "blend intentionally skipped."
            )
    else:
        if args.stage in ("blend", "all"):
            stage_blend(args.target, seed=args.seed, workers=n_workers)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
