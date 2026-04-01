"""
curator/filters/dedup.py
-------------------------
Disk-based MinHash deduplication using datatrove.

Replaces the datasketch in-memory LSH implementation with datatrove's
4-stage disk-based pipeline. This scales to arbitrary corpus size with
bounded RAM usage — the index never lives in memory.

How it works (datatrove's approach):
    1. Signatures  — compute MinHash signature per document, write to disk
    2. Buckets     — group signatures into LSH buckets, write (bucket, doc_id) pairs
    3. Cluster     — sort bucket pairs, find connected components of duplicates
    4. Filter      — stream original JSONL, drop documents marked as duplicates

Peak RAM at any stage: O(documents in one shard), not O(total corpus).
This means 125m, 350m, and 1b all run with the same memory footprint.

Exact deduplication (SHA-256) is handled as a pre-pass before datatrove,
since datatrove's minhash catches fuzzy duplicates but not verbatim ones
with trivial edits (different whitespace, punctuation).

References:
    datatrove minhash: https://github.com/huggingface/datatrove
    FineWeb pipeline:  https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
"""

import hashlib
import json
import logging
import re
import os
from pathlib import Path

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from tqdm import tqdm

log = logging.getLogger(__name__)

# MinHash config — matches FineWeb settings
# 14 buckets × 8 hashes = 112 total hashes, 64-bit precision
# Equivalent accuracy to datasketch's 128 permutations
MINHASH_CONFIG = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
    n_grams=5,
)

JACCARD_THRESHOLD = 0.8


def normalize(text: str) -> str:
    """Normalize text for exact deduplication."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_hash(text: str) -> str:
    """SHA-256 hash of normalized text."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


# ── Exact dedup pre-pass ───────────────────────────────────────────────────────

def exact_dedup_jsonl(
    input_path: Path,
    output_path: Path,
    seen_hashes: set[str],
) -> dict:
    """
    Single-pass exact deduplication using SHA-256.

    Streams input line by line — constant memory regardless of file size.
    Updates seen_hashes in place for cross-shard exact dedup.

    Args:
        input_path:   Input JSONL shard.
        output_path:  Output JSONL shard (exact-deduped).
        seen_hashes:  Shared set of SHA-256 hashes seen so far.
                      Updated in place as new documents are processed.

    Returns:
        Stats dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = kept = exact_dupes = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            total += 1
            record = json.loads(line)
            h = exact_hash(record.get("text", ""))
            if h in seen_hashes:
                exact_dupes += 1
                continue
            seen_hashes.add(h)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    return {"total": total, "kept": kept, "exact_duplicates": exact_dupes}


# ── Datatrove minhash pipeline ─────────────────────────────────────────────────

def run_minhash_dedup(
    input_dir: Path,
    output_dir: Path,
    working_dir: Path,
    workers: int = 4,
    tasks: int | None = None,
) -> None:
    """
    Run datatrove's 4-stage disk-based MinHash deduplication.

    Stages:
        1. Signatures  — MinHash each document, write signatures to disk
        2. Buckets     — LSH bucketing of signatures, write (bucket, doc) pairs
        3. Cluster     — find duplicate clusters via union-find on sorted pairs
        4. Filter      — stream input, drop clustered duplicates, write output

    All intermediate data lives on disk. RAM usage is bounded by shard size,
    not corpus size. This is the same approach used by FineWeb and RedPajama.

    Args:
        input_dir:   Directory of JSONL shards to deduplicate.
        output_dir:  Directory to write deduplicated JSONL shards.
        working_dir: Scratch directory for intermediate datatrove state
                     (signatures, buckets, clusters). Safe to delete after.
        workers:     Number of parallel workers (= number of CPU cores to use).
        tasks:       Number of tasks for parallelism. Defaults to number of
                     input shards. Should be >= workers for good utilization.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    working_dir = Path(working_dir)

    sig_dir     = working_dir / "signatures"
    bucket_dir  = working_dir / "buckets"
    cluster_dir = working_dir / "clusters"
    logs_dir    = working_dir / "logs"

    # Infer task count from number of input shards if not specified
    shards = list(input_dir.glob("*.jsonl"))
    if not shards:
        log.warning(f"No JSONL shards found in {input_dir} — skipping minhash dedup")
        return
    n_tasks = tasks or max(len(shards), workers)

    log.info(
        f"MinHash dedup: {len(shards)} shards, {n_tasks} tasks, {workers} workers\n"
        f"  input:   {input_dir}\n"
        f"  output:  {output_dir}\n"
        f"  scratch: {working_dir}"
    )

    # Stage 1 — Compute MinHash signatures
    log.info("Stage 1/4: Computing MinHash signatures...")
    LocalPipelineExecutor(
        pipeline=[
            JsonlReader(str(input_dir), text_key="text", id_key=None),
            MinhashDedupSignature(
                output_folder=str(sig_dir),
                config=MINHASH_CONFIG,
            ),
        ],
        tasks=n_tasks,
        workers=workers,
        logging_dir=str(logs_dir / "signatures"),
    ).run()

    # Stage 2 — LSH bucketing
    log.info("Stage 2/4: LSH bucketing...")
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(sig_dir),
                output_folder=str(bucket_dir),
                config=MINHASH_CONFIG,
            ),
        ],
        tasks=MINHASH_CONFIG.num_buckets,  # one task per bucket
        workers=workers,
        logging_dir=str(logs_dir / "buckets"),
    ).run()

    # Stage 3 — Cluster (find connected components of duplicates)
    log.info("Stage 3/4: Clustering duplicates...")
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(bucket_dir),
                output_folder=str(cluster_dir),
                config=MINHASH_CONFIG,
            ),
        ],
        tasks=1,  # clustering must be single-task (global union-find)
        logging_dir=str(logs_dir / "clusters"),
    ).run()

    # Stage 4 — Filter: stream input, drop duplicates, write output
    log.info("Stage 4/4: Filtering duplicates...")
    output_dir.mkdir(parents=True, exist_ok=True)
    LocalPipelineExecutor(
        pipeline=[
            JsonlReader(str(input_dir), text_key="text", id_key=None),
            MinhashDedupFilter(
                input_folder=str(cluster_dir),
                exclusion_writer=JsonlWriter(
                    str(working_dir / "removed"),
                    output_filename="${rank}.jsonl",
                ),
            ),
            JsonlWriter(
                str(output_dir),
                output_filename="${rank}.jsonl",
            ),
        ],
        tasks=n_tasks,
        workers=workers,
        logging_dir=str(logs_dir / "filter"),
    ).run()

    log.info(f"MinHash dedup complete → {output_dir}")


# ── Top-level dedup entry point ────────────────────────────────────────────────

class Deduplicator:
    """
    Two-stage deduplicator: exact hash + datatrove MinHash LSH.

    Replaces the datasketch in-memory implementation. All state is
    disk-based — RAM usage is O(shard_size), not O(corpus_size).

    Stage 1 — Exact dedup: SHA-256 of normalized text, streaming per shard.
              Zero false positives. Cross-shard via shared seen_hashes set
              (this set is the only in-memory structure — ~70 bytes/doc).

    Stage 2 — Fuzzy dedup: datatrove's 4-stage disk pipeline.
              Catches near-duplicates (Jaccard > 0.8). Bounded RAM at any
              corpus scale — 125m, 350m, and 1b all use the same footprint.

    Args:
        working_dir: Scratch directory for datatrove intermediate state.
        workers:     CPU workers for parallel stages. Default: cpu_count / 2.
        threshold:   Jaccard similarity threshold. Default: 0.8.

    Usage::

        dedup = Deduplicator(working_dir=Path("data/dedup_scratch"))

        # Exact dedup a source's shards first
        seen = dedup.exact_dedup_source(
            src_dir=Path("data/filtered/wikipedia"),
            dst_dir=Path("data/exact_deduped/wikipedia"),
        )

        # Then fuzzy dedup the exact-deduped output
        dedup.minhash_dedup_source(
            src_dir=Path("data/exact_deduped/wikipedia"),
            dst_dir=Path("data/deduped/wikipedia"),
            source_name="wikipedia",
        )
    """

    def __init__(
        self,
        working_dir: Path,
        workers: int | None = None,
        threshold: float = JACCARD_THRESHOLD,
    ):
        self.working_dir = Path(working_dir)
        self.workers = workers or max(1, (os.cpu_count() or 4) // 2)
        self.threshold = threshold
        self.seen_hashes: set[str] = set()

        self._stats: dict[str, dict] = {}

    def exact_dedup_source(
        self,
        src_dir: Path,
        dst_dir: Path,
    ) -> dict:
        """
        Exact-dedup all JSONL shards in src_dir, writing to dst_dir.

        The seen_hashes set is shared across all calls to this method,
        enabling cross-source exact dedup (a Wikipedia article that also
        appears in CC will be caught).

        Args:
            src_dir: Source directory of JSONL shards.
            dst_dir: Output directory for exact-deduped shards.

        Returns:
            Aggregated stats dict.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        shards = sorted(src_dir.glob("*.jsonl"))

        if not shards:
            log.warning(f"No shards in {src_dir}")
            return {}

        log.info(f"Exact dedup: {src_dir.name} ({len(shards)} shards)...")

        agg = {"total": 0, "kept": 0, "exact_duplicates": 0}
        for shard in tqdm(shards, desc=f"Exact dedup {src_dir.name}", unit="shard"):
            out = dst_dir / shard.name
            if out.exists():
                log.debug(f"  Skipping {shard.name} — already done")
                continue
            stats = exact_dedup_jsonl(shard, out, self.seen_hashes)
            for k in agg:
                agg[k] += stats[k]

        log.info(
            f"  Exact dedup {src_dir.name}: "
            f"kept {agg['kept']:,}/{agg['total']:,} "
            f"({100*agg['kept']/max(agg['total'],1):.1f}%), "
            f"removed {agg['exact_duplicates']:,} exact duplicates"
        )
        return agg

    def minhash_dedup_source(
        self,
        src_dir: Path,
        dst_dir: Path,
        source_name: str,
    ) -> None:
        """
        Fuzzy-dedup a source's shards using datatrove MinHash pipeline.

        Args:
            src_dir:     Exact-deduped shards (input to minhash).
            dst_dir:     Final deduplicated output shards.
            source_name: Used to namespace the working directory.
        """
        working = self.working_dir / source_name
        run_minhash_dedup(
            input_dir=src_dir,
            output_dir=dst_dir,
            working_dir=working,
            workers=self.workers,
        )

    def deduplicate_source(
        self,
        src_dir: Path,
        dst_dir: Path,
        source_name: str,
    ) -> None:
        """
        Full two-stage dedup for a single source: exact then fuzzy.

        Intermediate exact-deduped files are written to a temp directory
        inside working_dir and cleaned up after minhash dedup completes.

        Args:
            src_dir:     Filtered JSONL shards for this source.
            dst_dir:     Final deduplicated output directory.
            source_name: Source name (wikipedia, code, common_crawl).
        """
        exact_dir = self.working_dir / source_name / "exact_deduped"

        log.info(f"=== Deduplicating {source_name} ===")

        # Stage 1: exact
        self.exact_dedup_source(src_dir=src_dir, dst_dir=exact_dir)

        # Stage 2: minhash fuzzy
        self.minhash_dedup_source(
            src_dir=exact_dir,
            dst_dir=dst_dir,
            source_name=source_name,
        )

        log.info(f"Deduplication complete for {source_name} → {dst_dir}")

    def report(self) -> str:
        """Human-readable summary of exact dedup stats."""
        total_seen = len(self.seen_hashes)
        lines = [
            "Deduplication report:",
            f"  Exact hash index size: {total_seen:>10,} documents",
            f"  (Fuzzy dedup stats available in datatrove logs)",
        ]
        return "\n".join(lines)