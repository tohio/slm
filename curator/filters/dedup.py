"""
curator/filters/dedup.py
-------------------------
Disk-based MinHash deduplication using datatrove.

Two-stage pipeline per source:
    1. Exact dedup (SHA-256 8-byte prefix, streaming) — cross-source index.
    2. Fuzzy dedup (datatrove's 4-stage MinHash LSH) — bounded RAM.

Peak RAM at any stage: O(shard_size), not O(corpus_size). Scales to 125m,
350m, and 1b with the same memory footprint.

Hash compaction:
    seen_hashes stores 8-byte binary prefixes of SHA-256 rather than
    64-character hex strings. At 80M docs that's ~640MB vs ~5GB.
    Collision probability at 80M docs: ~1 in 2.3 × 10^10.

References:
    datatrove minhash: https://github.com/huggingface/datatrove
    FineWeb pipeline:  https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
"""

import hashlib
import logging
import os
import re
import shutil
from pathlib import Path

import orjson
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

MINHASH_CONFIG = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
    n_grams=5,
)

JACCARD_THRESHOLD = 0.8


def _default_workers() -> int:
    cpu = os.cpu_count() or 4
    return max(1, cpu - 2)


def _dir_size_gb(path: Path) -> float:
    """Sum file sizes under a directory tree, in GB. Best-effort — broken
    symlinks and unreadable files are silently skipped."""
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total / (1024 ** 3)


# Pre-compiled for normalize() — previously compiled on every call, which at
# 80M docs was a meaningful fraction of exact-dedup wall time.
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize text for exact deduplication."""
    text = _PUNCT_RE.sub("", text.lower())
    return _WS_RE.sub(" ", text).strip()


def exact_hash(text: str) -> bytes:
    """
    First 8 bytes of SHA-256 of normalized text.

    Returns binary bytes (not hex) — 8× smaller in the seen_hashes set.
    Collision probability at 80M docs: ~1 in 2.3 × 10^10.
    """
    return hashlib.sha256(normalize(text).encode("utf-8")).digest()[:8]


# ── Exact dedup pre-pass ───────────────────────────────────────────────────────

def exact_dedup_jsonl(
    input_path: Path,
    output_path: Path,
    seen_hashes: set[bytes],
) -> dict:
    """
    Single-pass exact dedup. Updates seen_hashes in place so cross-shard
    and cross-source duplicates are caught.

    Returns:
        Stats dict with total, kept, exact_duplicates counts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = kept = exact_dupes = 0

    with open(input_path, "rb", buffering=8 * 1024 * 1024) as fin, \
         open(output_path, "wb", buffering=8 * 1024 * 1024) as fout:
        for line in fin:
            total += 1
            try:
                record = orjson.loads(line)
            except Exception:
                continue
            h = exact_hash(record.get("text", ""))
            if h in seen_hashes:
                exact_dupes += 1
                continue
            seen_hashes.add(h)
            fout.write(orjson.dumps(record))
            fout.write(b"\n")
            kept += 1

    return {"total": total, "kept": kept, "exact_duplicates": exact_dupes}


def _scan_hashes_into(input_path: Path, seen_hashes: set[bytes]) -> int:
    """
    Read a completed output shard and populate seen_hashes from it.

    Used on resume so cross-shard duplicate detection remains correct
    when restarting a partially-completed dedup run.
    """
    added = 0
    with open(input_path, "rb", buffering=8 * 1024 * 1024) as fin:
        for line in fin:
            try:
                record = orjson.loads(line)
            except Exception:
                continue
            seen_hashes.add(exact_hash(record.get("text", "")))
            added += 1
    return added


# ── Datatrove minhash pipeline ─────────────────────────────────────────────────

def run_minhash_dedup(
    input_dir: Path,
    output_dir: Path,
    working_dir: Path,
    workers: int | None = None,
    tasks: int | None = None,
) -> None:
    """
    Run datatrove's 4-stage disk-based MinHash deduplication.

    Args:
        input_dir:   Directory of JSONL shards to deduplicate.
        output_dir:  Directory to write deduplicated JSONL shards.
        working_dir: Scratch directory for datatrove intermediate state.
        workers:     Parallel workers. Defaults to cpu_count - 2.
        tasks:       Task count. Defaults to number of input shards.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    working_dir = Path(working_dir)

    n_workers = workers or _default_workers()

    sig_dir     = working_dir / "signatures"
    bucket_dir  = working_dir / "buckets"
    cluster_dir = working_dir / "clusters"
    logs_dir    = working_dir / "logs"

    shards = list(input_dir.glob("*.jsonl"))
    if not shards:
        log.warning(f"No JSONL shards in {input_dir} — skipping minhash dedup")
        return

    n_tasks = tasks or max(len(shards), 1)

    log.info(
        f"MinHash dedup: {len(shards)} shards, {n_tasks} tasks, {n_workers} workers\n"
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
        workers=n_workers,
        logging_dir=str(logs_dir / "signatures"),
    ).run()

    # Stage 2 — LSH bucketing (parallelism capped at num_buckets)
    log.info("Stage 2/4: LSH bucketing...")
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(sig_dir),
                output_folder=str(bucket_dir),
                config=MINHASH_CONFIG,
            ),
        ],
        tasks=MINHASH_CONFIG.num_buckets,
        workers=n_workers,
        logging_dir=str(logs_dir / "buckets"),
    ).run()

    # Stage 3 — Cluster (single-threaded by datatrove design)
    log.info("Stage 3/4: Clustering duplicates...")
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(bucket_dir),
                output_folder=str(cluster_dir),
                config=MINHASH_CONFIG,
            ),
        ],
        tasks=1,
        logging_dir=str(logs_dir / "clusters"),
    ).run()

    # Stage 4 — Filter
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
                    compression=None,
                ),
            ),
            JsonlWriter(
                str(output_dir),
                output_filename="${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=n_tasks,
        workers=n_workers,
        logging_dir=str(logs_dir / "filter"),
    ).run()

    log.info(f"MinHash dedup complete → {output_dir}")


# ── Top-level dedup entry point ────────────────────────────────────────────────

class Deduplicator:
    """
    Two-stage deduplicator: exact hash + datatrove MinHash LSH.

    Args:
        working_dir: Scratch directory for datatrove state.
        workers:     CPU workers. Default: cpu_count - 2.
        threshold:   Jaccard similarity threshold. Default: 0.8.
    """

    def __init__(
        self,
        working_dir: Path,
        workers: int | None = None,
        threshold: float = JACCARD_THRESHOLD,
    ):
        self.working_dir = Path(working_dir)
        self.workers = workers or _default_workers()
        self.threshold = threshold
        self.seen_hashes: set[bytes] = set()
        self._stats: dict[str, dict] = {}

    def exact_dedup_source(self, src_dir: Path, dst_dir: Path) -> dict:
        """
        Exact-dedup all JSONL shards in src_dir → dst_dir.

        Shards are processed in sorted order, so the earliest-named shard
        wins on collision. On resume, already-processed output shards are
        scanned into seen_hashes (in the same sort order) before processing
        new ones, so cross-shard dedup is consistent.
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
                added = _scan_hashes_into(out, self.seen_hashes)
                log.debug(f"  Resume: scanned {added:,} hashes from {out.name}")
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
        self, src_dir: Path, dst_dir: Path, source_name: str
    ) -> None:
        """Fuzzy-dedup a source's shards using datatrove MinHash pipeline."""
        working = self.working_dir / source_name
        run_minhash_dedup(
            input_dir=src_dir,
            output_dir=dst_dir,
            working_dir=working,
            workers=self.workers,
        )

    def deduplicate_source(
        self, src_dir: Path, dst_dir: Path, source_name: str
    ) -> None:
        """
        Full two-stage dedup for a single source: exact then fuzzy.

        On success, removes the per-source scratch directory
        (working_dir/<source_name>/) which contains exact-dedup intermediate
        output plus all MinHash stage scratch (signatures, buckets, clusters,
        removed, logs). The deduplicated output in dst_dir is preserved —
        dst_dir lives outside working_dir.

        Without this cleanup the 125m run accumulated 135 GB of scratch
        across all sources; at 1b that scales to ~780 GB and would not fit
        on a 2 TB disk alongside raw + filtered + curated.

        Cleanup is deliberately NOT in a finally block — if MinHash crashes
        mid-pipeline we want the scratch preserved for debugging.
        """
        scratch_dir = self.working_dir / source_name
        exact_dir = scratch_dir / "exact_deduped"
        log.info(f"=== Deduplicating {source_name} ===")
        self.exact_dedup_source(src_dir=src_dir, dst_dir=exact_dir)
        self.minhash_dedup_source(
            src_dir=exact_dir, dst_dir=dst_dir, source_name=source_name
        )

        # Verify dst_dir actually has output before removing scratch — cheap
        # insurance against a silent MinHash failure that produces no shards.
        if dst_dir.exists() and any(dst_dir.glob("*.jsonl")):
            scratch_size_gb = _dir_size_gb(scratch_dir)
            log.info(
                f"  {source_name}: removing scratch "
                f"({scratch_dir}, {scratch_size_gb:.2f} GB)..."
            )
            shutil.rmtree(scratch_dir, ignore_errors=True)
        else:
            log.warning(
                f"  {source_name}: dst_dir {dst_dir} has no JSONL output — "
                f"keeping scratch at {scratch_dir} for inspection"
            )
        log.info(f"Deduplication complete for {source_name} → {dst_dir}")

    def report(self) -> str:
        hash_mem_mb = len(self.seen_hashes) * 8 / 1024 / 1024
        return (
            f"Deduplication report:\n"
            f"  Exact hash index size: {len(self.seen_hashes):>10,} documents\n"
            f"  Hash index memory:     {hash_mem_mb:>10.1f} MB\n"
            f"  (Fuzzy dedup stats available in datatrove logs)"
        )