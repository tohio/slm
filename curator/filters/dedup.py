"""
curator/filters/dedup.py
-------------------------
Disk-based MinHash deduplication using datatrove.

Two-stage pipeline per source:
    1. Exact dedup (SHA-256 16-byte prefix, streaming) — cross-source index.
    2. Fuzzy dedup (datatrove's 4-stage MinHash LSH) — bounded RAM.

MinHash stages are disk-backed and bounded by shard/task size. Exact
cross-source dedup intentionally keeps one 16-byte digest per unique document
in a Python set, so that stage is O(unique documents) in RAM; Python object and
hash-table overhead is additional to the raw digest payload reported below.

Hash compaction:
    seen_hashes stores 16-byte binary prefixes of SHA-256 rather than
    64-character hex strings. At 80M docs the raw digest payload is ~1.28GB
    and the birthday-bound collision probability is approximately 9.4e-24.

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

MINHASH_CONTRACT = {
    "hash_fc": "sha1",
    "precision": 64,
    "num_buckets": 14,
    "hashes_per_bucket": 8,
    "n_grams": 5,
}
MINHASH_CONFIG = MinhashConfig(
    hash_config=HashConfig(
        precision=MINHASH_CONTRACT["precision"],
        hash_fc=MINHASH_CONTRACT["hash_fc"],
    ),
    num_buckets=MINHASH_CONTRACT["num_buckets"],
    hashes_per_bucket=MINHASH_CONTRACT["hashes_per_bucket"],
    n_grams=MINHASH_CONTRACT["n_grams"],
)

# With 14 bands × 8 hashes, the LSH candidate probability is
# 1 - (1 - s**8)**14. Its 50% crossover is ~0.685. Datatrove's MinHash
# pipeline is probabilistic; this is not a strict Jaccard cutoff.
MINHASH_LSH_CROSSOVER = (
    1 - (0.5 ** (1 / MINHASH_CONTRACT["num_buckets"]))
) ** (1 / MINHASH_CONTRACT["hashes_per_bucket"])


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


def _count_jsonl_records(path: Path) -> int:
    """Count records in flat JSONL shards without parsing document payloads."""
    count = 0
    for shard in sorted(Path(path).glob("*.jsonl")):
        with open(shard, "rb", buffering=8 * 1024 * 1024) as handle:
            last_byte = b""
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                count += chunk.count(b"\n")
                last_byte = chunk[-1:]
            if last_byte and last_byte != b"\n":
                count += 1
    return count


def build_dedup_stats(
    *,
    source: str,
    exact_stats: dict,
    final_documents: int,
    fuzzy_enabled: bool,
    fuzzy_partition_field: str | None = None,
    fuzzy_partitions: list[str] | None = None,
) -> dict:
    """Build and validate durable per-source deduplication counts."""
    input_documents = int(exact_stats.get("total", 0))
    exact_kept = int(exact_stats.get("kept", 0))
    exact_duplicates = int(exact_stats.get("exact_duplicates", 0))
    if input_documents != exact_kept + exact_duplicates:
        raise RuntimeError(
            f"{source}: inconsistent exact-dedup counts: "
            f"input={input_documents}, kept={exact_kept}, "
            f"duplicates={exact_duplicates}"
        )
    if final_documents < 0 or final_documents > exact_kept:
        raise RuntimeError(
            f"{source}: invalid fuzzy-dedup output count "
            f"{final_documents} for {exact_kept} exact-kept documents"
        )

    fuzzy_duplicates = exact_kept - final_documents
    if not fuzzy_enabled and fuzzy_duplicates:
        raise RuntimeError(
            f"{source}: fuzzy dedup is disabled but output removed "
            f"{fuzzy_duplicates} documents"
        )

    return {
        "source": source,
        "input_documents": input_documents,
        "exact_kept_documents": exact_kept,
        "exact_duplicate_documents": exact_duplicates,
        "fuzzy_enabled": fuzzy_enabled,
        "fuzzy_partition_field": fuzzy_partition_field,
        "fuzzy_partitions": sorted(fuzzy_partitions or []),
        "fuzzy_kept_documents": final_documents,
        "fuzzy_duplicate_documents": fuzzy_duplicates,
        "final_documents": final_documents,
    }


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
    First 16 bytes of SHA-256 of normalized text.

    A 128-bit prefix keeps the exact-dedup false-positive risk negligible
    while remaining much more compact than Python hex strings.
    """
    return hashlib.sha256(normalize(text).encode("utf-8")).digest()[:16]


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

    tmp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    parse_errors = 0
    try:
        with open(input_path, "rb", buffering=8 * 1024 * 1024) as fin, \
             open(tmp_path, "wb", buffering=8 * 1024 * 1024) as fout:
            for line in fin:
                total += 1
                try:
                    record = orjson.loads(line)
                except Exception:
                    parse_errors += 1
                    continue
                h = exact_hash(record.get("text", ""))
                if h in seen_hashes:
                    exact_dupes += 1
                    continue
                seen_hashes.add(h)
                fout.write(orjson.dumps(record))
                fout.write(b"\n")
                kept += 1
            fout.flush()
            os.fsync(fout.fileno())
        if parse_errors:
            raise RuntimeError(
                f"{input_path}: {parse_errors:,} invalid JSONL records"
            )
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return {"total": total, "kept": kept, "exact_duplicates": exact_dupes}


def _scan_hashes_into(input_path: Path, seen_hashes: set[bytes]) -> int:
    """
    Read a completed output shard and populate seen_hashes from it.

    Used on resume so cross-shard duplicate detection remains correct
    when restarting a partially-completed dedup run.
    """
    added = 0
    parse_errors = 0
    with open(input_path, "rb", buffering=8 * 1024 * 1024) as fin:
        for line in fin:
            try:
                record = orjson.loads(line)
            except Exception:
                parse_errors += 1
                continue
            seen_hashes.add(exact_hash(record.get("text", "")))
            added += 1
    if parse_errors:
        raise RuntimeError(
            f"{input_path}: {parse_errors:,} invalid JSONL records while "
            f"reconstructing the exact-dedup index"
        )
    return added


# ── Datatrove minhash pipeline ─────────────────────────────────────────────────

def run_minhash_dedup(
    input_dir: Path,
    output_dir: Path,
    working_dir: Path,
    workers: int | None = None,
    tasks: int | None = None,
    output_filename: str = "${rank}.jsonl",
) -> None:
    """
    Run datatrove's 4-stage disk-based MinHash deduplication.

    Args:
        input_dir:   Directory of JSONL shards to deduplicate.
        output_dir:  Directory to write deduplicated JSONL shards.
        working_dir: Scratch directory for datatrove intermediate state.
        workers:     Parallel workers. Defaults to cpu_count - 2.
        tasks:       Task count. Defaults to number of input shards.
        output_filename: Datatrove output template. Partitioned runs must use
                         distinct templates when sharing an output directory.
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
                output_filename=output_filename,
                compression=None,
            ),
        ],
        tasks=n_tasks,
        workers=n_workers,
        logging_dir=str(logs_dir / "filter"),
    ).run()

    log.info(f"MinHash dedup complete → {output_dir}")


_PARTITION_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def partition_jsonl_by_field(
    input_dir: Path,
    output_dir: Path,
    field: str,
) -> dict[str, Path]:
    """Partition flat JSONL shards by a required top-level string field.

    Single-partition shards are hard-linked when possible, so the normal
    Common Crawl layout does not duplicate the exact-deduped corpus in scratch.
    Mixed shards are split without changing their serialized records.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    partitions: dict[str, Path] = {}

    for shard in sorted(input_dir.glob("*.jsonl")):
        values: set[str] = set()
        with open(shard, "rb", buffering=8 * 1024 * 1024) as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = orjson.loads(line)
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid JSONL record in {shard}:{line_number}"
                    ) from exc
                value = record.get(field)
                if not isinstance(value, str) or not value:
                    raise RuntimeError(
                        f"Missing required string field {field!r} in "
                        f"{shard}:{line_number}"
                    )
                if not _PARTITION_NAME_RE.fullmatch(value):
                    raise RuntimeError(
                        f"Unsafe {field!r} partition value {value!r} in "
                        f"{shard}:{line_number}"
                    )
                values.add(value)

        if not values:
            continue

        for value in values:
            partition_dir = output_dir / value
            partition_dir.mkdir(parents=True, exist_ok=True)
            partitions[value] = partition_dir

        if len(values) == 1:
            value = next(iter(values))
            destination = partitions[value] / shard.name
            try:
                os.link(shard, destination)
            except OSError:
                shutil.copy2(shard, destination)
            continue

        writers = {
            value: open(
                partitions[value] / shard.name,
                "wb",
                buffering=8 * 1024 * 1024,
            )
            for value in values
        }
        try:
            with open(shard, "rb", buffering=8 * 1024 * 1024) as handle:
                for line in handle:
                    record = orjson.loads(line)
                    writers[record[field]].write(line)
        finally:
            for writer in writers.values():
                writer.close()

    return dict(sorted(partitions.items()))


# ── Top-level dedup entry point ────────────────────────────────────────────────

class Deduplicator:
    """
    Two-stage deduplicator: exact hash + datatrove MinHash LSH.

    Args:
        working_dir: Scratch directory for datatrove state.
        workers:     CPU workers. Default: cpu_count - 2.
        MinHash uses the probabilistic LSH contract described by
        MINHASH_CONFIG; it does not expose a strict Jaccard threshold.
    """

    def __init__(
        self,
        working_dir: Path,
        workers: int | None = None,
    ):
        self.working_dir = Path(working_dir)
        self.workers = workers or _default_workers()
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

    def index_source_input(self, input_dir: Path) -> int:
        """Rebuild the cross-source exact-hash index from a source input.

        The fresh-run index contains every unique exact hash seen before fuzzy
        filtering. Reconstructing from final fuzzy output would omit records
        removed by MinHash and make resumed cross-source behavior differ.
        """
        added = 0
        for shard in sorted(Path(input_dir).glob("*.jsonl")):
            added += _scan_hashes_into(shard, self.seen_hashes)
        return added

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
        exact_stats = self.exact_dedup_source(src_dir=src_dir, dst_dir=exact_dir)
        self.minhash_dedup_source(
            src_dir=exact_dir, dst_dir=dst_dir, source_name=source_name
        )
        self._stats[source_name] = build_dedup_stats(
            source=source_name,
            exact_stats=exact_stats,
            final_documents=_count_jsonl_records(dst_dir),
            fuzzy_enabled=True,
        )

        self._cleanup_completed_scratch(
            source_name=source_name,
            scratch_dir=scratch_dir,
            dst_dir=dst_dir,
        )
        log.info(f"Deduplication complete for {source_name} → {dst_dir}")

    def deduplicate_source_by_partition(
        self,
        src_dir: Path,
        dst_dir: Path,
        source_name: str,
        partition_field: str,
    ) -> None:
        """Exact-dedup globally, then fuzzy-dedup each field value alone."""
        scratch_dir = self.working_dir / source_name
        exact_dir = scratch_dir / "exact_deduped"
        partitioned_dir = scratch_dir / "partitioned"

        log.info(
            f"=== Deduplicating {source_name} by {partition_field} partition ==="
        )
        exact_stats = self.exact_dedup_source(src_dir=src_dir, dst_dir=exact_dir)
        partitions = partition_jsonl_by_field(
            input_dir=exact_dir,
            output_dir=partitioned_dir,
            field=partition_field,
        )
        if not partitions:
            raise RuntimeError(
                f"{source_name}: no non-empty {partition_field!r} partitions"
            )

        for partition, partition_dir in partitions.items():
            log.info(
                f"  {source_name}: fuzzy MinHash partition "
                f"{partition_field}={partition}"
            )
            run_minhash_dedup(
                input_dir=partition_dir,
                output_dir=dst_dir,
                working_dir=scratch_dir / "minhash" / partition,
                workers=self.workers,
                output_filename=f"{partition}_${{rank}}.jsonl",
            )
            if not any(dst_dir.glob(f"{partition}_*.jsonl")):
                raise RuntimeError(
                    f"{source_name}: MinHash partition {partition!r} "
                    "produced no output shards"
                )

        self._stats[source_name] = build_dedup_stats(
            source=source_name,
            exact_stats=exact_stats,
            final_documents=_count_jsonl_records(dst_dir),
            fuzzy_enabled=True,
            fuzzy_partition_field=partition_field,
            fuzzy_partitions=list(partitions),
        )

        self._cleanup_completed_scratch(
            source_name=source_name,
            scratch_dir=scratch_dir,
            dst_dir=dst_dir,
        )
        log.info(f"Deduplication complete for {source_name} → {dst_dir}")

    def deduplicate_source_exact_only(
        self,
        src_dir: Path,
        dst_dir: Path,
        source_name: str,
    ) -> None:
        """Run exact deduplication and record the same durable audit schema."""
        scratch_dir = self.working_dir / source_name
        exact_dir = scratch_dir / "exact_deduped"
        log.info(f"=== Exact-deduplicating {source_name} ===")
        exact_stats = self.exact_dedup_source(src_dir=src_dir, dst_dir=exact_dir)

        dst_dir.mkdir(parents=True, exist_ok=True)
        for shard in sorted(exact_dir.glob("*.jsonl")):
            shutil.copy2(shard, dst_dir / shard.name)

        self._stats[source_name] = build_dedup_stats(
            source=source_name,
            exact_stats=exact_stats,
            final_documents=_count_jsonl_records(dst_dir),
            fuzzy_enabled=False,
        )
        self._cleanup_completed_scratch(
            source_name=source_name,
            scratch_dir=scratch_dir,
            dst_dir=dst_dir,
        )
        log.info(f"Exact deduplication complete for {source_name} → {dst_dir}")

    def stats_for(self, source_name: str) -> dict:
        """Return completed audit counts for a source or fail closed."""
        if source_name not in self._stats:
            raise RuntimeError(f"No deduplication stats recorded for {source_name}")
        return dict(self._stats[source_name])

    @staticmethod
    def _cleanup_completed_scratch(
        source_name: str,
        scratch_dir: Path,
        dst_dir: Path,
    ) -> None:
        """Remove scratch only after a dedup run produced output shards."""
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

    def report(self) -> str:
        hash_mem_mb = len(self.seen_hashes) * 16 / 1024 / 1024
        return (
            f"Deduplication report:\n"
            f"  Exact hash index size: {len(self.seen_hashes):>10,} documents\n"
            f"  Raw digest payload:    {hash_mem_mb:>10.1f} MB\n"
            f"  MinHash LSH crossover: {MINHASH_LSH_CROSSOVER:>10.3f}\n"
            f"  (Python set/index overhead is additional)\n"
            f"  (Per-source exact/fuzzy counts are stored in stage manifests)"
        )
