"""Disk-backed MinHash audit for train/validation near-duplicate overlap."""

from __future__ import annotations

import logging
import os
import shutil
import struct
from collections import Counter
from pathlib import Path

import orjson
from datatrove.data import Document, DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup import MinhashBuildIndex
from datatrove.pipeline.dedup.minhash import (
    MinhashDedupBuckets,
    MinhashDedupSignature,
)

from curator.filters.dedup import (
    MINHASH_CONFIG,
    MINHASH_CONTRACT,
    MINHASH_LSH_CROSSOVER,
)

log = logging.getLogger(__name__)

_SENTINEL = (1 << 32) - 1
_PAIR = struct.Struct("<4I")
_DOC_ID = struct.Struct("<I")


class JsonlByteRangeReader(PipelineStep):
    """Read one JSONL file in deterministic newline-aligned byte ranges."""

    type = "Reader"
    name = "JSONL byte-range reader"

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)

    def run(
        self,
        data: DocumentsPipeline = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> DocumentsPipeline:
        if data is not None:
            raise ValueError("JsonlByteRangeReader must be the first pipeline step")
        file_size = self.path.stat().st_size
        start = file_size * rank // world_size
        end = file_size * (rank + 1) // world_size
        local_document = 0

        with open(self.path, "rb", buffering=8 * 1024 * 1024) as handle:
            if start:
                handle.seek(start - 1)
                starts_after_newline = handle.read(1) == b"\n"
                handle.seek(start)
                if not starts_after_newline:
                    handle.readline()
            else:
                handle.seek(0)
            while rank + 1 == world_size or handle.tell() < end:
                line = handle.readline()
                if not line:
                    break
                try:
                    record = orjson.loads(line)
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid JSONL record in {self.path}, byte range {rank}"
                    ) from exc
                text = record.get("text")
                source = record.get("source")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(
                        f"Missing non-empty text in {self.path}, byte range {rank}"
                    )
                if not isinstance(source, str) or not source:
                    raise RuntimeError(
                        f"Missing source in {self.path}, byte range {rank}"
                    )
                record_id = record.get("id")
                yield Document(
                    text=text,
                    id=(
                        str(record_id)
                        if record_id is not None
                        else f"{rank}:{local_document}"
                    ),
                    metadata={"source": source},
                )
                local_document += 1


class NearOverlapReporter(PipelineStep):
    """Write bounded, text-free per-range reports for removal identifiers."""

    type = "Audit"
    name = "MinHash near-overlap reporter"

    def __init__(
        self,
        removal_dir: Path,
        report_dir: Path,
        *,
        sample_limit: int = 5,
    ):
        super().__init__()
        self.removal_dir = Path(removal_dir)
        self.report_dir = Path(report_dir)
        self.sample_limit = sample_limit

    def run(
        self,
        data: DocumentsPipeline,
        rank: int = 0,
        world_size: int = 1,
    ) -> DocumentsPipeline:
        removal_path = self.removal_dir / f"{rank:06d}.remove"
        removal_handle = open(removal_path, "rb") if removal_path.exists() else None

        def next_removal() -> int | None:
            if removal_handle is None:
                return None
            value = removal_handle.read(_DOC_ID.size)
            if not value:
                return None
            if len(value) != _DOC_ID.size:
                raise RuntimeError(f"Corrupt removal file: {removal_path}")
            return _DOC_ID.unpack(value)[0]

        documents = matched_documents = 0
        source_counts: Counter[str] = Counter()
        samples: list[dict] = []
        next_id = next_removal()
        try:
            for document_id, document in enumerate(data):
                documents += 1
                if next_id is not None and next_id < document_id:
                    raise RuntimeError(f"Unsorted removal file: {removal_path}")
                if next_id == document_id:
                    matched_documents += 1
                    source = document.metadata["source"]
                    source_counts[source] += 1
                    if len(samples) < self.sample_limit:
                        samples.append({
                            "range_rank": rank,
                            "range_document": document_id,
                            "record_id": document.id,
                            "source": source,
                        })
                    next_id = next_removal()
                yield document
            if next_id is not None:
                raise RuntimeError(
                    f"Removal id {next_id} exceeds document range {rank}"
                )
        finally:
            if removal_handle is not None:
                removal_handle.close()

        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.report_dir / f"{rank:06d}.json"
        tmp_path = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
        payload = {
            "rank": rank,
            "documents": documents,
            "matched_documents": matched_documents,
            "matched_documents_by_source": dict(sorted(source_counts.items())),
            "samples": samples,
        }
        with open(tmp_path, "wb") as handle:
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(report_path)


def _run_signatures(
    path: Path,
    output_dir: Path,
    logs_dir: Path,
    *,
    tasks: int,
    workers: int,
) -> None:
    LocalPipelineExecutor(
        pipeline=[
            JsonlByteRangeReader(path),
            MinhashDedupSignature(
                output_folder=str(output_dir),
                config=MINHASH_CONFIG,
            ),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(logs_dir),
        skip_completed=False,
    ).run()


def build_cross_index_removals(
    duplicate_dir: Path,
    removal_dir: Path,
) -> dict:
    """Keep only candidate components connected to the validation index."""
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    sizes: dict[tuple[int, int], int] = {}
    sentinel = (_SENTINEL, _SENTINEL)
    candidate_pairs = index_pairs = 0

    def parent(node: tuple[int, int]) -> tuple[int, int]:
        root = parents.get(node)
        if root is None:
            parents[node] = node
            return node
        if root != node:
            parents[node] = parent(root)
        return parents[node]

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root = parent(left)
        right_root = parent(right)
        if left_root == right_root:
            return
        left_size = sizes.get(left_root, 1)
        right_size = sizes.get(right_root, 1)
        if right_root == sentinel or (
            left_root != sentinel and left_size < right_size
        ):
            left_root, right_root = right_root, left_root
            left_size, right_size = right_size, left_size
        parents[right_root] = left_root
        sizes[left_root] = left_size + right_size
        sizes.pop(right_root, None)

    for duplicate_path in sorted(Path(duplicate_dir).glob("*.dups")):
        with open(duplicate_path, "rb") as handle:
            while chunk := handle.read(_PAIR.size * 65536):
                if len(chunk) % _PAIR.size:
                    raise RuntimeError(f"Corrupt duplicate file: {duplicate_path}")
                for file_1, doc_1, file_2, doc_2 in struct.iter_unpack(
                    _PAIR.format, chunk
                ):
                    candidate_pairs += 1
                    if (file_1, doc_1) == sentinel:
                        index_pairs += 1
                    union((file_1, doc_1), (file_2, doc_2))

    matched_by_file: dict[int, list[int]] = {}
    for node in sorted(parents):
        if node == sentinel or parent(node) != sentinel:
            continue
        file_id, document_id = node
        matched_by_file.setdefault(file_id, []).append(document_id)

    removal_dir = Path(removal_dir)
    removal_dir.mkdir(parents=True, exist_ok=True)
    for file_id, document_ids in sorted(matched_by_file.items()):
        with open(removal_dir / f"{file_id:06d}.remove", "wb") as handle:
            for document_id in document_ids:
                handle.write(_DOC_ID.pack(document_id))

    return {
        "candidate_pairs": candidate_pairs,
        "validation_index_pairs": index_pairs,
        "candidate_nodes": max(0, len(parents) - int(sentinel in parents)),
        "matched_documents": sum(map(len, matched_by_file.values())),
    }


def _aggregate_reports(
    report_dir: Path,
    *,
    tasks: int,
    cluster_stats: dict,
) -> dict:
    report_paths = sorted(Path(report_dir).glob("*.json"))
    if len(report_paths) != tasks:
        raise RuntimeError(
            f"Near-overlap audit produced {len(report_paths)} reports, expected {tasks}"
        )

    documents = matched_documents = 0
    source_counts: Counter[str] = Counter()
    samples: list[dict] = []
    for report_path in report_paths:
        with open(report_path, "rb") as handle:
            report = orjson.loads(handle.read())
        documents += int(report["documents"])
        matched_documents += int(report["matched_documents"])
        source_counts.update(report["matched_documents_by_source"])
        samples.extend(report["samples"])

    if matched_documents != cluster_stats["matched_documents"]:
        raise RuntimeError(
            "Near-overlap cluster/report count mismatch: "
            f"{cluster_stats['matched_documents']} != {matched_documents}"
        )
    return {
        "documents": documents,
        "matched_documents": matched_documents,
        "matched_documents_by_source": dict(sorted(source_counts.items())),
        "samples": samples[:20],
    }


def audit_minhash_split_overlap(
    train_path: Path,
    validation_path: Path,
    working_dir: Path,
    *,
    workers: int | None = None,
) -> dict:
    """Audit all train documents against a disk-backed validation MinHash index."""
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    working_dir = Path(working_dir)
    n_workers = max(1, workers or (os.cpu_count() or 4) - 2)
    range_tasks = min(256, max(16, n_workers * 4))
    range_workers = min(n_workers, range_tasks)

    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True)

    validation_signatures = working_dir / "validation_signatures"
    validation_index = working_dir / "validation_index"
    train_signatures = working_dir / "train_signatures"
    duplicate_dir = working_dir / "duplicate_pairs"
    removal_dir = working_dir / "removals"
    report_dir = working_dir / "reports"
    logs_dir = working_dir / "logs"

    _run_signatures(
        validation_path,
        validation_signatures,
        logs_dir / "validation_signatures",
        tasks=range_tasks,
        workers=range_workers,
    )
    LocalPipelineExecutor(
        pipeline=[
            MinhashBuildIndex(
                input_folder=str(validation_signatures),
                output_folder=str(validation_index),
                index_name="validation",
                config=MINHASH_CONFIG,
            )
        ],
        tasks=MINHASH_CONFIG.num_buckets,
        workers=min(n_workers, MINHASH_CONFIG.num_buckets),
        logging_dir=str(logs_dir / "validation_index"),
        skip_completed=False,
    ).run()

    _run_signatures(
        train_path,
        train_signatures,
        logs_dir / "train_signatures",
        tasks=range_tasks,
        workers=range_workers,
    )
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(train_signatures),
                output_folder=str(duplicate_dir),
                index_folder=str(validation_index),
                config=MINHASH_CONFIG,
                only_dedup_in_index=False,
            )
        ],
        tasks=MINHASH_CONFIG.num_buckets,
        workers=min(n_workers, MINHASH_CONFIG.num_buckets),
        logging_dir=str(logs_dir / "cross_split_buckets"),
        skip_completed=False,
    ).run()

    cluster_stats = build_cross_index_removals(duplicate_dir, removal_dir)
    LocalPipelineExecutor(
        pipeline=[
            JsonlByteRangeReader(train_path),
            NearOverlapReporter(removal_dir, report_dir),
        ],
        tasks=range_tasks,
        workers=range_workers,
        logging_dir=str(logs_dir / "report"),
        skip_completed=False,
    ).run()
    aggregate = _aggregate_reports(
        report_dir,
        tasks=range_tasks,
        cluster_stats=cluster_stats,
    )

    return {
        "schema_version": 1,
        "algorithm": "datatrove_minhash_validation_index_lsh",
        "scope": "full_train_against_full_validation",
        "minhash": {
            **MINHASH_CONTRACT,
            "lsh_probability_50pct": MINHASH_LSH_CROSSOVER,
        },
        "range_tasks": range_tasks,
        "train_documents": aggregate["documents"],
        "matched_train_documents": aggregate["matched_documents"],
        "matched_train_documents_by_source": aggregate[
            "matched_documents_by_source"
        ],
        "candidate_pairs": cluster_stats["candidate_pairs"],
        "validation_index_pairs": cluster_stats["validation_index_pairs"],
        "passed": aggregate["matched_documents"] == 0,
        "samples": aggregate["samples"],
    }
