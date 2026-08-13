"""Exact benchmark-query contamination audit for finalized corpus splits."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from config.benchmarks import BENCHMARKS, benchmark_decontamination_contract
from curator.filters.dedup import normalize


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def extract_benchmark_query(extractor: str, row: dict) -> str:
    """Render the canonical lm-eval v0.4.9 benchmark input/query."""
    if extractor == "hellaswag_query_v1":
        context = row["ctx_a"] + " " + str(row["ctx_b"]).capitalize()
        return _hellaswag_preprocess(row["activity_label"] + ": " + context)
    if extractor == "arc_query_v1":
        return f"Question: {row['question']}\nAnswer:"
    if extractor == "mmlu_query_v1":
        choices = row["choices"]
        if len(choices) != 4:
            raise RuntimeError(f"MMLU row has {len(choices)} choices, expected 4")
        return (
            f"{str(row['question']).strip()}\n"
            f"A. {choices[0]}\nB. {choices[1]}\n"
            f"C. {choices[2]}\nD. {choices[3]}\nAnswer:"
        )
    if extractor == "truthfulqa_query_v1":
        return row["question"]
    if extractor == "humaneval_query_v1":
        return row["prompt"]
    raise ValueError(f"Unknown benchmark query extractor: {extractor!r}")


@dataclass
class BenchmarkIndex:
    matcher: Any
    query_count: int
    unique_pattern_count: int
    task_query_counts: dict[str, int]
    query_sha256: str
    contract: dict


def build_benchmark_index() -> BenchmarkIndex:
    """Load pinned benchmark splits and build an in-memory exact matcher."""
    try:
        import ahocorasick
    except ImportError as exc:
        raise RuntimeError(
            "pyahocorasick is required for benchmark decontamination"
        ) from exc

    from curator.sources.hf import load_dataset

    patterns: dict[str, list[dict]] = {}
    task_counts: Counter[str] = Counter()
    digest = hashlib.sha256()

    for benchmark, spec in BENCHMARKS.items():
        dataset_args = [spec["dataset_path"]]
        if spec["dataset_name"] is not None:
            dataset_args.append(spec["dataset_name"])
        rows = load_dataset(
            *dataset_args,
            revision=spec["dataset_revision"],
            split=spec["split"],
            streaming=True,
        )
        for row_number, row in enumerate(rows):
            query = extract_benchmark_query(spec["query_extractor"], row)
            normalized = normalize(query)
            if not normalized:
                raise RuntimeError(
                    f"{benchmark}:{row_number} produced an empty normalized query"
                )
            query_id = f"{benchmark}:{row_number}"
            query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            reference = {
                "benchmark": benchmark,
                "query_id": query_id,
                "query_sha256": query_hash,
            }
            pattern = f" {normalized} "
            patterns.setdefault(pattern, []).append(reference)
            task_counts[benchmark] += 1
            digest.update(query_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(normalized.encode("utf-8"))
            digest.update(b"\0")

    matcher = ahocorasick.Automaton()
    for pattern, references in sorted(patterns.items()):
        matcher.add_word(pattern, tuple(references))
    matcher.make_automaton()

    return BenchmarkIndex(
        matcher=matcher,
        query_count=sum(task_counts.values()),
        unique_pattern_count=len(patterns),
        task_query_counts=dict(sorted(task_counts.items())),
        query_sha256=digest.hexdigest(),
        contract=benchmark_decontamination_contract(),
    )


def audit_benchmark_contamination(
    split_paths: dict[str, Path],
    index: BenchmarkIndex,
    *,
    sample_limit: int = 20,
) -> dict:
    """Scan every finalized document for exact normalized benchmark queries."""
    auditor = BenchmarkContaminationAuditor(index, sample_limit=sample_limit)
    split_documents: dict[str, int] = {}
    for split, path in sorted(split_paths.items()):
        documents = 0
        with open(path, "rb", buffering=8 * 1024 * 1024) as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = orjson.loads(line)
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid JSONL record in {path}:{line_number}"
                    ) from exc
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(
                        f"Missing non-empty text in {path}:{line_number}"
                    )
                documents += 1
                auditor.observe(split, line_number, record)
        split_documents[split] = documents
    return auditor.report(split_documents)


class BenchmarkContaminationAuditor:
    """Incremental benchmark matcher that can share another corpus scan."""

    def __init__(self, index: BenchmarkIndex, *, sample_limit: int = 20):
        if sample_limit < 0:
            raise ValueError("sample_limit must be non-negative")
        self.index = index
        self.sample_limit = sample_limit
        self.matched_documents = 0
        self.matched_query_ids: set[str] = set()
        self.matched_documents_by_task: Counter[str] = Counter()
        self.samples: list[dict] = []

    def observe(self, split: str, line_number: int, record: dict) -> None:
        """Inspect one already-validated corpus record."""
        text = record["text"]
        document_matches: dict[str, dict] = {}
        normalized_document = f" {normalize(text)} "
        for _, references in self.index.matcher.iter(normalized_document):
            for reference in references:
                document_matches[reference["query_id"]] = reference
        if not document_matches:
            return

        self.matched_documents += 1
        tasks_in_document = {
            reference["benchmark"] for reference in document_matches.values()
        }
        for task in tasks_in_document:
            self.matched_documents_by_task[task] += 1
        self.matched_query_ids.update(document_matches)
        if len(self.samples) < self.sample_limit:
            self.samples.append({
                "split": split,
                "line": line_number,
                "source": record.get("source"),
                "matches": sorted(
                    document_matches.values(), key=lambda x: x["query_id"]
                ),
            })

    def report(self, split_documents: dict[str, int]) -> dict:
        """Return the durable report after all records have been observed."""
        return {
            "schema_version": 1,
            "algorithm": "normalized_exact_query_substring_aho_corasick",
            "scope": "full_corpus",
            "contract": self.index.contract,
            "query_count": self.index.query_count,
            "unique_pattern_count": self.index.unique_pattern_count,
            "task_query_counts": self.index.task_query_counts,
            "query_sha256": self.index.query_sha256,
            "split_documents": dict(sorted(split_documents.items())),
            "matched_documents": self.matched_documents,
            "matched_unique_queries": len(self.matched_query_ids),
            "matched_documents_by_benchmark": dict(
                sorted(self.matched_documents_by_task.items())
            ),
            "passed": self.matched_documents == 0,
            "samples": self.samples,
        }
