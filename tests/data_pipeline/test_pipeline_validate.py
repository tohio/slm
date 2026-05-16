"""
tests/data_pipeline/test_pipeline_validate.py
----------------------------------------------
Validates real outputs from 'make validate'.

Run after: make validate SIZE=mini
Command:   make test-validate

Checks:
    - data/validated/train.jsonl exists and is non-empty
    - Validated output is a sampled subset of curated input
    - Validated prose-like docs pass quality checks
    - Source-aware validation stats are internally consistent
    - validation_stats.json exists and matches output files

This file is intentionally fast. Tests stream/sample large JSONL files rather
than loading whole corpora when a quick invariant is enough.
"""

import json

import pytest

from tests.data_pipeline.helpers import requires_stage, pipeline_path
from curator.filters.quality import QualityFilter
from curator.filters.dedup import exact_hash


pytestmark = requires_stage("validate")


PROSE_HEURISTIC_SKIP_SOURCES = {
    "codesearchnet",
    "stack_smol",
    "stack_v1",
    "jupyter",
    "conala",
    "synthetic_arithmetic",
    "synthetic_task_code",
    "educational_qa_mcq",
    "factual_restraint",
    "nemotron_cc_math",
    "nemotron_specialized",
}

VALIDATE_SAMPLE_DOCS = 500


def _count_jsonl(path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


class TestValidatedOutput:
    def test_validated_train_jsonl_exists(self):
        assert pipeline_path("validated", "train.jsonl").exists()

    def test_validated_val_jsonl_exists(self):
        assert pipeline_path("validated", "val.jsonl").exists()

    def test_validated_train_jsonl_non_empty(self):
        path = pipeline_path("validated", "train.jsonl")
        with open(path, encoding="utf-8") as f:
            assert any(line.strip() for line in f), f"{path} is empty"

    def test_validated_is_subset_of_curated_sample(self):
        """
        Validation only filters — sampled validated docs must exist in curated.

        This is bounded so make test-validate stays quick on larger runs.
        """
        validated_hashes: list[bytes] = []
        with open(pipeline_path("validated", "train.jsonl"), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                validated_hashes.append(exact_hash(json.loads(line).get("text", "")))
                if len(validated_hashes) >= VALIDATE_SAMPLE_DOCS:
                    break

        assert validated_hashes, "validated/train.jsonl has no sampled docs"

        wanted = set(validated_hashes)
        found: set[bytes] = set()
        with open(pipeline_path("curated", "train.jsonl"), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                h = exact_hash(json.loads(line).get("text", ""))
                if h in wanted:
                    found.add(h)
                if found == wanted:
                    break

        missing = wanted - found
        assert not missing, (
            f"{len(missing)} sampled validated docs not found in curated output — "
            f"validation may be adding documents, not just filtering"
        )

    def test_validated_docs_pass_quality_checks(self):
        """Validated prose-like docs should still pass quality filters."""
        qf = QualityFilter()
        failures = []
        checked = 0

        with open(pipeline_path("validated", "train.jsonl"), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                doc = json.loads(line)
                if doc.get("source") in PROSE_HEURISTIC_SKIP_SOURCES:
                    continue

                passed, reason = qf.check(doc)
                if not passed:
                    failures.append(f"rejected '{reason}': {doc['text'][:80]}")

                checked += 1
                if checked >= 100:
                    break

        assert len(failures) == 0, (
            f"{len(failures)} validated prose-like docs fail quality checks:\n"
            + "\n".join(failures[:5])
        )

    def test_validated_has_required_fields(self):
        checked = 0
        with open(pipeline_path("validated", "train.jsonl"), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                doc = json.loads(line)
                assert "text" in doc
                assert "source" in doc
                assert len(doc["text"]) > 0

                checked += 1
                if checked >= 20:
                    break

        assert checked > 0, "validated/train.jsonl has no docs to check"

    def test_validated_retention_rate_reasonable(self):
        """
        Validation should remove some docs but not too many.
        Expect 70-100% retention — if below 70% something is wrong.
        """
        curated_count = _count_jsonl(pipeline_path("curated", "train.jsonl"))
        validated_count = _count_jsonl(pipeline_path("validated", "train.jsonl"))
        retention = validated_count / max(curated_count, 1)
        assert retention >= 0.70, (
            f"Validation retention rate too low: {retention:.1%} "
            f"({validated_count}/{curated_count} docs retained). "
            f"Check KenLM threshold or filter settings."
        )


class TestValidationStats:
    def _load_stats(self) -> dict:
        path = pipeline_path("validated", "validation_stats.json")
        assert path.exists(), "validation_stats.json not found"
        with open(path) as f:
            return json.load(f)

    def test_validation_stats_exists(self):
        assert pipeline_path("validated", "validation_stats.json").exists()

    def test_validation_stats_fields(self):
        stats = self._load_stats()
        assert "total" in stats
        assert "kept" in stats
        assert "rejected_terminal_punct" in stats
        assert "rejected_repeated_lines" in stats
        assert "rejected_perplexity" in stats
        assert "skipped_prose_heuristics" in stats
        assert "splits" in stats

    def test_validation_stats_kept_le_total(self):
        stats = self._load_stats()
        assert stats["kept"] <= stats["total"]

    def test_validation_stats_kept_positive(self):
        stats = self._load_stats()
        assert stats["kept"] > 0, "Validation kept 0 documents — something is wrong"

    def test_validation_stats_record_source_aware_skips(self):
        stats = self._load_stats()
        assert stats["skipped_prose_heuristics"] > 0, (
            "Expected source-aware validation to skip prose heuristics for "
            "code/synthetic/symbol-heavy sources"
        )

    def test_validation_stats_matches_output_files(self):
        """
        Per-split kept counts should match per-split output line counts.

        The top-level `kept` field aggregates train + val, so comparing it
        directly to train.jsonl is wrong by exactly val.kept.
        """
        stats = self._load_stats()
        splits = stats.get("splits", {})
        assert splits, (
            "validation_stats.json missing 'splits' field — "
            "expected per-split train/val breakdown"
        )

        for split_name, split_stats in splits.items():
            path = pipeline_path("validated", f"{split_name}.jsonl")
            if not path.exists():
                continue
            actual = _count_jsonl(path)
            assert split_stats["kept"] == actual, (
                f"validation_stats.json splits.{split_name}.kept = "
                f"{split_stats['kept']} but {split_name}.jsonl has {actual} lines"
            )
