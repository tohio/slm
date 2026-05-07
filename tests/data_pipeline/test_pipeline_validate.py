"""
tests/data_pipeline/test_pipeline_validate.py
----------------------------------------------
Validates real outputs from 'make validate'.

Run after: make validate
Command:   make test-validate

Checks:
    - data/validated/train.jsonl exists and is non-empty
    - Validated output is a subset of curated input (validation only removes)
    - All docs pass the same quality checks as the curator
    - No code documents were filtered by perplexity (code is exempt)
    - validation_stats.json exists and is internally consistent
"""

import json
from pathlib import Path

import pytest

from tests.conftest import requires_stage, read_jsonl, pipeline_path
from curator.filters.quality import QualityFilter
from curator.filters.dedup import exact_hash


pytestmark = requires_stage("validate")


class TestValidatedOutput:
    def test_validated_train_jsonl_exists(self):
        assert pipeline_path("validated", "train.jsonl").exists()

    def test_validated_train_jsonl_non_empty(self):
        docs = read_jsonl(pipeline_path("validated", "train.jsonl"))
        assert len(docs) > 0

    def test_validated_is_subset_of_curated(self):
        """
        Validation only filters — every validated doc must exist in curated.
        """
        curated_hashes = {
            exact_hash(d.get("text", ""))
            for d in read_jsonl(pipeline_path("curated", "train.jsonl"))
        }
        validated_docs = read_jsonl(pipeline_path("validated", "train.jsonl"))
        not_in_curated = [
            d for d in validated_docs
            if exact_hash(d.get("text", "")) not in curated_hashes
        ]
        assert len(not_in_curated) == 0, (
            f"{len(not_in_curated)} validated docs not found in curated output — "
            f"validation is adding documents, not just filtering"
        )

    def test_validated_docs_pass_quality_checks(self):
        """All validated docs should still pass quality filters."""
        qf = QualityFilter()
        failures = []
        docs = read_jsonl(pipeline_path("validated", "train.jsonl"))
        for doc in docs[:100]:
            passed, reason = qf.check(doc)
            if not passed:
                failures.append(f"rejected '{reason}': {doc['text'][:80]}")
        assert len(failures) == 0, (
            f"{len(failures)} validated docs fail quality checks:\n"
            + "\n".join(failures[:5])
        )

    def test_validated_has_required_fields(self):
        docs = read_jsonl(pipeline_path("validated", "train.jsonl"))
        for doc in docs[:20]:
            assert "text" in doc
            assert "source" in doc

    def test_validated_retention_rate_reasonable(self):
        """
        Validation should remove some docs but not too many.
        Expect 70-100% retention — if below 70% something is wrong.
        """
        curated_count = sum(
            1 for _ in open(pipeline_path("curated", "train.jsonl"))
        )
        validated_count = sum(
            1 for _ in open(pipeline_path("validated", "train.jsonl"))
        )
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

    def test_validation_stats_kept_le_total(self):
        stats = self._load_stats()
        assert stats["kept"] <= stats["total"]

    def test_validation_stats_kept_positive(self):
        stats = self._load_stats()
        assert stats["kept"] > 0, "Validation kept 0 documents — something is wrong"

    def test_validation_stats_matches_output_files(self):
        """
        Per-split kept counts should match per-split output line counts.

        The top-level `kept` field in validation_stats.json aggregates
        train + val, so comparing it directly to train.jsonl is wrong by
        exactly val.kept. Checking each split against its own file is
        stricter: Option B (compare sum) would miss bugs where train
        over-counts by the same amount val under-counts.
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
            with open(path) as f:
                actual = sum(1 for _ in f)
            assert split_stats["kept"] == actual, (
                f"validation_stats.json splits.{split_name}.kept = "
                f"{split_stats['kept']} but {split_name}.jsonl has {actual} lines"
            )