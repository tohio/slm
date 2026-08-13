"""Focused tests for durable filter and deduplication audit counts."""

import pytest

from curator.filters.dedup import build_dedup_stats
from curator.filters.quality import QualityFilter
from curator.scripts.curate import _merge_filter_stats


def test_quality_filter_stats_snapshot_is_machine_readable():
    quality_filter = QualityFilter()
    quality_filter.check({"text": "short", "source": "wikipedia"})

    assert quality_filter.stats_snapshot() == {
        "total": 1,
        "kept": 0,
        "rejected": 1,
        "rejection_reasons": {"too_short": 1},
        "fasttext_prediction_errors": 0,
    }


def test_filter_worker_stats_merge_by_reason():
    aggregate = {
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
    _merge_filter_stats(
        aggregate,
        {
            "input_documents": 8,
            "segmented_input_documents": 2,
            "produced_segments": 10,
            "total": 10,
            "kept": 7,
            "rejected": 3,
            "rejection_reasons": {"too_short": 2, "non_english": 1},
            "fasttext_prediction_errors": 1,
        },
    )
    _merge_filter_stats(
        aggregate,
        {
            "input_documents": 5,
            "segmented_input_documents": 0,
            "produced_segments": 5,
            "total": 5,
            "kept": 4,
            "rejected": 1,
            "rejection_reasons": {"too_short": 1},
            "fasttext_prediction_errors": 0,
        },
    )

    assert aggregate == {
        "shards": 2,
        "input_documents": 13,
        "segmented_input_documents": 2,
        "produced_segments": 15,
        "total": 15,
        "kept": 11,
        "rejected": 4,
        "rejection_reasons": {"too_short": 3, "non_english": 1},
        "fasttext_prediction_errors": 1,
    }


def test_dedup_stats_separate_exact_and_fuzzy_removals():
    stats = build_dedup_stats(
        source="common_crawl",
        exact_stats={"total": 100, "kept": 90, "exact_duplicates": 10},
        final_documents=75,
        fuzzy_enabled=True,
        fuzzy_partition_field="crawl",
        fuzzy_partitions=["CC-MAIN-2024-10", "CC-MAIN-2023-50"],
    )

    assert stats["exact_duplicate_documents"] == 10
    assert stats["fuzzy_duplicate_documents"] == 15
    assert stats["final_documents"] == 75
    assert stats["fuzzy_partitions"] == [
        "CC-MAIN-2023-50",
        "CC-MAIN-2024-10",
    ]


def test_dedup_stats_reject_removals_when_fuzzy_is_disabled():
    with pytest.raises(RuntimeError, match="fuzzy dedup is disabled"):
        build_dedup_stats(
            source="synthetic_math",
            exact_stats={"total": 10, "kept": 9, "exact_duplicates": 1},
            final_documents=8,
            fuzzy_enabled=False,
        )
