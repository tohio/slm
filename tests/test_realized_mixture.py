import copy

import pytest

from config.data_mix import ALL_SOURCES
from pretrain.data.mixture import (
    build_realized_mixture_report,
    configured_source_shares,
    validate_realized_mixture_report,
)


def _metadata(multiplier: int = 1) -> dict:
    source_counts = {
        source: {"documents": multiplier, "tokens": (index + 1) * multiplier}
        for index, source in enumerate(ALL_SOURCES)
    }
    return {
        "n_docs": sum(row["documents"] for row in source_counts.values()),
        "n_tokens": sum(row["tokens"] for row in source_counts.values()),
        "source_counts": source_counts,
    }


def test_configured_source_shares_expand_code_bucket():
    shares = configured_source_shares()

    assert set(shares) == set(ALL_SOURCES)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["stack_v1"] == pytest.approx(0.15 * 0.83)


def test_report_uses_combined_authoritative_token_counts():
    train = _metadata(2)
    val = _metadata(1)

    report = build_realized_mixture_report(train, val)

    assert report["status"] == "passed_structural_checks_report_only"
    assert report["contract"]["deviation_policy"]["threshold"] is None
    assert report["total_tokens"] == train["n_tokens"] + val["n_tokens"]
    assert sum(row["tokens"] for row in report["sources"].values()) == report[
        "total_tokens"
    ]
    assert report["top_level"]["code"]["realized_token_share"] == pytest.approx(
        sum(
            report["sources"][source]["tokens"]
            for source in (
                "stack_v1",
                "codesearchnet",
                "stack_smol",
                "jupyter",
                "conala",
            )
        )
        / report["total_tokens"]
    )


def test_report_rejects_unknown_sources():
    train = _metadata()
    val = _metadata()
    train["source_counts"]["unknown"] = train["source_counts"].pop(ALL_SOURCES[0])

    with pytest.raises(RuntimeError, match="configured source set"):
        build_realized_mixture_report(train, val)


def test_report_allows_source_absent_from_one_split():
    train = _metadata()
    val = _metadata()
    removed = val["source_counts"].pop(ALL_SOURCES[0])
    val["n_docs"] -= removed["documents"]
    val["n_tokens"] -= removed["tokens"]

    report = build_realized_mixture_report(train, val)

    assert report["sources"][ALL_SOURCES[0]]["splits"]["val"] == {
        "documents": 0,
        "tokens": 0,
    }


def test_report_rejects_source_totals_that_disagree_with_metadata():
    train = _metadata()
    val = _metadata()
    train["n_tokens"] += 1

    with pytest.raises(RuntimeError, match="source token counts sum"):
        build_realized_mixture_report(train, val)


def test_report_validation_rejects_stale_tokenized_metadata():
    train = _metadata()
    val = _metadata()
    report = build_realized_mixture_report(train, val)
    changed_train = copy.deepcopy(train)
    changed_train["source_counts"][ALL_SOURCES[0]]["tokens"] += 1
    changed_train["n_tokens"] += 1

    with pytest.raises(RuntimeError, match="stale"):
        validate_realized_mixture_report(report, changed_train, val)


def test_report_validation_rejects_tampered_realized_share():
    train = _metadata()
    val = _metadata()
    report = build_realized_mixture_report(train, val)
    report["sources"][ALL_SOURCES[0]]["realized_token_share"] = 1.0

    with pytest.raises(RuntimeError, match="contents do not match"):
        validate_realized_mixture_report(report, train, val)
