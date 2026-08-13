"""Focused unit tests for source-aware pretraining quality filters."""

import pytest

from curator.filters.quality import QualityFilter


def _long_line(index: int, punctuated: bool = True) -> str:
    line = (
        f"The content of this page and the surrounding explanation form "
        f"unique English sentence number {index}"
    )
    return f"{line}." if punctuated else line


def test_fineweb_defaults_match_pinned_datatrove_contract():
    config = QualityFilter().config

    assert config.fineweb_sources == frozenset({"common_crawl"})
    assert config.fineweb_line_punct_thr == 0.12
    assert config.fineweb_short_line_thr == 0.67
    assert config.fineweb_short_line_length == 30
    assert config.fineweb_char_duplicates_ratio == 0.01
    assert config.fineweb_new_line_ratio == 0.3


def test_fineweb_line_punctuation_threshold_is_inclusive():
    quality = QualityFilter()
    at_threshold = "\n".join(
        _long_line(index, punctuated=index < 12) for index in range(100)
    )
    below_threshold = "\n".join(
        _long_line(index, punctuated=index < 11) for index in range(100)
    )

    assert quality._check_fineweb_quality(at_threshold) == (True, None)
    assert quality._check_fineweb_quality(below_threshold) == (
        False,
        "line_punct_ratio",
    )


def test_fineweb_short_line_threshold_is_inclusive():
    quality = QualityFilter()

    def build(short_lines: int) -> str:
        lines = [f"Brief item {index}." for index in range(short_lines)]
        lines.extend(_long_line(index) for index in range(short_lines, 100))
        return "\n".join(lines)

    assert quality._check_fineweb_quality(build(67)) == (True, None)
    assert quality._check_fineweb_quality(build(68)) == (
        False,
        "short_line_ratio",
    )


def test_fineweb_duplicate_character_ratio_rejects_repeated_lines():
    quality = QualityFilter()
    unique_lines = [_long_line(index) for index in range(100)]
    repeated_lines = unique_lines.copy()
    repeated_lines[-5:] = [unique_lines[0]] * 5

    assert quality._check_fineweb_quality("\n".join(unique_lines)) == (True, None)
    assert quality._check_fineweb_quality("\n".join(repeated_lines)) == (
        False,
        "char_dup_ratio",
    )


def test_fineweb_newline_to_word_ratio_rejects_list_like_text():
    quality = QualityFilter()
    list_like = "\n".join(
        f"singlewordtoken{index:02d}abcdefghijklmnopqrstu."
        for index in range(10)
    )

    assert quality._check_fineweb_quality(list_like) == (False, "list_ratio")


def test_fineweb_metrics_apply_only_to_raw_common_crawl():
    text = "\n".join(_long_line(index, punctuated=False) for index in range(10))

    common_crawl = QualityFilter()
    assert common_crawl.check({"text": text, "source": "common_crawl"}) == (
        False,
        "line_punct_ratio",
    )

    already_curated = QualityFilter()
    assert already_curated.check({"text": text, "source": "fineweb"}) == (
        True,
        None,
    )


def test_filter_rejects_record_source_mismatch():
    quality = QualityFilter()

    with pytest.raises(RuntimeError, match="does not match configured source"):
        quality.check(
            {"text": "short", "source": "synthetic_arithmetic"},
            expected_source="wikipedia",
        )
    assert quality.stats["total"] == 0


def test_filter_rejects_missing_or_unclassified_source():
    quality = QualityFilter()

    with pytest.raises(ValueError, match="exactly one filter family"):
        quality.check({"text": "content without a source"})
    with pytest.raises(ValueError, match="exactly one filter family"):
        quality.check({"text": "content", "source": "new_unclassified_source"})
    assert quality.stats["total"] == 0
