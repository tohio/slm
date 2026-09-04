from config import LONG_DOCUMENT_SEGMENT_SOURCES
from curator.filters.segments import (
    segment_long_document,
    split_long_text,
)


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _long_prose(paragraphs: int = 30) -> str:
    return "\n\n".join(
        (
            f"Paragraph {index} contains a distinct explanation. "
            + "supporting words " * 40
        ).strip()
        for index in range(paragraphs)
    )


def test_long_prose_is_split_without_content_loss_or_overlap():
    text = _long_prose()

    chunks = split_long_text(text, max_chars=1_000, min_chars=100)

    assert len(chunks) > 1
    assert all(100 <= len(chunk) <= 1_000 for chunk in chunks)
    assert _normalized(" ".join(chunks)) == _normalized(text)


def test_tiny_final_tail_is_rebalanced():
    text = ("word " * 399).strip() + "\n\nend."

    chunks = split_long_text(text, max_chars=1_000, min_chars=100)

    assert all(100 <= len(chunk) <= 1_000 for chunk in chunks)
    assert _normalized(" ".join(chunks)) == _normalized(text)


def test_segments_preserve_source_fields_and_record_parent_identity():
    record = {
        "source": "wikipedia",
        "title": "Example",
        "url": "https://example.invalid/article",
        "text": _long_prose(),
    }

    first = segment_long_document(
        record,
        eligible_sources=LONG_DOCUMENT_SEGMENT_SOURCES,
        max_chars=1_000,
        min_chars=100,
    )
    second = segment_long_document(
        record,
        eligible_sources=LONG_DOCUMENT_SEGMENT_SOURCES,
        max_chars=1_000,
        min_chars=100,
    )

    assert first == second
    assert all(segment["title"] == "Example" for segment in first)
    assert all(segment["url"] == record["url"] for segment in first)
    assert [
        segment["long_document_segment"]["index"] for segment in first
    ] == list(range(len(first)))
    assert all(
        segment["long_document_segment"]["count"] == len(first)
        for segment in first
    )
    assert len({
        segment["long_document_segment"]["parent_sha256"]
        for segment in first
    }) == 1


def test_source_contract_segments_only_bounded_prose():
    assert {
        "common_crawl",
        "fineweb",
        "fineweb_edu",
        "wikipedia",
        "pes2o",
        "stackexchange",
    } == set(LONG_DOCUMENT_SEGMENT_SOURCES)
    assert "pg19" not in LONG_DOCUMENT_SEGMENT_SOURCES
    assert "stack_v1" not in LONG_DOCUMENT_SEGMENT_SOURCES
    assert "synthetic_pretrain" not in LONG_DOCUMENT_SEGMENT_SOURCES

    record = {"source": "stack_v1", "text": _long_prose()}
    assert segment_long_document(
        record,
        eligible_sources=LONG_DOCUMENT_SEGMENT_SOURCES,
        max_chars=1_000,
        min_chars=100,
    ) == [record]
