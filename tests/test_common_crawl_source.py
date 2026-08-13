"""Focused unit tests for raw Common Crawl source filtering."""

from curator.sources.common_crawl import _url_is_allowed


class _RecordingUrlFilter:
    def __init__(self, result):
        self.result = result
        self.document = None

    def filter(self, document):
        self.document = document
        return self.result


def test_common_crawl_url_filter_passes_url_in_document_metadata():
    url_filter = _RecordingUrlFilter(True)
    url = "https://example.org/educational/article"

    assert _url_is_allowed(url, url_filter=url_filter) is True
    assert url_filter.document.id == url
    assert url_filter.document.metadata == {"url": url}


def test_common_crawl_url_filter_honors_rejection_tuple():
    url_filter = _RecordingUrlFilter((False, "hard_blacklisted"))

    assert _url_is_allowed(
        "https://example.org/blocked-path",
        url_filter=url_filter,
    ) is False


def test_common_crawl_url_filter_rejects_missing_url_without_dispatch():
    url_filter = _RecordingUrlFilter(True)

    assert _url_is_allowed("", url_filter=url_filter) is False
    assert url_filter.document is None
