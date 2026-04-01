"""
curator/sources/common_crawl.py
---------------------------------
Common Crawl data source.

Downloads WARC files from Common Crawl via HTTPS (data.commoncrawl.org),
extracts clean text using trafilatura, filters to English, and writes to
sharded JSONL.

Common Crawl is the largest freely available web crawl — petabytes of
raw HTML from billions of web pages. Quality varies significantly, so
aggressive filtering is applied in the validation stage.

This module handles only extraction — filtering and dedup happen in
curator/filters/.

Note: WARCs are fetched via HTTPS rather than direct S3 access. Direct
S3 access to the commoncrawl bucket fails when the instance has an IAM
role attached, as boto3 uses the role credentials which are rejected by
the bucket policy. HTTPS via CloudFront works reliably regardless of
instance credentials.

Output: JSONL with one document per line:
    {
        "text": "...",
        "source": "common_crawl",
        "url": "...",
        "crawl": "CC-MAIN-2024-10",
        "language": "en"
    }

Target contribution: ~2.1B tokens (~70% of 3B token 125M training mix).

Common Crawl crawl IDs: https://commoncrawl.org/the-data/get-started/
    Recent crawls: CC-MAIN-2024-10, CC-MAIN-2023-50, CC-MAIN-2023-40 ...

Usage:
    from curator.sources.common_crawl import CommonCrawlSource
    source = CommonCrawlSource(
        output_dir=Path("data/raw/common_crawl"),
        crawls=["CC-MAIN-2024-10"],
        max_segments=5,
    )
    source.download()
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator

import requests
import trafilatura
from langdetect import detect, LangDetectException
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

log = logging.getLogger(__name__)

# Common Crawl base URL — HTTPS via CloudFront, no credentials needed
CC_BASE_URL = "https://data.commoncrawl.org"
CC_PATHS_URL = f"{CC_BASE_URL}/crawl-data/{{crawl}}/warc.paths.gz"

# Default crawls to use — recent, high quality
DEFAULT_CRAWLS = [
    "CC-MAIN-2024-10",
    "CC-MAIN-2023-50",
]


class CommonCrawlSource:
    """
    Downloads and extracts text from Common Crawl WARC files.

    Fetches WARCs via HTTPS (data.commoncrawl.org) rather than direct S3
    access. This avoids IAM credential conflicts on EC2 instances with
    attached roles.

    Uses trafilatura for text extraction — it removes boilerplate,
    navigation, ads, and other non-content HTML elements, leaving
    clean article text.

    Language detection via langdetect filters to English only.

    Args:
        output_dir: Directory to write output JSONL files.
        crawls: List of Common Crawl crawl IDs to use.
        max_segments: Maximum WARC segments to process per crawl.
            Each segment is ~1GB and contains ~35k documents.
            Set to None to process all segments (very large).
        min_text_length: Minimum extracted text character length.
        shard_size: Number of documents per output JSONL shard.
        num_workers: Number of parallel WARC processing workers.
    """

    SOURCE_TAG = "common_crawl"

    def __init__(
        self,
        output_dir: Path,
        crawls: list[str] = DEFAULT_CRAWLS,
        max_segments: int | None = 10,
        min_text_length: int = 300,
        shard_size: int = 50_000,
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.crawls = crawls
        self.max_segments = max_segments
        self.min_text_length = min_text_length
        self.shard_size = shard_size
        self.num_workers = num_workers
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """
        Download and process Common Crawl WARCs.

        Returns:
            List of paths to output JSONL files.
        """
        output_files = []
        shard_idx = 0
        buffer = []
        total_written = 0
        total_skipped = 0

        for crawl in self.crawls:
            log.info(f"Processing crawl: {crawl}")
            warc_paths = self._get_warc_paths(crawl)

            if self.max_segments is not None:
                warc_paths = warc_paths[: self.max_segments]
                log.info(f"  Limited to {len(warc_paths)} segments")

            for warc_path in tqdm(warc_paths, desc=f"{crawl}", unit="segment"):
                for record in self._process_warc(warc_path, crawl):
                    if record is None:
                        total_skipped += 1
                        continue

                    buffer.append(record)

                    if len(buffer) >= self.shard_size:
                        path = self._write_shard(buffer, shard_idx)
                        output_files.append(path)
                        shard_idx += 1
                        total_written += len(buffer)
                        buffer = []

        if buffer:
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)

        log.info(
            f"Common Crawl complete — "
            f"written: {total_written:,}, "
            f"skipped: {total_skipped:,}, "
            f"shards: {len(output_files)}"
        )
        return output_files

    def _get_warc_paths(self, crawl: str) -> list[str]:
        """
        Fetch the list of WARC segment paths for a given crawl.

        Downloads the gzipped paths file from Common Crawl's index.
        """
        url = CC_PATHS_URL.format(crawl=crawl)
        log.info(f"Fetching WARC paths from {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = gzip.decompress(response.content).decode("utf-8")
            paths = [line.strip() for line in content.splitlines() if line.strip()]
            log.info(f"  Found {len(paths):,} WARC segments")
            return paths
        except Exception as e:
            log.error(f"Failed to fetch WARC paths for {crawl}: {e}")
            return []

    def _process_warc(self, warc_path: str, crawl: str) -> Iterator[dict | None]:
        """
        Stream and process a single WARC segment via HTTPS.

        Downloads the WARC in streaming mode, iterates over HTTP response
        records, extracts text with trafilatura, and yields clean records.
        """
        url = f"{CC_BASE_URL}/{warc_path}"
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            for record in ArchiveIterator(response.raw):
                # Only process HTTP response records
                if record.rec_type != "response":
                    continue

                target_url = record.rec_headers.get_header("WARC-Target-URI", "")
                content_type = record.http_headers.get_header("Content-Type", "")

                # Only process HTML
                if "text/html" not in content_type:
                    continue

                try:
                    html = record.content_stream().read()
                    text = self._extract_text(html, target_url)
                    if text is None:
                        yield None
                        continue

                    yield {
                        "text": text,
                        "source": self.SOURCE_TAG,
                        "url": target_url,
                        "crawl": crawl,
                        "language": "en",
                    }

                except Exception:
                    yield None

        except Exception as e:
            log.warning(f"Failed to process {warc_path}: {e}")

    def _extract_text(self, html: bytes, url: str) -> str | None:
        """
        Extract clean text from HTML using trafilatura.

        trafilatura removes boilerplate (navigation, ads, footers),
        extracts the main content, and returns clean text. Returns None
        if extraction fails or the text doesn't meet quality thresholds.
        """
        try:
            text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=True,
            )
        except Exception:
            return None

        if text is None or len(text) < self.min_text_length:
            return None

        # Language detection — filter to English only
        try:
            lang = detect(text[:500])  # detect on first 500 chars for speed
            if lang != "en":
                return None
        except LangDetectException:
            return None

        return text.strip()

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write a list of records to a JSONL shard file."""
        path = self.output_dir / f"cc_{shard_idx:04d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} documents → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("cc_*.jsonl"))
        total_docs = 0
        total_chars = 0
        crawl_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard) as f:
                for line in f:
                    record = json.loads(line)
                    total_docs += 1
                    total_chars += len(record["text"])
                    crawl = record.get("crawl", "unknown")
                    crawl_counts[crawl] = crawl_counts.get(crawl, 0) + 1

        return {
            "shards": len(shards),
            "documents": total_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": total_chars // max(total_docs, 1),
            "estimated_tokens": total_chars // 4,
            "by_crawl": crawl_counts,
        }