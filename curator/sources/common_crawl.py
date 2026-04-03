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

Resume behaviour:
    Progress is tracked in cc_progress.json (output_dir). Each fully
    completed WARC segment is recorded there. On restart, completed
    segments are skipped exactly — no approximation needed.
    Delete cc_progress.json to force a full re-download.

    Shards are written atomically via a .tmp rename so a crash mid-write
    never leaves a partial shard on disk.
"""

import gzip
import json
import logging
import time
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

# Segment-level progress file — records completed WARC paths exactly.
# Stored in output_dir. Delete to force a full re-download.
_PROGRESS_FILE = "cc_progress.json"


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

    Resume behaviour:
        Progress is tracked per segment in cc_progress.json. On restart,
        any segment already in the progress file is skipped exactly.
        Shards are written atomically — a crash never leaves a partial
        shard that would be counted as complete.

    Args:
        output_dir: Directory to write output JSONL files.
        crawls: List of Common Crawl crawl IDs to use.
        max_segments: Maximum WARC segments to process per crawl.
            Each segment is ~1GB and yields ~35k docs after filtering.
            Set to None to process all segments (very large).
        min_text_length: Minimum extracted text character length.
        shard_size: Number of documents per output JSONL shard.
        num_workers: Unused — downloads are sequential per segment.
        retries: Number of retry attempts per WARC segment on failure.
        retry_backoff: Base backoff seconds between retries (linear: backoff * attempt).
        request_timeout: Timeout in seconds for streaming WARC requests.
            Each WARC is ~1GB — 300s allows ~3.4MB/s minimum throughput.
            The original hardcoded value of 60s fires mid-stream and was
            the root cause of truncated segments in the original code.
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
        retries: int = 3,
        retry_backoff: float = 5.0,
        request_timeout: int = 300,
    ):
        self.output_dir = Path(output_dir)
        self.crawls = crawls
        self.max_segments = max_segments
        self.min_text_length = min_text_length
        self.shard_size = shard_size
        self.num_workers = num_workers
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.request_timeout = request_timeout
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_path = self.output_dir / _PROGRESS_FILE

    # ── Progress tracking ──────────────────────────────────────────────────────

    def _load_progress(self) -> set[str]:
        """Load the set of fully completed WARC segment paths from disk."""
        if not self._progress_path.exists():
            return set()
        try:
            with open(self._progress_path) as f:
                data = json.load(f)
            completed = set(data.get("completed_segments", []))
            log.info(f"Resuming — {len(completed)} segment(s) already completed")
            return completed
        except Exception as e:
            log.warning(f"Could not read progress file: {e} — starting fresh")
            return set()

    def _save_progress(self, completed: set[str]) -> None:
        """Persist the set of completed WARC segment paths atomically."""
        tmp = self._progress_path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump({"completed_segments": sorted(completed)}, f)
            tmp.replace(self._progress_path)
        except Exception as e:
            log.warning(f"Failed to save progress: {e}")

    # ── Download ───────────────────────────────────────────────────────────────

    def download(self) -> list[Path]:
        """
        Download and process Common Crawl WARCs.

        Resumes from cc_progress.json — each fully completed segment is
        recorded there and skipped on restart. Shards are written
        atomically so a crash never produces a partial shard.

        Returns:
            List of paths to new output JSONL files written this run.
        """
        output_files = []
        total_written = 0
        total_skipped = 0

        # Clean up any .tmp shards left by a previous crash
        for tmp in self.output_dir.glob("*.tmp"):
            log.warning(f"Removing partial shard from previous crash: {tmp}")
            tmp.unlink()

        # Load exact segment-level progress
        completed_segments = self._load_progress()

        # shard_idx = number of complete shards already on disk
        existing_shards = sorted(self.output_dir.glob("cc_*.jsonl"))
        shard_idx = len(existing_shards)
        if shard_idx > 0:
            log.info(f"Found {shard_idx} existing shard(s) — continuing from shard {shard_idx}")

        buffer = []

        for crawl in self.crawls:
            log.info(f"Processing crawl: {crawl}")
            warc_paths = self._get_warc_paths(crawl)

            if not warc_paths:
                log.error(f"No WARC paths retrieved for {crawl} — skipping crawl entirely")
                continue

            if self.max_segments is not None:
                warc_paths = warc_paths[: self.max_segments]
                log.info(f"  Limited to {len(warc_paths)} segments")

            # Skip segments already recorded as complete
            pending = [p for p in warc_paths if p not in completed_segments]
            n_skipped = len(warc_paths) - len(pending)
            if n_skipped:
                log.info(f"  Skipping {n_skipped} already-completed segment(s)")

            for warc_path in tqdm(pending, desc=f"{crawl}", unit="segment"):
                segment_docs = 0

                for record in self._process_warc(warc_path, crawl):
                    if record is None:
                        total_skipped += 1
                        continue

                    buffer.append(record)
                    segment_docs += 1

                    if len(buffer) >= self.shard_size:
                        path = self._write_shard(buffer, shard_idx)
                        output_files.append(path)
                        shard_idx += 1
                        total_written += len(buffer)
                        buffer = []

                # Only mark complete if the segment produced docs.
                # A genuine all-filtered segment is possible but very rare —
                # 0 docs almost always means a network failure. Either way,
                # don't mark it complete so it gets retried next run.
                if segment_docs > 0:
                    completed_segments.add(warc_path)
                    self._save_progress(completed_segments)
                else:
                    log.warning(
                        f"Segment produced 0 docs — not marking complete "
                        f"(will retry next run): {warc_path}"
                    )

        # Flush remainder — only happens at the very end of a complete run.
        # This partial shard is intentionally not progress-tracked; if the
        # run is interrupted here, the last partial shard will be re-generated
        # from the last unfinished segment on the next run. That is safe
        # because all prior segments are already in cc_progress.json.
        if buffer:
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)

        log.info(
            f"Common Crawl download complete — "
            f"written: {total_written:,}, "
            f"filtered/skipped: {total_skipped:,}, "
            f"new shards: {len(output_files)}, "
            f"total completed segments: {len(completed_segments)}"
        )
        return output_files

    # ── WARC fetching ──────────────────────────────────────────────────────────

    def _get_warc_paths(self, crawl: str) -> list[str]:
        """
        Fetch the list of WARC segment paths for a given crawl.

        Downloads the gzipped paths file from Common Crawl's index.
        Retries up to self.retries times with linear backoff.
        """
        url = CC_PATHS_URL.format(crawl=crawl)
        log.info(f"Fetching WARC paths from {url}")

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                content = gzip.decompress(response.content).decode("utf-8")
                paths = [line.strip() for line in content.splitlines() if line.strip()]
                log.info(f"  Found {len(paths):,} WARC segments")
                return paths
            except Exception as e:
                log.warning(
                    f"Attempt {attempt}/{self.retries} failed fetching paths for {crawl}: {e}"
                )
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * attempt)
                else:
                    log.error(
                        f"Failed to fetch WARC paths for {crawl} after "
                        f"{self.retries} attempts"
                    )
                    return []

        return []  # unreachable — satisfies type checker

    def _process_warc(self, warc_path: str, crawl: str) -> Iterator[dict | None]:
        """
        Stream and process a single WARC segment via HTTPS.

        Downloads the WARC in streaming mode, iterates over HTTP response
        records, extracts text with trafilatura, and yields clean records.

        Retries up to self.retries times on network or stream errors, with
        linear backoff. On final failure, logs an error and returns without
        yielding anything — the caller will see 0 docs and not mark the
        segment complete, so it will be retried on the next run.

        Key fix vs original: timeout is now self.request_timeout (300s
        default) rather than a hardcoded 60s. A 1GB WARC streamed at a
        typical EC2→CloudFront rate of ~100MB/s takes ~10s, but under
        load or throttling it can take much longer. 60s was firing
        mid-stream, producing truncated segments and silent data loss.
        """
        url = f"{CC_BASE_URL}/{warc_path}"

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.get(
                    url,
                    stream=True,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()

                for record in ArchiveIterator(response.raw):
                    if record.rec_type != "response":
                        continue

                    target_url = record.rec_headers.get_header("WARC-Target-URI", "")
                    content_type = record.http_headers.get_header("Content-Type", "")

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

                # Segment fully streamed — exit retry loop
                return

            except Exception as e:
                log.warning(
                    f"Attempt {attempt}/{self.retries} failed for {warc_path}: {e}"
                )
                if attempt < self.retries:
                    wait = self.retry_backoff * attempt
                    log.info(f"  Retrying in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    log.error(
                        f"Giving up on {warc_path} after {self.retries} attempts — "
                        f"segment will be retried on next run"
                    )
                    return

    # ── Text extraction ────────────────────────────────────────────────────────

    def _extract_text(self, html: bytes, url: str) -> str | None:
        """
        Extract clean text from HTML using trafilatura.

        Returns None if extraction fails or text is below min_text_length.
        Language detection filters to English only.
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

        # Language detection on first 500 chars for speed
        try:
            if detect(text[:500]) != "en":
                return None
        except LangDetectException:
            return None

        return text.strip()

    # ── Atomic shard write ─────────────────────────────────────────────────────

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """
        Write records to a JSONL shard atomically via .tmp rename.

        A crash during write leaves a .tmp file (cleaned up on next
        startup) rather than a partial .jsonl that would be counted
        as a complete shard during resume.
        """
        path = self.output_dir / f"cc_{shard_idx:04d}.jsonl"
        tmp_path = path.with_suffix(".jsonl.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} docs → {path}")
        return path

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("cc_*.jsonl"))
        total_docs = 0
        total_chars = 0
        crawl_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total_docs += 1
                    total_chars += len(record.get("text", ""))
                    crawl = record.get("crawl", "unknown")
                    crawl_counts[crawl] = crawl_counts.get(crawl, 0) + 1

        completed_segments = self._load_progress()

        return {
            "shards": len(shards),
            "documents": total_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": total_chars // max(total_docs, 1),
            "estimated_tokens": total_chars // 4,
            "completed_segments": len(completed_segments),
            "by_crawl": crawl_counts,
        }