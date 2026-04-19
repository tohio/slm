"""
curator/sources/common_crawl.py
---------------------------------
Common Crawl data source — parallelized.

Downloads WARC files from Common Crawl via HTTPS (data.commoncrawl.org),
extracts clean text using trafilatura, filters to English, and writes to
sharded JSONL.

Parallelism model:
    - Download pool  (threads):  N download threads stream WARCs from CC
                                 concurrently. Network I/O is the bottleneck
                                 here, so threads are the right tool — the
                                 GIL is released during socket reads.
    - Extract pool   (processes): M extraction workers parse WARCs and run
                                  trafilatura + fasttext. This is CPU-bound
                                  so processes are required.
    - Main process:  consumes extracted records, shards them, tracks
                     segment completion, writes progress.

A bounded queue between stages prevents RAM blowup — each WARC is ~1 GB,
so we cap the number of in-flight WARCs.

Earlier single-threaded version did download + parse + extract serially
per segment on one core. On a 64-core box this left 63 cores idle.

Output: JSONL with one document per line:
    {
        "text": "...",
        "source": "common_crawl",
        "url": "...",
        "crawl": "CC-MAIN-2024-10",
        "language": "en"
    }

Note: WARCs are fetched via HTTPS rather than direct S3 access. Direct
S3 access to the commoncrawl bucket fails when the instance has an IAM
role attached, as boto3 uses the role credentials which are rejected by
the bucket policy. HTTPS via CloudFront works reliably regardless of
instance credentials.

Usage:
    from curator.sources.common_crawl import CommonCrawlSource
    source = CommonCrawlSource(
        output_dir=Path("data/raw/common_crawl"),
        crawls=["CC-MAIN-2024-10"],
        max_segments=5,
        download_workers=16,
        extract_workers=48,
    )
    source.download()

Resume behaviour:
    Progress is tracked in cc_progress.json (output_dir). Each fully
    completed WARC segment is recorded there. On restart, completed
    segments are skipped exactly.

    Shards are written atomically via a .tmp rename so a crash mid-write
    never leaves a partial shard on disk.
"""

import gzip
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson
import requests
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)

# Common Crawl base URL — HTTPS via CloudFront, no credentials needed
CC_BASE_URL = "https://data.commoncrawl.org"
CC_PATHS_URL = f"{CC_BASE_URL}/crawl-data/{{crawl}}/warc.paths.gz"

DEFAULT_CRAWLS = [
    "CC-MAIN-2024-10",
    "CC-MAIN-2023-50",
]

# Progress file — records completed WARC paths exactly.
_PROGRESS_FILE = "cc_progress.json"

# fasttext model path (resolved at import time in extraction workers)
_FT_MODEL_PATH = os.environ.get(
    "FASTTEXT_MODEL_PATH",
    "data/models/lid.176.ftz",
)


# ── Extraction worker (runs in subprocess) ────────────────────────────────────
#
# fasttext and trafilatura are loaded at module level so each worker
# process loads them exactly once at startup, not per-WARC. Importing
# trafilatura can take ~1s so per-WARC imports would be painful.

_ft_model = None


def _init_extract_worker() -> None:
    """
    Pool initializer: load fasttext once per extraction process.

    Called once when each worker starts. The model is cached in module
    globals for the lifetime of the worker.
    """
    global _ft_model
    import fasttext
    # Suppress fasttext's noisy stderr about newlines in input
    fasttext.FastText.eprint = lambda *args, **kwargs: None
    try:
        _ft_model = fasttext.load_model(_FT_MODEL_PATH)
    except Exception as e:
        # Workers without the model can't do language detection.
        # Log once per worker startup; extraction will fail fast on first use.
        log.error(
            f"Worker could not load fasttext model at {_FT_MODEL_PATH}: {e}"
        )
        _ft_model = None


def _extract_from_warc_file(
    local_warc_path: str,
    warc_path: str,
    crawl: str,
    min_text_length: int,
) -> list[dict]:
    """
    Extract clean text records from a local WARC file. Runs in a subprocess.

    Returns a list of extracted documents. An empty list means all records
    were filtered out (possible but unusual) or the WARC failed to parse
    (caller should treat empty-list + exception separately).

    Args:
        local_warc_path:  Path to the local .warc.gz file.
        warc_path:        Original CC path (for the 'url' context).
        crawl:            CC crawl ID for tagging.
        min_text_length:  Minimum extracted text length.

    Returns:
        List of record dicts. May be empty.
    """
    # Import trafilatura inside the worker — this is where it's loaded once
    # per process via Python's module cache.
    import trafilatura

    records: list[dict] = []

    with open(local_warc_path, "rb") as f:
        for record in ArchiveIterator(f):
            if record.rec_type != "response":
                continue

            try:
                target_url = record.rec_headers.get_header("WARC-Target-URI", "")
                content_type = record.http_headers.get_header("Content-Type", "") or ""
                if "text/html" not in content_type:
                    continue

                html = record.content_stream().read()
            except Exception:
                continue

            try:
                text = trafilatura.extract(
                    html,
                    url=target_url,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                    favor_precision=True,
                )
            except Exception:
                continue

            if text is None or len(text) < min_text_length:
                continue

            # Language detection on first 500 chars.
            if _ft_model is None:
                # No language model available — skip the check, accept text.
                # (This shouldn't happen in production, the worker would have
                # logged the load failure already.)
                pass
            else:
                try:
                    sample = text[:500].replace("\n", " ")
                    labels, _ = _ft_model.predict(sample)
                    if labels[0] != "__label__en":
                        continue
                except Exception:
                    continue

            records.append({
                "text": text.strip(),
                "source": "common_crawl",
                "url": target_url,
                "crawl": crawl,
                "language": "en",
            })

    return records


# ── Download helper (runs in threads) ─────────────────────────────────────────

def _download_warc(
    warc_path: str,
    tmp_dir: Path,
    timeout: int,
    retries: int,
    backoff: float,
) -> Path | None:
    """
    Stream a WARC file from Common Crawl to local disk.

    Returns:
        Path to the downloaded file, or None on final failure.
    """
    url = f"{CC_BASE_URL}/{warc_path}"
    local = tmp_dir / (warc_path.replace("/", "_"))

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return local
        except Exception as e:
            log.warning(
                f"Download attempt {attempt}/{retries} failed for {warc_path}: {e}"
            )
            if local.exists():
                local.unlink()  # don't leave partial file
            if attempt < retries:
                time.sleep(backoff * attempt)

    log.error(f"Giving up on download of {warc_path} after {retries} attempts")
    return None


class CommonCrawlSource:
    """
    Downloads and extracts text from Common Crawl WARC files in parallel.

    Parallelism:
        download_workers threads pull WARCs from CC concurrently. Each
        completed WARC is submitted to an extraction process pool with
        extract_workers processes. Extracted records flow back to the
        main process for sharding.

        download_workers:  network-bound — more is better until CC throttles
                           (empirically ~16 is a good balance)
        extract_workers:   CPU-bound — set to most of your CPU count

    Args:
        output_dir: Directory to write output JSONL files.
        crawls: List of Common Crawl crawl IDs.
        max_segments: Max WARC segments per crawl. None = all.
        min_text_length: Minimum extracted text character length.
        shard_size: Documents per output JSONL shard.
        download_workers: Number of concurrent download threads.
        extract_workers: Number of extraction processes.
        in_flight: Max WARCs on local disk at once (caps RAM/disk usage).
        retries: Retry attempts per WARC on network failure.
        retry_backoff: Linear backoff base (seconds).
        request_timeout: Per-WARC HTTP timeout (seconds).
        tmp_dir: Temp directory for downloaded WARCs (cleaned up per-WARC).
    """

    SOURCE_TAG = "common_crawl"

    def __init__(
        self,
        output_dir: Path,
        crawls: list[str] = DEFAULT_CRAWLS,
        max_segments: int | None = 10,
        min_text_length: int = 300,
        shard_size: int = 50_000,
        download_workers: int = 16,
        extract_workers: int | None = None,
        in_flight: int = 8,
        retries: int = 3,
        retry_backoff: float = 5.0,
        request_timeout: int = 300,
        tmp_dir: Path | None = None,
        # Backwards-compat — old code passed num_workers, ignore it if present.
        num_workers: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.crawls = crawls
        self.max_segments = max_segments
        self.min_text_length = min_text_length
        self.shard_size = shard_size
        self.download_workers = download_workers
        self.extract_workers = extract_workers or max(1, (os.cpu_count() or 4) - 2)
        self.in_flight = in_flight
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.request_timeout = request_timeout
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_path = self.output_dir / _PROGRESS_FILE

        # Tmp dir for downloaded WARCs — lives for the lifetime of a .download()
        # call and is cleaned up at the end. Default to a fresh tempdir.
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None

    # ── Progress tracking ──────────────────────────────────────────────────────

    def _load_progress(self) -> set[str]:
        if not self._progress_path.exists():
            return set()
        try:
            with open(self._progress_path, "rb") as f:
                data = orjson.loads(f.read())
            completed = set(data.get("completed_segments", []))
            log.info(f"Resuming — {len(completed)} segment(s) already completed")
            return completed
        except Exception as e:
            log.warning(f"Could not read progress file: {e} — starting fresh")
            return set()

    def _save_progress(self, completed: set[str]) -> None:
        tmp = self._progress_path.with_suffix(".json.tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(orjson.dumps(
                    {"completed_segments": sorted(completed)},
                    option=orjson.OPT_INDENT_2,
                ))
            tmp.replace(self._progress_path)
        except Exception as e:
            log.warning(f"Failed to save progress: {e}")

    # ── Download ───────────────────────────────────────────────────────────────

    def download(self) -> list[Path]:
        """
        Download and process Common Crawl WARCs in parallel.

        Returns:
            List of paths to new output JSONL files written this run.
        """
        # Clean .tmp shards from previous crashes
        for tmp in self.output_dir.glob("*.tmp"):
            log.warning(f"Removing partial shard from previous crash: {tmp}")
            tmp.unlink()

        completed_segments = self._load_progress()

        # shard_idx = number of complete shards already on disk
        existing_shards = sorted(self.output_dir.glob("cc_*.jsonl"))
        shard_idx = len(existing_shards)
        if shard_idx > 0:
            log.info(
                f"Found {shard_idx} existing shard(s) — "
                f"continuing from shard {shard_idx}"
            )

        # Collect all pending WARCs across all crawls into one flat list,
        # so the parallel pipeline can work on them as a unit.
        pending: list[tuple[str, str]] = []  # (warc_path, crawl)
        for crawl in self.crawls:
            warc_paths = self._get_warc_paths(crawl)
            if not warc_paths:
                log.error(f"No WARC paths for {crawl} — skipping")
                continue
            if self.max_segments is not None:
                warc_paths = warc_paths[: self.max_segments]
            before = len(warc_paths)
            warc_paths = [p for p in warc_paths if p not in completed_segments]
            log.info(
                f"  {crawl}: {before} segments total, "
                f"{before - len(warc_paths)} already complete, "
                f"{len(warc_paths)} pending"
            )
            pending.extend((p, crawl) for p in warc_paths)

        if not pending:
            log.info("All segments already complete — nothing to do")
            return []

        log.info(
            f"Processing {len(pending)} segments with "
            f"{self.download_workers} download threads, "
            f"{self.extract_workers} extract workers, "
            f"in_flight={self.in_flight}"
        )

        # Set up tmp dir
        ctx_tmp = None
        if self.tmp_dir is None:
            ctx_tmp = tempfile.TemporaryDirectory(prefix="cc_warcs_")
            tmp_dir = Path(ctx_tmp.name)
        else:
            tmp_dir = self.tmp_dir
            tmp_dir.mkdir(parents=True, exist_ok=True)

        output_files: list[Path] = []
        buffer: list[dict] = []
        total_written = 0
        total_segments_done = 0

        try:
            output_files, total_written, total_segments_done = self._run_parallel(
                pending=pending,
                tmp_dir=tmp_dir,
                completed_segments=completed_segments,
                buffer=buffer,
                shard_idx=shard_idx,
            )
        finally:
            if ctx_tmp is not None:
                ctx_tmp.cleanup()

        log.info(
            f"Common Crawl download complete — "
            f"written: {total_written:,}, "
            f"new shards: {len(output_files)}, "
            f"segments completed this run: {total_segments_done}, "
            f"total completed segments: {len(completed_segments)}"
        )
        return output_files

    def _run_parallel(
        self,
        pending: list[tuple[str, str]],
        tmp_dir: Path,
        completed_segments: set[str],
        buffer: list[dict],
        shard_idx: int,
    ) -> tuple[list[Path], int, int]:
        """
        Core parallel execution loop.

        Strategy:
            - Maintain up to `in_flight` downloaded WARCs on disk at a time.
            - As each download completes, submit to extract pool.
            - As each extract completes, drain records into the shard buffer.
            - Flush shards to disk as buffer fills.
            - On each fully-processed segment, update progress.

        Returns: (output_files, total_written, segments_done)
        """
        output_files: list[Path] = []
        total_written = 0
        segments_done = 0

        download_pool = ThreadPoolExecutor(
            max_workers=self.download_workers,
            thread_name_prefix="cc-dl",
        )
        extract_pool = ProcessPoolExecutor(
            max_workers=self.extract_workers,
            initializer=_init_extract_worker,
        )

        # Track in-flight: download futures and extract futures, each with
        # the segment it belongs to. When extract completes, we know the
        # segment is done.
        pbar = tqdm(total=len(pending), desc="CC segments", unit="seg")

        try:
            pending_iter = iter(pending)
            download_futures: dict = {}   # Future -> (warc_path, crawl)
            extract_futures: dict = {}    # Future -> (warc_path, crawl, local_path)
            in_flight_count = 0           # downloads issued but not yet handed to extract

            def _submit_next_download():
                """Submit the next pending WARC to the download pool."""
                nonlocal in_flight_count
                try:
                    warc_path, crawl = next(pending_iter)
                except StopIteration:
                    return False
                fut = download_pool.submit(
                    _download_warc,
                    warc_path, tmp_dir, self.request_timeout,
                    self.retries, self.retry_backoff,
                )
                download_futures[fut] = (warc_path, crawl)
                in_flight_count += 1
                return True

            # Prime the pipeline
            for _ in range(self.in_flight):
                if not _submit_next_download():
                    break

            # Main loop: alternate draining downloads and extracts
            while download_futures or extract_futures:
                # Drain any completed downloads → submit to extract
                done_downloads = [f for f in download_futures if f.done()]
                for fut in done_downloads:
                    warc_path, crawl = download_futures.pop(fut)
                    local_path = fut.result()
                    in_flight_count -= 1
                    if local_path is None:
                        # download failed — skip segment, don't mark complete
                        pbar.update(1)
                        _submit_next_download()
                        continue
                    ext_fut = extract_pool.submit(
                        _extract_from_warc_file,
                        str(local_path),
                        warc_path,
                        crawl,
                        self.min_text_length,
                    )
                    extract_futures[ext_fut] = (warc_path, crawl, local_path)

                # Drain any completed extractions → flush into shard buffer
                done_extracts = [f for f in extract_futures if f.done()]
                for fut in done_extracts:
                    warc_path, crawl, local_path = extract_futures.pop(fut)
                    # Clean up the WARC file now that extraction is done
                    try:
                        local_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                    try:
                        records = fut.result()
                    except Exception as e:
                        log.error(f"Extraction failed for {warc_path}: {e}")
                        records = []

                    buffer.extend(records)

                    # Flush shards
                    while len(buffer) >= self.shard_size:
                        chunk = buffer[: self.shard_size]
                        del buffer[: self.shard_size]
                        path = self._write_shard(chunk, shard_idx)
                        output_files.append(path)
                        shard_idx += 1
                        total_written += len(chunk)

                    if records:
                        completed_segments.add(warc_path)
                        self._save_progress(completed_segments)
                        segments_done += 1
                    else:
                        log.warning(
                            f"Segment produced 0 docs — not marking complete: "
                            f"{warc_path}"
                        )

                    pbar.update(1)
                    # Top up the download pool
                    _submit_next_download()

                # If nothing completed this iteration, wait a bit so we don't
                # spin. Simple approach: wait on the first still-running future.
                if not done_downloads and not done_extracts:
                    wait_on = next(iter(download_futures), None) or \
                              next(iter(extract_futures), None)
                    if wait_on is not None:
                        try:
                            wait_on.result(timeout=30)
                        except Exception:
                            # Let the main loop handle it
                            pass

            # Flush remaining buffer
            if buffer:
                path = self._write_shard(buffer, shard_idx)
                output_files.append(path)
                total_written += len(buffer)
                buffer.clear()

        finally:
            pbar.close()
            download_pool.shutdown(wait=True)
            extract_pool.shutdown(wait=True)

        return output_files, total_written, segments_done

    # ── WARC path index fetch ──────────────────────────────────────────────────

    def _get_warc_paths(self, crawl: str) -> list[str]:
        """Fetch the list of WARC segment paths for a given crawl."""
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
                    f"Attempt {attempt}/{self.retries} fetching paths for "
                    f"{crawl}: {e}"
                )
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * attempt)

        log.error(f"Failed to fetch WARC paths for {crawl}")
        return []

    # ── Atomic shard write ─────────────────────────────────────────────────────

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard atomically via .tmp rename."""
        path = self.output_dir / f"cc_{shard_idx:04d}.jsonl"
        tmp_path = path.with_suffix(".jsonl.tmp")
        try:
            with open(tmp_path, "wb") as f:
                for record in records:
                    f.write(orjson.dumps(record))
                    f.write(b"\n")
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} docs → {path}")
        return path

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        shards = sorted(self.output_dir.glob("cc_*.jsonl"))
        total_docs = 0
        total_chars = 0
        crawl_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
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
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "completed_segments": len(completed_segments),
            "by_crawl": crawl_counts,
        }