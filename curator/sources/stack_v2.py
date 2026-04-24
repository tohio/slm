"""
curator/sources/stack_v2.py
--------------------------------
the-stack-v2-dedup data source.

Loads `bigcode/the-stack-v2-dedup` filtered to 4 languages (Python, Go,
Rust, Shell) by iterating each language's data shard independently, and
fetches file content on-demand from the Software Heritage Archive (SWH).
Acts as the bulk code source at 50% of the code share, complementing
curated sources like CodeSearchNet.

Why per-language iteration?
    The-stack-v2-dedup stores records sharded by language under
    `data/<Language>/`. Streaming the full dataset iterates alphabetically
    through 600+ languages (AMPL, ABAP, ...) before reaching Python, Rust,
    Shell. That's billions of records of filter-and-skip before the first
    match — unusable at any scale, not just mini. Instead we load each
    target language's data_dir directly, iterate only records we care
    about, and stop when we hit our cap.

Why content fetching?
    The-stack-v2 stores only file metadata and blob IDs in the HF
    dataset — the actual source code is hosted on SWH and must be
    fetched per record. This is required by BigCode's content licensing
    arrangement with SWH. The HF dataset card documents the SWH content
    URL pattern.

Rate limiting and retries:
    SWH throttles per-IP. We keep the content-fetch thread pool modest
    (default 8 workers) and retry transient failures with backoff. An
    SWH_AUTH_TOKEN in the environment enables higher rate limits.

License note: Users must accept BigCode's Terms of Use on the HF dataset
page before first download. All fetched files carry permissive licenses
per the-stack-v2's filtering, but specific attribution requirements for
redistribution may apply — see the dataset card.

Output: JSONL with one file per line:
    {
        "text": "<source code>",
        "source": "stack_v2",
        "language": "Python",
        "repo": "...",
        "path": "...",
        "blob_id": "..."
    }

Usage:
    from curator.sources.stack_v2 import StackV2Source
    source = StackV2Source(output_dir=Path("data/raw/stack_v2"))
    source.download()
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
import requests
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)

# SWH content endpoint. Files stored by SHA1 blob ID.
# See https://docs.softwareheritage.org/devel/swh-web/uri-scheme-api.html
_SWH_CONTENT_URL = "https://archive.softwareheritage.org/api/1/content/sha1_git:{blob_id}/raw/"

# Languages targeted. Matches the locked decision from data mix planning:
# Python (curated CSN overlap), Go (curated CSN overlap), Rust (no CSN),
# Shell (no CSN). Gives deep + medium coverage across our code share.
DEFAULT_LANGUAGES = ["Python", "Go", "Rust", "Shell"]


def _fetch_blob(
    blob_id: str,
    encoding: str,
    retries: int,
    backoff: float,
    timeout: int,
    auth_token: str | None,
) -> str | None:
    """
    Fetch one file's content from SWH.

    Returns the decoded content string, or None on final failure.
    """
    url = _SWH_CONTENT_URL.format(blob_id=blob_id)
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                # Response is raw bytes; decode using the record's encoding.
                try:
                    return response.content.decode(encoding or "utf-8", errors="replace")
                except LookupError:
                    # Unknown encoding → fall back to utf-8 replace
                    return response.content.decode("utf-8", errors="replace")
            elif response.status_code == 429:
                # Rate limited — back off harder
                sleep = backoff * attempt * 2
                log.debug(f"SWH 429 for {blob_id[:12]} — sleeping {sleep:.1f}s")
                time.sleep(sleep)
            elif response.status_code == 404:
                # File gone from SWH — permanent, don't retry
                return None
            else:
                log.debug(f"SWH status {response.status_code} for {blob_id[:12]}")
                time.sleep(backoff * attempt)
        except Exception as e:
            log.debug(f"SWH fetch attempt {attempt}/{retries} for {blob_id[:12]}: {e}")
            if attempt < retries:
                time.sleep(backoff * attempt)

    return None


class StackV2Source:
    """
    Loads the-stack-v2-dedup one language at a time, fetching content from
    SWH on-demand.

    Args:
        output_dir: Directory to write output JSONL files.
        languages: Languages to include. Default: Python, Go, Rust, Shell.
            Matches the locked 4-language scope for our 1b code supply.
        min_length: Minimum file character length after fetch.
        shard_size: Files per output JSONL shard.
        max_docs: Maximum files to write. None = no limit. Used for
            mini runs to validate the pipeline.
        fetch_workers: Concurrent SWH fetch threads. Keep modest to avoid
            rate limits. Default 8.
        retries: Retry attempts per blob fetch.
        retry_backoff: Linear backoff base (seconds) for retries.
        request_timeout: Per-blob HTTP timeout (seconds).
    """

    DATASET_NAME = "bigcode/the-stack-v2-dedup"
    SOURCE_TAG = "stack_v2"

    def __init__(
        self,
        output_dir: Path,
        languages: list[str] | None = None,
        min_length: int = 50,
        shard_size: int = 10_000,
        max_docs: int | None = None,
        fetch_workers: int = 8,
        retries: int = 3,
        retry_backoff: float = 2.0,
        request_timeout: int = 30,
    ):
        self.output_dir = Path(output_dir)
        # Keep as list (not set) so iteration order is deterministic.
        self.languages = list(languages or DEFAULT_LANGUAGES)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.fetch_workers = fetch_workers
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.request_timeout = request_timeout
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optional SWH auth token for higher rate limits.
        # Register at https://archive.softwareheritage.org/ to obtain one.
        self._swh_token = os.environ.get("SWH_AUTH_TOKEN")

    def download(self) -> list[Path]:
        """Iterate per-language shards of the-stack-v2-dedup and write JSONL."""
        existing_shards = sorted(self.output_dir.glob("stack_v2_*.jsonl"))
        shard_idx = len(existing_shards)

        log.info(f"Loading {self.DATASET_NAME} per-language...")
        log.info(f"  Languages: {self.languages}")
        if not self._swh_token:
            log.warning(
                "No SWH_AUTH_TOKEN set — unauthenticated SWH fetches are "
                "aggressively rate-limited. Register at "
                "https://archive.softwareheritage.org/ and set SWH_AUTH_TOKEN "
                "in .env for higher throughput."
            )

        if self.max_docs:
            log.info(f"the-stack-v2: capped at {self.max_docs:,} files (mini run)")

        output_files: list[Path] = []
        buffer: list[dict] = []
        total_written = 0
        total_skipped_short = 0
        total_fetch_failed = 0
        total_no_blob = 0
        stop = False

        pbar = tqdm(desc="the-stack-v2", unit="file")

        # Per-language metadata batch before hitting SWH. We intentionally
        # don't fill the metadata batch to shard_size (which would be 10k
        # fetches in one blocking wave); instead we keep it modest so the
        # first shard lands soon, mini runs finish fast, and a crash
        # mid-fetch loses at most one batch.
        fetch_batch_size = min(self.shard_size, 500)

        for lang in self.languages:
            if stop:
                break

            log.info(f"  Loading language: {lang}")
            try:
                ds = load_dataset(
                    self.DATASET_NAME,
                    data_dir=f"data/{lang}",
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
            except Exception as e:
                log.exception(f"  Failed to load {lang} — skipping")
                continue

            batch: list[dict] = []

            for sample in ds:
                if stop:
                    break

                blob_id = sample.get("blob_id", "")
                if not blob_id:
                    total_no_blob += 1
                    continue

                # The language is known from the data_dir we loaded, but keep
                # the record's own field when present (useful for provenance).
                if not sample.get("language"):
                    sample = dict(sample)
                    sample["language"] = lang

                batch.append(sample)

                if len(batch) >= fetch_batch_size:
                    records, failed = self._fetch_batch(batch)
                    total_fetch_failed += failed
                    kept = [r for r in records if r is not None]
                    total_skipped_short += (len(records) - failed - len(kept))
                    buffer.extend(kept)
                    batch = []

                    while len(buffer) >= self.shard_size:
                        chunk = buffer[: self.shard_size]
                        del buffer[: self.shard_size]
                        path = self._write_shard(chunk, shard_idx)
                        output_files.append(path)
                        shard_idx += 1
                        total_written += len(chunk)
                        pbar.update(len(chunk))

                    if self.max_docs is not None:
                        if total_written + len(buffer) >= self.max_docs:
                            trim_to = max(0, self.max_docs - total_written)
                            buffer = buffer[:trim_to]
                            stop = True

            # End of this language's stream — drain any leftover batch.
            if batch and not stop:
                records, failed = self._fetch_batch(batch)
                total_fetch_failed += failed
                kept = [r for r in records if r is not None]
                total_skipped_short += (len(records) - failed - len(kept))
                buffer.extend(kept)

                while len(buffer) >= self.shard_size:
                    chunk = buffer[: self.shard_size]
                    del buffer[: self.shard_size]
                    path = self._write_shard(chunk, shard_idx)
                    output_files.append(path)
                    shard_idx += 1
                    total_written += len(chunk)
                    pbar.update(len(chunk))

                if self.max_docs is not None:
                    if total_written + len(buffer) >= self.max_docs:
                        trim_to = max(0, self.max_docs - total_written)
                        buffer = buffer[:trim_to]
                        stop = True

        # Flush any remaining buffer (mini cap usually lands here).
        if buffer:
            if self.max_docs is not None:
                overflow = (total_written + len(buffer)) - self.max_docs
                if overflow > 0:
                    buffer = buffer[: len(buffer) - overflow]
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)
            pbar.update(len(buffer))

        pbar.close()

        log.info(
            f"the-stack-v2 complete — "
            f"written: {total_written:,}, "
            f"skipped short: {total_skipped_short:,} (< {self.min_length} chars), "
            f"no blob_id: {total_no_blob:,}, "
            f"fetch failures: {total_fetch_failed:,}, "
            f"new shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _fetch_batch(self, batch: list[dict]) -> tuple[list[dict | None], int]:
        """
        Fetch content for a batch of records in parallel.

        Returns (records_or_none_list, num_failed). None entries in the
        returned list represent either fetch failures OR content below
        min_length. The caller uses `failed` to distinguish the two.
        """
        results: list[dict | None] = [None] * len(batch)
        failed = 0

        with ThreadPoolExecutor(
            max_workers=self.fetch_workers,
            thread_name_prefix="swh-fetch",
        ) as executor:
            future_to_idx = {
                executor.submit(
                    _fetch_blob,
                    sample.get("blob_id", ""),
                    sample.get("src_encoding", "utf-8"),
                    self.retries,
                    self.retry_backoff,
                    self.request_timeout,
                    self._swh_token,
                ): idx
                for idx, sample in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                sample = batch[idx]
                try:
                    content = future.result()
                except Exception as e:
                    log.debug(f"Fetch exception: {e}")
                    content = None

                if content is None:
                    failed += 1
                    continue

                content = content.strip()
                if len(content) < self.min_length:
                    # Left as None; counted as "skipped short" by the caller.
                    continue

                results[idx] = {
                    "text": content,
                    "source": self.SOURCE_TAG,
                    "language": sample.get("language", ""),
                    "repo": sample.get("repo_name", ""),
                    "path": sample.get("path", ""),
                    "blob_id": sample.get("blob_id", ""),
                }

        return results, failed

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard atomically via .tmp rename."""
        path = self.output_dir / f"stack_v2_{shard_idx:04d}.jsonl"
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
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} files → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("stack_v2_*.jsonl"))
        total_files = 0
        total_chars = 0
        lang_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_files += 1
                    total_chars += len(record.get("text", ""))
                    lang = record.get("language", "unknown")
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

        return {
            "shards": len(shards),
            "files": total_files,
            "total_chars": total_chars,
            "avg_chars_per_file": total_chars // max(total_files, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "by_language": lang_counts,
        }