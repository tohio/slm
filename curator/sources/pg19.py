"""
curator/sources/pg19.py
------------------------
Project Gutenberg 19 (pg19) data source.

Downloads ~28k public-domain books published before 1919. The train split
index is loaded from an immutable revision of the canonical Hugging Face
repository, while metadata and book text are fetched directly from DeepMind's
canonical PG-19 storage. This avoids the repository's legacy dataset loading
script, which current versions of Hugging Face Datasets no longer execute.

Books are fetched lazily in the pinned split order, so mini and character-
capped runs download only the source documents they consume.

Split: train only (validation/test are held out for downstream use).

Output: JSONL with one book per line:
    {
        "text": "...",
        "source": "pg19",
        "title": "...",
        "publication_date": "1861",
        "url": "..."
    }

Usage:
    from curator.sources.pg19 import PG19Source
    source = PG19Source(output_dir=Path("data/raw/pg19"))
    source.download()
"""

import csv
import logging
import time
from pathlib import Path

import orjson
import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from config import CHARS_PER_TOKEN
from curator.sources.hf import resolve_dataset_revision

log = logging.getLogger(__name__)

# Raw character buffer for PG-19. The blend stage consumes only the target
# character budget, so the source only needs enough raw text to survive
# filtering/dedup with headroom.
PG19_CHAR_OVERFETCH_FACTOR = 1.30
PG19_ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg"
PG19_METADATA_URL = f"{PG19_ASSET_ROOT_URL}/metadata.csv"
PG19_TRAIN_FILES = "data/train_files.txt"
PG19_DOWNLOAD_RETRIES = 5
PG19_BACKOFF_MAX_SECONDS = 30


class PG19Source:
    """
    Downloads and extracts PG-19 public-domain books from canonical storage.

    pg19 books are long (mean ~100k tokens each), so shard_size is
    kept small to keep individual JSONL files manageable.

    Args:
        output_dir: Directory to write output JSONL files.
        min_length: Minimum book character length. Books shorter than this
            are skipped (rare — most pg19 books are 100k+ chars).
        shard_size: Books per output JSONL shard.
        max_docs: Maximum books to write. None = no limit. Used for
            mini runs to validate the pipeline.
    """

    # The repository supplies the canonical split membership. Its legacy
    # loading script is intentionally not executed.
    DATASET_NAME = "deepmind/pg19"
    SOURCE_TAG = "pg19"

    def __init__(
        self,
        output_dir: Path,
        min_length: int = 10_000,
        shard_size: int = 250,
        max_docs: int | None = None,
        max_chars: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.max_chars = max_chars
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Stream pg19 and write to sharded JSONL files."""
        existing = sorted(self.output_dir.glob("pg19_*.jsonl"))
        if existing:
            raise RuntimeError(
                "pg19 output directory is not empty. Use the canonical "
                "curator's manifest-aware restart/replacement flow."
            )

        revision = resolve_dataset_revision(self.DATASET_NAME)
        split_path = hf_hub_download(
            repo_id=self.DATASET_NAME,
            repo_type="dataset",
            filename=PG19_TRAIN_FILES,
            revision=revision,
        )
        train_files = sorted(
            line.strip()
            for line in Path(split_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

        log.info(
            "Streaming %s canonical assets using pinned revision %s...",
            self.DATASET_NAME,
            revision,
        )

        if self.max_docs:
            log.info(f"pg19: capped at {self.max_docs:,} books")
        if self.max_chars:
            log.info(f"pg19: capped at {self.max_chars:,} raw chars")

        output_files: list[Path] = []
        shard_idx = 0
        buffer: list[dict] = []
        total_written = 0
        total_chars_written = 0
        total_skipped = 0
        stop = False

        with requests.Session() as session:
            metadata = self._load_metadata(session)
            pbar = tqdm(train_files, desc="Processing pg19", unit="book")

            for relative_path in pbar:
                book_id = Path(relative_path).stem
                book_metadata = metadata.get(book_id)
                if book_metadata is None:
                    raise RuntimeError(
                        f"PG-19 metadata is missing book id {book_id}"
                    )

                text = self._download_text(
                    session,
                    f"{PG19_ASSET_ROOT_URL}/{relative_path}",
                ).strip()
                if len(text) < self.min_length:
                    total_skipped += 1
                    continue

                buffer.append({
                    "text": text,
                    "source": self.SOURCE_TAG,
                    "title": book_metadata.get("short_book_title", ""),
                    "publication_date": str(
                        book_metadata.get("publication_date", "")
                    ),
                    "url": book_metadata.get("url", ""),
                })

                if len(buffer) >= self.shard_size:
                    path = self._write_shard(buffer, shard_idx)
                    output_files.append(path)
                    shard_idx += 1
                    total_written += len(buffer)
                    total_chars_written += sum(
                        len(record.get("text", "")) for record in buffer
                    )
                    buffer = []

                buffered_chars = sum(
                    len(record.get("text", "")) for record in buffer
                )

                if (
                    self.max_docs is not None
                    and total_written + len(buffer) >= self.max_docs
                ):
                    trim_to = max(0, self.max_docs - total_written)
                    buffer = buffer[:trim_to]
                    stop = True
                    break

                if (
                    self.max_chars is not None
                    and total_chars_written + buffered_chars >= self.max_chars
                ):
                    stop = True
                    break

        if buffer:
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)
            total_chars_written += sum(len(r.get("text", "")) for r in buffer)

        log.info(
            f"pg19 complete — "
            f"written: {total_written:,}, "
            f"chars: {total_chars_written:,}, "
            f"skipped: {total_skipped:,} (< {self.min_length} chars), "
            f"shards: {len(output_files)}"
            f"{' (stopped at cap)' if stop else ''}"
        )
        return output_files

    def _load_metadata(self, session: requests.Session) -> dict[str, dict[str, str]]:
        """Load canonical PG-19 book metadata keyed by Gutenberg id."""
        metadata_text = self._download_text(session, PG19_METADATA_URL)
        fields = ["_id", "short_book_title", "publication_date", "url"]
        return {
            row["_id"]: row
            for row in csv.DictReader(metadata_text.splitlines(), fieldnames=fields)
        }

    @staticmethod
    def _download_text(session: requests.Session, url: str) -> str:
        """Fetch one UTF-8 PG-19 asset with bounded retries."""
        for attempt in range(1, PG19_DOWNLOAD_RETRIES + 1):
            try:
                response = session.get(url, timeout=(10, 120))
                response.raise_for_status()
                return response.content.decode("utf-8")
            except (requests.RequestException, UnicodeDecodeError) as exc:
                if attempt == PG19_DOWNLOAD_RETRIES:
                    raise RuntimeError(
                        f"Failed to download PG-19 asset after {attempt} attempts: "
                        f"{url}"
                    ) from exc
                delay = min(PG19_BACKOFF_MAX_SECONDS, 2 ** (attempt - 1))
                log.warning(
                    "PG-19 asset download failed (attempt %d/%d); retrying "
                    "in %ds: %s",
                    attempt,
                    PG19_DOWNLOAD_RETRIES,
                    delay,
                    url,
                )
                time.sleep(delay)

        raise AssertionError("unreachable")

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard."""
        path = self.output_dir / f"pg19_{shard_idx:04d}.jsonl"
        with open(path, "wb") as f:
            for record in records:
                f.write(orjson.dumps(record))
                f.write(b"\n")
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} books → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("pg19_*.jsonl"))
        total_books = 0
        total_chars = 0
        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_books += 1
                    total_chars += len(record.get("text", ""))
        return {
            "shards": len(shards),
            "books": total_books,
            "total_chars": total_chars,
            "avg_chars_per_book": total_chars // max(total_books, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "max_docs": self.max_docs,
            "max_chars": self.max_chars,
            "char_overfetch_factor": PG19_CHAR_OVERFETCH_FACTOR,
        }
