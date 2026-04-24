"""
curator/sources/pg19.py
------------------------
Project Gutenberg 19 (pg19) data source.

Downloads the pg19 dataset via HuggingFace datasets — ~28k public-domain
books published before 1919, ~2.9B tokens total. Provides long-form
coherent prose that complements the web-heavy sources (FineWeb, CC) and
the short-form reference sources (Wikipedia).

We use streaming mode instead of materializing the full dataset. pg19
on HF stores each book as an individual parquet file (~30k files total),
so a non-streaming load_dataset call triggers ~30k sequential HTTP
requests just to populate the cache — pathologically slow even at
125m/350m/1b scale, and utterly wasted for mini (50 books). Streaming
pulls parquet files lazily as iteration progresses, so the number of
downloads is proportional to how many books we actually read, not
the full corpus size.

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

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class PG19Source:
    """
    Downloads and extracts pg19 public-domain books via HF streaming.

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

    # Canonical namespaced name. The bare "pg19" alias redirects to an
    # old loading-script path that 404s on dataset_infos.json on
    # datasets>=3.x, causing silent load failures.
    DATASET_NAME = "deepmind/pg19"
    SOURCE_TAG = "pg19"

    def __init__(
        self,
        output_dir: Path,
        min_length: int = 10_000,
        shard_size: int = 1_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Stream pg19 and write to sharded JSONL files."""
        log.info(f"Streaming {self.DATASET_NAME} from HuggingFace...")
        dataset = load_dataset(
            self.DATASET_NAME,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        if self.max_docs:
            log.info(f"pg19: capped at {self.max_docs:,} books (mini run)")

        output_files: list[Path] = []
        shard_idx = 0
        buffer: list[dict] = []
        total_written = 0
        total_skipped = 0
        stop = False

        # Streaming dataset has no len(); use tqdm without a total.
        pbar = tqdm(desc="Processing pg19", unit="book")

        for book in dataset:
            text = (book.get("text") or "").strip()
            if len(text) < self.min_length:
                total_skipped += 1
                pbar.update(1)
                continue

            buffer.append({
                "text": text,
                "source": self.SOURCE_TAG,
                "title": book.get("short_book_title", ""),
                "publication_date": str(book.get("publication_date", "")),
                "url": book.get("url", ""),
            })
            pbar.update(1)

            if len(buffer) >= self.shard_size:
                path = self._write_shard(buffer, shard_idx)
                output_files.append(path)
                shard_idx += 1
                total_written += len(buffer)
                buffer = []

            if self.max_docs is not None:
                if total_written + len(buffer) >= self.max_docs:
                    trim_to = max(0, self.max_docs - total_written)
                    buffer = buffer[:trim_to]
                    stop = True
                    break

        if buffer:
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)

        pbar.close()

        log.info(
            f"pg19 complete — "
            f"written: {total_written:,}, "
            f"skipped: {total_skipped:,} (< {self.min_length} chars), "
            f"shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

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
        }