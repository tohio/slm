"""
curator/sources/wikipedia.py
-----------------------------
Wikipedia English data source.

Downloads the English Wikipedia dump via HuggingFace datasets.
The HuggingFace wikipedia dataset is already clean (no wikitext markup,
no templates), so no cleaning is applied beyond whitespace normalization.

Output: JSONL with one article per line:
    {"text": "...", "source": "wikipedia", "title": "...", "url": "..."}

Usage:
    from curator.sources.wikipedia import WikipediaSource
    source = WikipediaSource(output_dir=Path("data/raw/wikipedia"))
    source.download()
"""

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class WikipediaSource:
    """
    Downloads and extracts English Wikipedia articles.

    Uses the HuggingFace wikimedia/wikipedia 20231101.en dump — a recent
    snapshot commonly used in LLM pretraining pipelines. The dataset is
    already stripped of wikitext markup, so no HTML or template cleanup
    is needed.

    Args:
        output_dir: Directory to write output JSONL files.
        min_length: Minimum article character length (articles below
            this are skipped — typically stubs / disambiguation pages).
        shard_size: Number of articles per output JSONL shard.
        max_docs: Maximum articles to write. None = no limit. Used for
            mini runs to validate the pipeline.
    """

    DATASET_NAME = "wikimedia/wikipedia"
    DATASET_CONFIG = "20231101.en"
    SOURCE_TAG = "wikipedia"

    def __init__(
        self,
        output_dir: Path,
        min_length: int = 1000,
        shard_size: int = 100_000,
        max_docs: int | None = None,
        # Accepted for backwards compatibility; the HF loader manages its own
        # parallelism so we don't currently expose it as a knob here.
        num_proc: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Download Wikipedia and write to sharded JSONL files."""
        log.info(
            f"Loading {self.DATASET_NAME}/{self.DATASET_CONFIG} from HuggingFace..."
        )
        dataset = load_dataset(
            self.DATASET_NAME,
            self.DATASET_CONFIG,
            split="train",
            trust_remote_code=True,
        )
        log.info(f"Wikipedia: {len(dataset):,} articles loaded")

        if self.max_docs:
            log.info(f"Wikipedia: capped at {self.max_docs:,} articles (mini run)")

        output_files: list[Path] = []
        shard_idx = 0
        buffer: list[dict] = []
        total_written = 0
        total_skipped = 0
        stop = False

        for article in tqdm(dataset, desc="Processing Wikipedia", unit="article"):
            text = self._clean(article["text"])
            if len(text) < self.min_length:
                total_skipped += 1
                continue

            buffer.append({
                "text": text,
                "source": self.SOURCE_TAG,
                "title": article.get("title", ""),
                "url": article.get("url", ""),
            })

            # Flush a full shard
            if len(buffer) >= self.shard_size:
                path = self._write_shard(buffer, shard_idx)
                output_files.append(path)
                shard_idx += 1
                total_written += len(buffer)
                buffer = []

            # Enforce max_docs cap — trim to exact cap, then stop.
            if self.max_docs is not None:
                if total_written + len(buffer) >= self.max_docs:
                    # Trim buffer to exactly max_docs total
                    trim_to = max(0, self.max_docs - total_written)
                    buffer = buffer[:trim_to]
                    stop = True
                    break

        # Flush remainder (or trimmed cap)
        if buffer:
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)

        log.info(
            f"Wikipedia complete — "
            f"written: {total_written:,}, "
            f"skipped: {total_skipped:,} (< {self.min_length} chars), "
            f"shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _clean(self, text: str) -> str:
        """
        Normalize whitespace on Wikipedia article text.

        The HuggingFace wikipedia dataset is already stripped of wikitext
        markup, templates, and most structural artifacts. We intentionally
        do NOT drop short lines — the previous implementation dropped any
        line <20 chars, which silently removed short paragraphs, single-
        sentence facts, and table rows. On a corpus of this quality that
        cost real tokens without any quality benefit.
        """
        return text.strip()

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard."""
        path = self.output_dir / f"wikipedia_{shard_idx:04d}.jsonl"
        with open(path, "wb") as f:
            for record in records:
                f.write(orjson.dumps(record))
                f.write(b"\n")
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} articles → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("wikipedia_*.jsonl"))
        total_articles = 0
        total_chars = 0
        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_articles += 1
                    total_chars += len(record.get("text", ""))
        return {
            "shards": len(shards),
            "articles": total_articles,
            "total_chars": total_chars,
            "avg_chars_per_article": total_chars // max(total_articles, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
        }