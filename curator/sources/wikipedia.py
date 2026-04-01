"""
curator/sources/wikipedia.py
-----------------------------
Wikipedia English data source.

Downloads the English Wikipedia dump via HuggingFace datasets and
extracts clean article text. Wikipedia is one of the highest quality
text sources available — factual, well-structured, and broad in coverage.

Output: JSONL with one article per line:
    {"text": "...", "source": "wikipedia", "title": "...", "url": "..."}

Target contribution: ~600M tokens (~20% of 3B token 125M training mix).

Usage:
    from curator.sources.wikipedia import WikipediaSource
    source = WikipediaSource(output_dir=Path("data/raw/wikipedia"))
    source.download()
"""

import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


class WikipediaSource:
    """
    Downloads and extracts English Wikipedia articles.

    Uses the HuggingFace datasets wikipedia 20220301.en dump —
    the standard snapshot used by most LLM pretraining pipelines.

    Args:
        output_dir: Directory to write output JSONL files.
        min_length: Minimum article character length. Articles shorter
            than this are skipped — typically stubs or disambiguation pages.
        num_proc: Number of processes for parallel processing.
        shard_size: Number of articles per output JSONL shard.
        max_docs: Maximum number of articles to write. None = no limit.
            Used for mini runs to validate the pipeline without downloading
            the full dataset.
    """

    DATASET_NAME = "wikipedia"
    DATASET_CONFIG = "20220301.en"
    SOURCE_TAG = "wikipedia"

    def __init__(
        self,
        output_dir: Path,
        min_length: int = 1000,
        num_proc: int = 4,
        shard_size: int = 100_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.num_proc = num_proc
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """
        Download Wikipedia and write to sharded JSONL files.

        Returns:
            List of paths to output JSONL files.
        """
        log.info(f"Loading {self.DATASET_NAME}/{self.DATASET_CONFIG} from HuggingFace...")
        dataset = load_dataset(
            self.DATASET_NAME,
            self.DATASET_CONFIG,
            split="train",
            trust_remote_code=True,
        )
        log.info(f"Wikipedia: {len(dataset):,} articles loaded")

        if self.max_docs:
            log.info(f"Wikipedia: capped at {self.max_docs:,} articles (mini run)")

        output_files = []
        shard_idx = 0
        buffer = []
        total_written = 0
        total_skipped = 0

        for article in tqdm(dataset, desc="Processing Wikipedia", unit="article"):
            text = self._clean(article["text"])

            if len(text) < self.min_length:
                total_skipped += 1
                continue

            record = {
                "text": text,
                "source": self.SOURCE_TAG,
                "title": article.get("title", ""),
                "url": article.get("url", ""),
            }
            buffer.append(record)

            if self.max_docs and total_written + len(buffer) >= self.max_docs:
                break

            if len(buffer) >= self.shard_size:
                path = self._write_shard(buffer, shard_idx)
                output_files.append(path)
                shard_idx += 1
                total_written += len(buffer)
                buffer = []

        # Write remaining
        if buffer:
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)

        log.info(
            f"Wikipedia complete — "
            f"written: {total_written:,}, "
            f"skipped: {total_skipped:,} (< {self.min_length} chars), "
            f"shards: {len(output_files)}"
        )
        return output_files

    def _clean(self, text: str) -> str:
        """
        Light cleaning of Wikipedia article text.

        The HuggingFace Wikipedia dataset already strips most markup.
        We just normalize whitespace and remove very short lines
        (typically section headers with no content).
        """
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            # Skip very short lines — usually section headers or empty
            if len(line) < 20:
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write a list of records to a JSONL shard file."""
        import json
        path = self.output_dir / f"wikipedia_{shard_idx:04d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} articles → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        import json
        shards = sorted(self.output_dir.glob("wikipedia_*.jsonl"))
        total_articles = 0
        total_chars = 0
        for shard in shards:
            with open(shard) as f:
                for line in f:
                    record = json.loads(line)
                    total_articles += 1
                    total_chars += len(record["text"])
        return {
            "shards": len(shards),
            "articles": total_articles,
            "total_chars": total_chars,
            "avg_chars_per_article": total_chars // max(total_articles, 1),
            "estimated_tokens": total_chars // 4,
        }