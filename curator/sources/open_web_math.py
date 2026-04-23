"""
curator/sources/open_web_math.py
---------------------------------
open-web-math data source.

Streams the `open-web-math/open-web-math` dataset — ~14.7B tokens of
mathematical web content filtered from Common Crawl. Covers LaTeX-heavy
pages, math forums, lecture notes, textbook excerpts, and problem sets.

Math content is underrepresented in generic web crawls (FineWeb filters
out heavy-LaTeX pages as "low quality" by its heuristics), so this source
adds coverage that would otherwise be missing. At 10% of the mix it
provides meaningful mathematical reasoning capability without dominating
the distribution.

Streams in full rather than loading all at once. Resume works at shard
granularity.

Output: JSONL with one document per line:
    {
        "text": "...",
        "source": "open_web_math",
        "url": "...",
        "date": "...",
        "subdomain": "..."
    }

Usage:
    from curator.sources.open_web_math import OpenWebMathSource
    source = OpenWebMathSource(output_dir=Path("data/raw/open_web_math"))
    source.download()
"""

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class OpenWebMathSource:
    """
    Streams open-web-math and writes sharded JSONL.

    Args:
        output_dir: Directory to write output JSONL files.
        min_length: Minimum document character length. Below this, skipped.
        shard_size: Documents per output JSONL shard.
        max_docs: Maximum documents to write. None = no limit. Used for
            mini runs to validate the pipeline.
    """

    DATASET_NAME = "open-web-math/open-web-math"
    SOURCE_TAG = "open_web_math"

    def __init__(
        self,
        output_dir: Path,
        min_length: int = 200,
        shard_size: int = 50_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Stream open-web-math and write to sharded JSONL files."""
        existing_shards = sorted(self.output_dir.glob("open_web_math_*.jsonl"))
        shard_idx = len(existing_shards)
        skip_records = shard_idx * self.shard_size

        if skip_records > 0:
            log.info(
                f"open-web-math: found {shard_idx} existing shard(s) — "
                f"skipping first {skip_records:,} streamed records"
            )

        log.info(f"Streaming {self.DATASET_NAME} from HuggingFace...")
        stream = load_dataset(
            self.DATASET_NAME,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        if self.max_docs:
            log.info(
                f"open-web-math: capped at {self.max_docs:,} documents (mini run)"
            )

        output_files: list[Path] = []
        buffer: list[dict] = []
        total_written = 0
        total_skipped_short = 0
        total_stream_skipped = 0
        stop = False

        pbar = tqdm(desc="Streaming open-web-math", unit="doc")

        for idx, sample in enumerate(stream):
            # Resume: skip records belonging to already-written shards
            if idx < skip_records:
                total_stream_skipped += 1
                if total_stream_skipped % 100_000 == 0:
                    pbar.set_postfix_str(
                        f"skipping {total_stream_skipped:,}/{skip_records:,}"
                    )
                continue

            text = (sample.get("text") or "").strip()
            if len(text) < self.min_length:
                total_skipped_short += 1
                continue

            # Metadata is a JSON-encoded string; parse it defensively.
            metadata = sample.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = orjson.loads(metadata)
                except Exception:
                    metadata = {}
            elif not isinstance(metadata, dict):
                metadata = {}

            buffer.append({
                "text": text,
                "source": self.SOURCE_TAG,
                "url": sample.get("url", ""),
                "date": str(sample.get("date", "")),
                "subdomain": metadata.get("subdomain", ""),
            })

            if len(buffer) >= self.shard_size:
                path = self._write_shard(buffer, shard_idx)
                output_files.append(path)
                shard_idx += 1
                total_written += len(buffer)
                buffer = []
                pbar.update(self.shard_size)

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
            f"open-web-math complete — "
            f"written: {total_written:,}, "
            f"skipped short: {total_skipped_short:,} (< {self.min_length} chars), "
            f"stream-skipped (resume): {total_stream_skipped:,}, "
            f"new shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard atomically via .tmp rename."""
        path = self.output_dir / f"open_web_math_{shard_idx:04d}.jsonl"
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

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("open_web_math_*.jsonl"))
        total_docs = 0
        total_chars = 0
        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_docs += 1
                    total_chars += len(record.get("text", ""))
        return {
            "shards": len(shards),
            "documents": total_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": total_chars // max(total_docs, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
        }