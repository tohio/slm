"""
FineWeb-Edu data source.

Streams the HuggingFace FineWeb-Edu corpus (HuggingFaceFW/fineweb-edu).
This is kept as a separate source implementation instead of refactoring
FineWebSource so the educational-web bucket can be removed or changed without
touching the existing FineWeb path.

Output: JSONL with one document per line:
    {
        "text": "...",
        "source": "fineweb_edu",
        "url": "...",
        "dump": "...",
        "language": "en"
    }
"""

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class FineWebEduSource:
    """
    Streams FineWeb-Edu and writes sharded JSONL.

    Args:
        output_dir: Directory to write output JSONL files.
        config: HF dataset config name. Default is `sample-10BT`.
        min_length: Minimum document character length. Shorter docs skipped.
        shard_size: Documents per output JSONL shard.
        max_docs: Maximum documents to write. None = no limit.
    """

    DATASET_NAME = "HuggingFaceFW/fineweb-edu"
    DATASET_CONFIG = "sample-10BT"
    SOURCE_TAG = "fineweb_edu"
    SHARD_PREFIX = "fineweb_edu"

    def __init__(
        self,
        output_dir: Path,
        config: str | None = None,
        min_length: int = 200,
        shard_size: int = 100_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or self.DATASET_CONFIG
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Stream FineWeb-Edu and write to sharded JSONL files."""
        existing_shards = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        shard_idx = len(existing_shards)
        skip_records = shard_idx * self.shard_size

        if skip_records > 0:
            log.info(
                f"FineWeb-Edu: found {shard_idx} existing shard(s) — "
                f"skipping first {skip_records:,} streamed records"
            )

        log.info(f"Streaming {self.DATASET_NAME}/{self.config} from HuggingFace...")
        stream = load_dataset(
            self.DATASET_NAME,
            self.config,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        if self.max_docs:
            log.info(f"FineWeb-Edu: capped at {self.max_docs:,} documents")

        output_files: list[Path] = []
        buffer: list[dict] = []
        total_written = 0
        total_skipped_short = 0
        total_stream_skipped = 0
        stop = False

        pbar = tqdm(desc="Streaming FineWeb-Edu", unit="doc")

        for idx, sample in enumerate(stream):
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

            buffer.append({
                "text": text,
                "source": self.SOURCE_TAG,
                "url": sample.get("url", ""),
                "dump": sample.get("dump", ""),
                "language": sample.get("language", "en"),
            })

            if len(buffer) >= self.shard_size:
                path = self._write_shard(buffer, shard_idx)
                output_files.append(path)
                shard_idx += 1
                total_written += len(buffer)
                buffer = []
                pbar.update(self.shard_size)

            if self.max_docs is not None and total_written + len(buffer) >= self.max_docs:
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
            f"FineWeb-Edu complete — "
            f"written: {total_written:,}, "
            f"skipped short: {total_skipped_short:,} (< {self.min_length} chars), "
            f"stream-skipped (resume): {total_stream_skipped:,}, "
            f"new shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard atomically via .tmp rename."""
        path = self.output_dir / f"{self.SHARD_PREFIX}_{shard_idx:04d}.jsonl"
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
        shards = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        total_docs = 0
        total_chars = 0
        dump_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_docs += 1
                    total_chars += len(record.get("text", ""))
                    dump = record.get("dump", "unknown")
                    dump_counts[dump] = dump_counts.get(dump, 0) + 1

        return {
            "shards": len(shards),
            "documents": total_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": total_chars // max(total_docs, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "by_dump": dump_counts,
        }
