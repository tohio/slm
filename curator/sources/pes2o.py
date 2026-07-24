"""
curator/sources/pes2o.py
-------------------------
peS2o academic papers data source.

Streams `common-pile/peS2o_filtered`, the Common Pile release of peS2o
restricted to openly licensed full-text papers and distributed as ordinary
data files. Unlike the legacy `allenai/peS2o` loader, it does not require a
dataset script or `trust_remote_code`.

Provides academic prose and technical writing that's underrepresented in
FineWeb/Wikipedia. Helpful for reasoning, formal writing style, and
vocabulary in technical domains.

The source is streamed rather than materialized. Stage-level completion and
restart safety are owned by `curator/scripts/curate.py`; a source invocation
always starts from an empty staging directory.

Schema note: Common Pile tags records as `pes2o/s2orc`. The release contains
full-text S2ORC documents; the old abstract-only `s2ag` subset is not assumed.

Output: JSONL with one document per line:
    {
        "text": "...",
        "source": "pes2o",
        "paper_id": "...",
        "subset": "s2orc"
    }

Usage:
    from curator.sources.pes2o import PeS2oSource
    source = PeS2oSource(output_dir=Path("data/raw/pes2o"))
    source.download()
"""

import logging
from pathlib import Path

import orjson
from curator.sources.hf import load_dataset
from tqdm import tqdm

from config import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class PeS2oSource:
    """
    Streams peS2o and writes sharded JSONL.

    Args:
        output_dir: Directory to write output JSONL files.
        subsets: Which peS2o subsets to include. The Common Pile release uses
            the `s2orc` full-text subset.
        min_length: Minimum document character length. Below this, skipped.
        shard_size: Documents per output JSONL shard.
        max_docs: Maximum documents to write. None = no limit. Used for
            mini runs to validate the pipeline.
    """

    DATASET_NAME = "common-pile/peS2o_filtered"
    SOURCE_TAG = "pes2o"
    DEFAULT_SUBSETS = ("s2orc",)

    def __init__(
        self,
        output_dir: Path,
        subsets: tuple[str, ...] | list[str] = DEFAULT_SUBSETS,
        min_length: int = 500,
        shard_size: int = 50_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.subsets = list(subsets)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Stream peS2o and write to sharded JSONL files."""
        existing_shards = sorted(self.output_dir.glob("pes2o_*.jsonl"))
        if existing_shards:
            raise RuntimeError(
                "peS2o output directory is not empty. The canonical curator "
                "owns restart/replacement semantics; shard count alone is not "
                "a safe streaming checkpoint."
            )
        shard_idx = 0

        log.info(f"Streaming {self.DATASET_NAME} from HuggingFace...")
        stream = load_dataset(
            self.DATASET_NAME,
            split="train",
            streaming=True,
        )

        if self.max_docs:
            log.info(f"peS2o: capped at {self.max_docs:,} documents")

        output_files: list[Path] = []
        buffer: list[dict] = []
        total_written = 0
        total_skipped_short = 0
        total_skipped_subset = 0
        stop = False

        pbar = tqdm(desc="Streaming peS2o", unit="doc")

        for sample in stream:
            # Common Pile tags the source as "pes2o/s2orc".
            raw_subset = sample.get("source", "")
            parts = raw_subset.split("/", 1)
            subset = parts[1] if len(parts) == 2 and parts[0] == "pes2o" else parts[0]
            if subset not in self.subsets:
                total_skipped_subset += 1
                continue

            text = (sample.get("text") or "").strip()
            if len(text) < self.min_length:
                total_skipped_short += 1
                continue

            metadata = sample.get("metadata")
            record = {
                "text": text,
                "source": self.SOURCE_TAG,
                "paper_id": str(sample.get("id", "")),
                "subset": subset,
            }
            if isinstance(metadata, dict):
                record["metadata"] = metadata
            buffer.append(record)

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
            f"peS2o complete — "
            f"written: {total_written:,}, "
            f"skipped short: {total_skipped_short:,} (< {self.min_length} chars), "
            f"skipped subset: {total_skipped_subset:,} (not in {self.subsets}), "
            f"new shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard atomically via .tmp rename."""
        path = self.output_dir / f"pes2o_{shard_idx:04d}.jsonl"
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
        shards = sorted(self.output_dir.glob("pes2o_*.jsonl"))
        total_docs = 0
        total_chars = 0
        subset_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_docs += 1
                    total_chars += len(record.get("text", ""))
                    subset = record.get("subset", "unknown")
                    subset_counts[subset] = subset_counts.get(subset, 0) + 1

        return {
            "shards": len(shards),
            "documents": total_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": total_chars // max(total_docs, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "by_subset": subset_counts,
        }
