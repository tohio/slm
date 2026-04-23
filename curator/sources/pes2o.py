"""
curator/sources/pes2o.py
-------------------------
peS2o academic papers data source.

Streams the `allenai/peS2o` dataset — a cleaned, filtered, and deduplicated
subset of Semantic Scholar's open research corpus (S2ORC). Covers ~42B
tokens across scientific papers from many fields: CS, biology, physics,
medicine, social sciences, and more.

Provides academic prose and technical writing that's underrepresented in
FineWeb/Wikipedia. Helpful for reasoning, formal writing style, and
vocabulary in technical domains.

Streams in full rather than loading all at once — at 42B tokens the corpus
is too large to materialize. Resume works at shard granularity: existing
shards on disk mean that many records of the stream are skipped before
writing resumes.

Schema note: peS2o splits documents into s2orc (full-text papers) and
s2ag (abstracts + titles). We consume both — s2ag provides broad topical
coverage, s2orc provides long-form reasoning chains. Filtering on
document length happens at the shared filter step.

Output: JSONL with one document per line:
    {
        "text": "...",
        "source": "pes2o",
        "paper_id": "...",
        "subset": "s2orc | s2ag"
    }

Usage:
    from curator.sources.pes2o import PeS2oSource
    source = PeS2oSource(output_dir=Path("data/raw/pes2o"))
    source.download()
"""

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class PeS2oSource:
    """
    Streams peS2o and writes sharded JSONL.

    Args:
        output_dir: Directory to write output JSONL files.
        config: HF dataset config. Default `v2` — the latest published
            version at time of writing. Specific versions (`v1`) are
            also available.
        subsets: Which peS2o subsets to include. Default both s2orc
            (full-text papers) and s2ag (abstracts + titles).
        min_length: Minimum document character length. Below this, skipped.
        shard_size: Documents per output JSONL shard.
        max_docs: Maximum documents to write. None = no limit. Used for
            mini runs to validate the pipeline.
    """

    DATASET_NAME = "allenai/peS2o"
    DATASET_CONFIG = "v2"
    SOURCE_TAG = "pes2o"
    DEFAULT_SUBSETS = ("s2orc", "s2ag")

    def __init__(
        self,
        output_dir: Path,
        config: str | None = None,
        subsets: tuple[str, ...] | list[str] = DEFAULT_SUBSETS,
        min_length: int = 500,
        shard_size: int = 50_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or self.DATASET_CONFIG
        self.subsets = list(subsets)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Stream peS2o and write to sharded JSONL files."""
        existing_shards = sorted(self.output_dir.glob("pes2o_*.jsonl"))
        shard_idx = len(existing_shards)
        skip_records = shard_idx * self.shard_size

        if skip_records > 0:
            log.info(
                f"peS2o: found {shard_idx} existing shard(s) — "
                f"skipping first {skip_records:,} streamed records"
            )

        log.info(
            f"Streaming {self.DATASET_NAME}/{self.config} from HuggingFace..."
        )
        stream = load_dataset(
            self.DATASET_NAME,
            self.config,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        if self.max_docs:
            log.info(f"peS2o: capped at {self.max_docs:,} documents (mini run)")

        output_files: list[Path] = []
        buffer: list[dict] = []
        total_written = 0
        total_skipped_short = 0
        total_skipped_subset = 0
        total_stream_skipped = 0
        stop = False

        pbar = tqdm(desc="Streaming peS2o", unit="doc")

        for idx, sample in enumerate(stream):
            # Resume: skip records belonging to already-written shards
            if idx < skip_records:
                total_stream_skipped += 1
                if total_stream_skipped % 100_000 == 0:
                    pbar.set_postfix_str(
                        f"skipping {total_stream_skipped:,}/{skip_records:,}"
                    )
                continue

            subset = sample.get("source", "")  # peS2o's own source tag: s2orc or s2ag
            if subset not in self.subsets:
                total_skipped_subset += 1
                continue

            text = (sample.get("text") or "").strip()
            if len(text) < self.min_length:
                total_skipped_short += 1
                continue

            buffer.append({
                "text": text,
                "source": self.SOURCE_TAG,
                "paper_id": str(sample.get("id", "")),
                "subset": subset,
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
            f"peS2o complete — "
            f"written: {total_written:,}, "
            f"skipped short: {total_skipped_short:,} (< {self.min_length} chars), "
            f"skipped subset: {total_skipped_subset:,} (not in {self.subsets}), "
            f"stream-skipped (resume): {total_stream_skipped:,}, "
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