"""
curator/sources/conala.py
--------------------------
CoNaLa (mined) data source.

Loads the `neulab/conala-mined` dataset — ~600k natural-language-intent
to Python-code snippet pairs mined from StackOverflow. Small source
(~20M tokens) but provides explicit NL-to-code alignment that pure code
corpora lack.

Each record is formatted with the intent as a comment line above the
code, matching the CodeSearchNet docstring-as-comment pattern:

    # <natural language intent>
    <python code snippet>

Output: JSONL with one pair per line:
    {
        "text": "# ...\\n<code>",
        "source": "conala",
        "language": "python",
        "question_id": "..."
    }

Usage:
    from curator.sources.conala import ConalaSource
    source = ConalaSource(output_dir=Path("data/raw/conala"))
    source.download()
"""

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class ConalaSource:
    """
    Downloads and formats CoNaLa-mined NL-to-code pairs.

    Args:
        output_dir: Directory to write output JSONL files.
        min_intent_length: Minimum intent (NL) character length.
        min_snippet_length: Minimum code snippet character length.
        shard_size: Pairs per output JSONL shard.
        max_docs: Maximum pairs to write. None = no limit. Used for
            mini runs to validate the pipeline.
    """

    DATASET_NAME = "neulab/conala"
    DATASET_CONFIG = "mined"
    SOURCE_TAG = "conala"

    def __init__(
        self,
        output_dir: Path,
        min_intent_length: int = 10,
        min_snippet_length: int = 10,
        shard_size: int = 50_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.min_intent_length = min_intent_length
        self.min_snippet_length = min_snippet_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Download CoNaLa and write to sharded JSONL files."""
        log.info(f"Loading {self.DATASET_NAME}/{self.DATASET_CONFIG} from HuggingFace...")
        dataset = load_dataset(
            self.DATASET_NAME,
            self.DATASET_CONFIG,
            split="train",
            trust_remote_code=True,
        )
        log.info(f"CoNaLa: {len(dataset):,} pairs loaded")

        if self.max_docs:
            log.info(f"CoNaLa: capped at {self.max_docs:,} pairs (mini run)")

        output_files: list[Path] = []
        shard_idx = 0
        buffer: list[dict] = []
        total_written = 0
        total_skipped = 0
        stop = False

        for sample in tqdm(dataset, desc="Processing CoNaLa", unit="pair"):
            record = self._format(sample)
            if record is None:
                total_skipped += 1
                continue

            buffer.append(record)

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

        log.info(
            f"CoNaLa complete — "
            f"written: {total_written:,}, "
            f"skipped: {total_skipped:,}, "
            f"shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _format(self, sample: dict) -> dict | None:
        """Format a CoNaLa sample as `# <intent>\\n<snippet>`."""
        intent = (sample.get("intent") or "").strip()
        snippet = (sample.get("snippet") or "").strip()

        if len(intent) < self.min_intent_length:
            return None
        if len(snippet) < self.min_snippet_length:
            return None

        intent_lines = "\n".join(f"# {line}" for line in intent.split("\n"))
        text = f"{intent_lines}\n{snippet}"

        return {
            "text": text,
            "source": self.SOURCE_TAG,
            "language": "python",
            "question_id": str(sample.get("question_id", "")),
        }

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard."""
        path = self.output_dir / f"conala_{shard_idx:04d}.jsonl"
        with open(path, "wb") as f:
            for record in records:
                f.write(orjson.dumps(record))
                f.write(b"\n")
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} pairs → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("conala_*.jsonl"))
        total_pairs = 0
        total_chars = 0
        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_pairs += 1
                    total_chars += len(record.get("text", ""))
        return {
            "shards": len(shards),
            "pairs": total_pairs,
            "total_chars": total_chars,
            "avg_chars_per_pair": total_chars // max(total_pairs, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
        }