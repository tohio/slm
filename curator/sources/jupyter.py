"""
curator/sources/jupyter.py
---------------------------
Jupyter notebook data source.

Loads `codeparrot/github-jupyter-code-to-text` — a cleaned corpus of
GitHub Jupyter notebooks with code cells and markdown cells interleaved.
Provides training signal for code-with-explanation content that neither
raw-code sources (CodeSearchNet, stack-v2) nor pure-prose sources capture.

The source dataset already concatenates cells into a single `text` field
per notebook with cell-type markers, so we don't need to reassemble.

Tagged as source="jupyter" which is in CODE_SOURCES, so the English-prose
quality filters skip this source. The prose components of notebooks will
therefore not be language-filtered — accepted trade-off for not needing
per-cell filter dispatch.

Output: JSONL with one notebook per line:
    {
        "text": "...",
        "source": "jupyter",
        "repo": "..."
    }

Usage:
    from curator.sources.jupyter import JupyterSource
    source = JupyterSource(output_dir=Path("data/raw/jupyter"))
    source.download()
"""

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class JupyterSource:
    """
    Downloads and formats GitHub Jupyter notebooks.

    Args:
        output_dir: Directory to write output JSONL files.
        min_length: Minimum notebook character length.
        shard_size: Notebooks per output JSONL shard.
        max_docs: Maximum notebooks to write. None = no limit. Used for
            mini runs to validate the pipeline.
    """

    DATASET_NAME = "codeparrot/github-jupyter-code-to-text"
    SOURCE_TAG = "jupyter"

    def __init__(
        self,
        output_dir: Path,
        min_length: int = 500,
        shard_size: int = 10_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Download notebooks and write to sharded JSONL files."""
        log.info(f"Loading {self.DATASET_NAME} from HuggingFace...")
        dataset = load_dataset(
            self.DATASET_NAME,
            split="train",
            trust_remote_code=True,
        )
        log.info(f"Jupyter: {len(dataset):,} notebooks loaded")

        if self.max_docs:
            log.info(f"Jupyter: capped at {self.max_docs:,} notebooks (mini run)")

        output_files: list[Path] = []
        shard_idx = 0
        buffer: list[dict] = []
        total_written = 0
        total_skipped = 0
        stop = False

        for sample in tqdm(dataset, desc="Processing Jupyter", unit="nb"):
            text = (sample.get("text") or sample.get("content") or "").strip()
            if len(text) < self.min_length:
                total_skipped += 1
                continue

            buffer.append({
                "text": text,
                "source": self.SOURCE_TAG,
                "repo": sample.get("repo_name", "") or sample.get("repo", ""),
            })

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
            f"Jupyter complete — "
            f"written: {total_written:,}, "
            f"skipped: {total_skipped:,} (< {self.min_length} chars), "
            f"shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard."""
        path = self.output_dir / f"jupyter_{shard_idx:04d}.jsonl"
        with open(path, "wb") as f:
            for record in records:
                f.write(orjson.dumps(record))
                f.write(b"\n")
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} notebooks → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("jupyter_*.jsonl"))
        total_notebooks = 0
        total_chars = 0
        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_notebooks += 1
                    total_chars += len(record.get("text", ""))
        return {
            "shards": len(shards),
            "notebooks": total_notebooks,
            "total_chars": total_chars,
            "avg_chars_per_notebook": total_chars // max(total_notebooks, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
        }