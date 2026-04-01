"""
curator/sources/code_search_net.py
------------------------------------
CodeSearchNet data source.

Downloads the CodeSearchNet dataset via HuggingFace datasets.
CodeSearchNet contains ~2M code functions with docstrings across
6 programming languages: Python, Java, JavaScript, PHP, Ruby, Go.

We use the code + docstring pairs to give the model exposure to
both natural language descriptions and implementation — better for
instruction following on coding tasks than raw code alone.

Output: JSONL with one function per line:
    {
        "text": "...",           # formatted code + docstring
        "source": "code",
        "language": "python",
        "repo": "...",
        "path": "..."
    }

Target contribution: ~300M tokens (~10% of 3B token 125M training mix).

Usage:
    from curator.sources.code_search_net import CodeSearchNetSource
    source = CodeSearchNetSource(output_dir=Path("data/raw/code"))
    source.download()
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

log = logging.getLogger(__name__)

# Languages to include — all 6 available in CodeSearchNet
LANGUAGES = ["python", "java", "javascript", "php", "ruby", "go"]

# Language → file extension mapping for formatting
EXTENSIONS = {
    "python": "py",
    "java": "java",
    "javascript": "js",
    "php": "php",
    "ruby": "rb",
    "go": "go",
}


class CodeSearchNetSource:
    """
    Downloads and formats CodeSearchNet for LLM pretraining.

    Formats each sample as:
        # <docstring>
        <code>

    This gives the model a natural docstring → code association
    that transfers well to code generation tasks during SFT.

    Args:
        output_dir: Directory to write output JSONL files.
        languages: List of programming languages to include.
        min_code_length: Minimum code character length. Default: 100.
        min_docstring_length: Minimum docstring character length. Default: 20.
        splits: Dataset splits to use. Default: all (train + valid + test).
        shard_size: Number of samples per output JSONL shard.
        max_docs: Maximum number of samples to write. None = no limit.
            Used for mini runs to validate the pipeline without downloading
            the full dataset.
    """

    DATASET_NAME = "code_search_net"
    SOURCE_TAG = "code"

    def __init__(
        self,
        output_dir: Path,
        languages: list[str] = LANGUAGES,
        min_code_length: int = 100,
        min_docstring_length: int = 20,
        splits: list[str] = ["train", "validation", "test"],
        shard_size: int = 100_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.languages = languages
        self.min_code_length = min_code_length
        self.min_docstring_length = min_docstring_length
        self.splits = splits
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """
        Download CodeSearchNet and write to sharded JSONL files.

        Returns:
            List of paths to output JSONL files.
        """
        all_datasets = []

        for language in self.languages:
            log.info(f"Loading CodeSearchNet — {language}...")
            try:
                lang_datasets = []
                for split in self.splits:
                    ds = load_dataset(
                        self.DATASET_NAME,
                        language,
                        split=split,
                        trust_remote_code=True,
                    )
                    lang_datasets.append(ds)
                combined = concatenate_datasets(lang_datasets)
                log.info(f"  {language}: {len(combined):,} samples")
                all_datasets.append(combined)
            except Exception as e:
                log.warning(f"  Failed to load {language}: {e} — skipping")

        if not all_datasets:
            raise RuntimeError("No CodeSearchNet languages loaded successfully")

        dataset = concatenate_datasets(all_datasets)
        log.info(f"CodeSearchNet total: {len(dataset):,} samples across {len(self.languages)} languages")

        if self.max_docs:
            log.info(f"CodeSearchNet: capped at {self.max_docs:,} samples (mini run)")

        output_files = []
        shard_idx = 0
        buffer = []
        total_written = 0
        total_skipped = 0

        for sample in tqdm(dataset, desc="Processing CodeSearchNet", unit="sample"):
            record = self._format(sample)
            if record is None:
                total_skipped += 1
                continue

            buffer.append(record)

            if self.max_docs and total_written + len(buffer) >= self.max_docs:
                break

            if len(buffer) >= self.shard_size:
                path = self._write_shard(buffer, shard_idx)
                output_files.append(path)
                shard_idx += 1
                total_written += len(buffer)
                buffer = []

        if buffer:
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)

        log.info(
            f"CodeSearchNet complete — "
            f"written: {total_written:,}, "
            f"skipped: {total_skipped:,}, "
            f"shards: {len(output_files)}"
        )
        return output_files

    def _format(self, sample: dict) -> dict | None:
        """
        Format a CodeSearchNet sample into a pretraining record.

        Combines docstring and code into a natural format:
            # <docstring>
            <code>

        Returns None if the sample doesn't meet quality thresholds.
        """
        code = (sample.get("whole_func_string") or sample.get("code") or "").strip()
        docstring = (sample.get("func_documentation_string") or sample.get("docstring") or "").strip()
        language = sample.get("language", "unknown")
        ext = EXTENSIONS.get(language, language)

        if len(code) < self.min_code_length:
            return None
        if len(docstring) < self.min_docstring_length:
            return None

        # Format as a natural docstring + code block
        # The comment style varies by language but # works for most
        comment_char = "#" if language in ["python", "ruby"] else "//"
        docstring_lines = "\n".join(
            f"{comment_char} {line}" for line in docstring.strip().split("\n")
        )
        text = f"{docstring_lines}\n{code}"

        return {
            "text": text,
            "source": self.SOURCE_TAG,
            "language": language,
            "repo": sample.get("repository_name", ""),
            "path": sample.get("func_path_in_repository", ""),
        }

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write a list of records to a JSONL shard file."""
        path = self.output_dir / f"code_{shard_idx:04d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} samples → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("code_*.jsonl"))
        total_samples = 0
        total_chars = 0
        lang_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard) as f:
                for line in f:
                    record = json.loads(line)
                    total_samples += 1
                    total_chars += len(record["text"])
                    lang = record.get("language", "unknown")
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

        return {
            "shards": len(shards),
            "samples": total_samples,
            "total_chars": total_chars,
            "avg_chars_per_sample": total_chars // max(total_samples, 1),
            "estimated_tokens": total_chars // 4,
            "by_language": lang_counts,
        }