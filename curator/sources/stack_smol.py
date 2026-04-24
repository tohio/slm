"""
curator/sources/stack_smol.py
----------------------------------
the-stack-smol data source.

Loads `bigcode/the-stack-smol` — a 1000-files-per-language sample of
BigCode's The Stack, covering 30 programming languages. Each record is
a full source file with its content inline (no external fetch needed,
unlike the-stack-v2).

File layout note:
    the-stack-smol stores each language as a single `data/<lang>/data.json`
    file (not parquet, not sharded). We load each language via the
    data_files pattern rather than data_dir or get_dataset_config_names,
    both of which fail on this repo: get_dataset_config_names returns an
    empty list because there's no loading script or dataset_infos.json,
    and data_dir auto-detection can't resolve the single-JSON layout.

No language filtering — we take all 30 languages. This gives the model
broad syntactic exposure across the language ecosystem, including
languages (Rust, Shell, TypeScript, Scala, Haskell, etc.) that have no
curated coverage in CodeSearchNet. Majority of the tokens come from
popular languages; minority languages give surface recognition.

License note: Users must accept BigCode's Terms of Use on the HF dataset
page before first download. `load_dataset` will raise a clear error if
terms haven't been accepted.

Output: JSONL with one file per line:
    {
        "text": "<source code>",
        "source": "stack_smol",
        "language": "python",
        "repo": "...",
        "path": "..."
    }

Usage:
    from curator.sources.stack_smol import StackSmolSource
    source = StackSmolSource(output_dir=Path("data/raw/stack_smol"))
    source.download()
"""

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class StackSmolSource:
    """
    Downloads the-stack-smol across all available languages.

    Args:
        output_dir: Directory to write output JSONL files.
        languages: Specific languages to include. None = discover from
            the HF repo's data/<lang>/data.json layout.
        min_length: Minimum file character length.
        shard_size: Files per output JSONL shard.
        max_docs: Maximum files to write. None = no limit. Used for
            mini runs to validate the pipeline.
    """

    DATASET_NAME = "bigcode/the-stack-smol"
    SOURCE_TAG = "stack_smol"

    def __init__(
        self,
        output_dir: Path,
        languages: list[str] | None = None,
        min_length: int = 50,
        shard_size: int = 50_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.languages = languages
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_languages(self) -> list[str]:
        """
        Discover available languages by listing the HF repo's data/<lang>/
        subdirectories. Falls back to a hard-coded list if the API call
        fails (e.g. network or auth issue).
        """
        if self.languages is not None:
            return list(self.languages)

        try:
            api = HfApi()
            files = api.list_repo_files(self.DATASET_NAME, repo_type="dataset")
            langs = sorted({
                f.split("/")[1]
                for f in files
                if f.startswith("data/") and f.endswith("/data.json")
            })
            if not langs:
                raise RuntimeError("No data/<lang>/data.json paths found")
            log.info(f"the-stack-smol: discovered {len(langs)} languages")
            return langs
        except Exception as e:
            log.error(
                f"Could not list languages for {self.DATASET_NAME}: {e}. "
                f"Have you accepted the dataset's Terms of Use on HuggingFace?"
            )
            raise

    def download(self) -> list[Path]:
        """Download the-stack-smol and write to sharded JSONL files."""
        languages = self._resolve_languages()

        output_files: list[Path] = []
        shard_idx = 0
        buffer: list[dict] = []
        total_written = 0
        total_skipped = 0
        stop = False

        for lang in languages:
            if stop:
                break

            log.info(f"Loading the-stack-smol — {lang}...")
            try:
                ds = load_dataset(
                    self.DATASET_NAME,
                    data_files=f"data/{lang}/data.json",
                    split="train",
                    trust_remote_code=True,
                )
            except Exception as e:
                log.warning(f"  Failed to load {lang}: {e} — skipping")
                continue

            for sample in tqdm(ds, desc=f"stack-smol {lang}", unit="file", leave=False):
                record = self._format(sample, lang)
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
            f"the-stack-smol complete — "
            f"written: {total_written:,}, "
            f"skipped: {total_skipped:,}, "
            f"shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _format(self, sample: dict, language: str) -> dict | None:
        """Format one stack-smol sample."""
        text = (sample.get("content") or "").strip()
        if len(text) < self.min_length:
            return None

        return {
            "text": text,
            "source": self.SOURCE_TAG,
            "language": language,
            "repo": sample.get("repository_name", "") or sample.get("repo", ""),
            "path": sample.get("path", ""),
        }

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard."""
        path = self.output_dir / f"stack_smol_{shard_idx:04d}.jsonl"
        with open(path, "wb") as f:
            for record in records:
                f.write(orjson.dumps(record))
                f.write(b"\n")
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} files → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("stack_smol_*.jsonl"))
        total_files = 0
        total_chars = 0
        lang_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_files += 1
                    total_chars += len(record.get("text", ""))
                    lang = record.get("language", "unknown")
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

        return {
            "shards": len(shards),
            "files": total_files,
            "total_chars": total_chars,
            "avg_chars_per_file": total_chars // max(total_files, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "by_language": lang_counts,
        }