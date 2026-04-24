"""
curator/sources/stack_v1.py
----------------------------
the-stack-dedup data source (v1).

Streams `bigcode/the-stack-dedup` filtered to 4 languages (Python, Go,
Rust, Shell). Unlike the-stack-v2-dedup, v1 stores file content inline
in the parquet shards, so no external content-fetching is required.
This makes v1 practical at any scale — the bottleneck is HF parquet
download bandwidth, not a third-party API's rate limit.

Why v1 instead of v2?
    The-stack-v2-dedup stores only file metadata on HuggingFace and
    hosts content on the Software Heritage Archive (SWH). SWH's API
    rate limits (authenticated: ~1200 req/hour) make it impractical
    to fetch millions of files. v1 embeds content inline, sidestepping
    the issue entirely. Quality difference at SLM scale is negligible
    — both are deduped and license-filtered; v2 adds near-dup removal
    and stricter PII scrubbing that matter more at frontier scale.

Languages: Python, Go, Rust, Shell. Matches the locked 4-language
scope for the code mix — curated CSN overlap (Python, Go), plus
languages without CSN coverage (Rust, Shell).

Output: JSONL with one file per line:
    {
        "text": "<source code>",
        "source": "stack_v1",
        "language": "python",
        "repo": "...",
        "path": "..."
    }

Usage:
    from curator.sources.stack_v1 import StackV1Source
    source = StackV1Source(output_dir=Path("data/raw/stack_v1"))
    source.download()

License note: Users must accept BigCode's Terms of Use on the HF
dataset page before first download. All included files carry
permissive licenses per the-stack-dedup's filtering.
"""

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


# v1 directory names are lowercase (verified against the-stack-dedup
# HF repo tree). v2 used capitalized names ("Python", "Go") — be careful
# when cross-referencing between versions.
DEFAULT_LANGUAGES = ["python", "go", "rust", "shell"]


class StackV1Source:
    """
    Loads the-stack-dedup one language at a time via parquet streaming.

    Args:
        output_dir: Directory to write output JSONL files.
        languages: Languages to include. Default: Python, Go, Rust, Shell.
        min_length: Minimum file character length.
        shard_size: Files per output JSONL shard.
        max_docs: Maximum files to write. None = no limit. Used for
            mini runs to validate the pipeline.
    """

    DATASET_NAME = "bigcode/the-stack-dedup"
    SOURCE_TAG = "stack_v1"

    def __init__(
        self,
        output_dir: Path,
        languages: list[str] | None = None,
        min_length: int = 50,
        shard_size: int = 10_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        # Keep as list (not set) so iteration order is deterministic.
        self.languages = list(languages or DEFAULT_LANGUAGES)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Iterate per-language shards of the-stack-dedup and write JSONL."""
        existing_shards = sorted(self.output_dir.glob("stack_v1_*.jsonl"))
        shard_idx = len(existing_shards)

        log.info(f"Loading {self.DATASET_NAME} per-language...")
        log.info(f"  Languages: {self.languages}")
        if self.max_docs:
            log.info(f"the-stack-v1: capped at {self.max_docs:,} files (mini run)")

        output_files: list[Path] = []
        buffer: list[dict] = []
        total_written = 0
        total_skipped_short = 0
        stop = False

        pbar = tqdm(desc="the-stack-v1", unit="file")

        for lang in self.languages:
            if stop:
                break

            log.info(f"  Loading language: {lang}")
            try:
                # data_dir filters to just this language's parquet files,
                # avoiding a full-corpus scan.
                ds = load_dataset(
                    self.DATASET_NAME,
                    data_dir=f"data/{lang}",
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
            except Exception:
                log.exception(f"  Failed to load {lang} — skipping")
                continue

            for sample in ds:
                if stop:
                    break

                content = (sample.get("content") or "").strip()
                if len(content) < self.min_length:
                    total_skipped_short += 1
                    continue

                buffer.append({
                    "text": content,
                    "source": self.SOURCE_TAG,
                    "language": lang,
                    "repo": sample.get("repository_name") or sample.get("max_stars_repo_name") or "",
                    "path": sample.get("max_stars_repo_path") or sample.get("path") or "",
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
            pbar.update(len(buffer))

        pbar.close()

        log.info(
            f"the-stack-v1 complete — "
            f"written: {total_written:,}, "
            f"skipped short: {total_skipped_short:,} (< {self.min_length} chars), "
            f"new shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard atomically via .tmp rename."""
        path = self.output_dir / f"stack_v1_{shard_idx:04d}.jsonl"
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
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} files → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("stack_v1_*.jsonl"))
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