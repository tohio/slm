"""
curator/sources/code_search_net.py
------------------------------------
CodeSearchNet data source.

Downloads the CodeSearchNet dataset via HuggingFace datasets.
~2M code functions with docstrings across 6 languages: Python, Java,
JavaScript, PHP, Ruby, Go.

We use func_code_string (code without docstring) + func_documentation_string
joined with a comment-styled docstring header. This avoids the duplicated-
docstring bug in the earlier implementation which used whole_func_string
(which already includes the docstring) AND prepended the docstring — every
record had the docstring twice.

Output: JSONL with one function per line:
    {
        "text": "...",         # docstring-as-comment + code
        "source": "codesearchnet",
        "language": "python",
        "repo": "...",
        "path": "..."
    }

Usage:
    from curator.sources.code_search_net import CodeSearchNetSource
    source = CodeSearchNetSource(output_dir=Path("data/raw/codesearchnet"))
    source.download()
"""

import logging
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)

LANGUAGES = ["python", "java", "javascript", "php", "ruby", "go"]

# Comment style per language. Everything else falls back to "//".
_COMMENT_CHAR = {
    "python": "#",
    "ruby": "#",
    # java, javascript, php, go → //
}


def _strip_inline_docstring(code: str, docstring: str) -> str:
    """
    Remove the inline docstring from a function body.

    CodeSearchNet's func_code_string contains the full function including
    the docstring as a triple-quoted string literal. Since we prepend the
    docstring separately as a comment, we need to remove the inline one
    to avoid duplication.

    Strategy:
        1. Find the first triple-quoted block in the code.
        2. If its contents match the docstring (after whitespace normalization),
           splice it out along with any trailing newline.
        3. If no match, return the code unchanged — better to ship a doubled
           docstring than corrupt the code with an aggressive regex.

    Handles both triple-double and triple-single quote styles. Ruby's
    commented docstrings don't need this function but the loop skips strings
    that don't match anyway.
    """
    if not docstring or not code:
        return code

    # Normalize whitespace for matching — docstrings often have indentation
    # in the inline version that's stripped in func_documentation_string.
    def _norm(s: str) -> str:
        return " ".join(s.split())

    docstring_norm = _norm(docstring)

    for quote in ('"""', "'''"):
        start = code.find(quote)
        if start == -1:
            continue
        end = code.find(quote, start + 3)
        if end == -1:
            continue
        inline = code[start + 3:end]
        if _norm(inline) == docstring_norm or docstring_norm in _norm(inline):
            # Splice out. Drop a trailing newline if present so we don't
            # leave a blank line at the start of the function body.
            after = end + 3
            if after < len(code) and code[after] == "\n":
                after += 1
            return (code[:start].rstrip() + "\n" + code[after:]).rstrip() + "\n"

    return code


class CodeSearchNetSource:
    """
    Downloads and formats CodeSearchNet for LLM pretraining.

    Each sample is formatted as:
        <comment> <docstring line 1>
        <comment> <docstring line 2>
        ...
        <code>

    Args:
        output_dir: Directory to write output JSONL files.
        languages: Programming languages to include.
        min_code_length: Minimum code character length.
        min_docstring_length: Minimum docstring character length.
        splits: Dataset splits to use.
        shard_size: Samples per output JSONL shard.
        max_docs: Maximum samples to write. None = no limit.
    """

    DATASET_NAME = "code_search_net"
    SOURCE_TAG = "codesearchnet"

    def __init__(
        self,
        output_dir: Path,
        languages: list[str] = LANGUAGES,
        min_code_length: int = 100,
        min_docstring_length: int = 20,
        splits: list[str] = ("train", "validation", "test"),
        shard_size: int = 100_000,
        max_docs: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.languages = list(languages)
        self.min_code_length = min_code_length
        self.min_docstring_length = min_docstring_length
        self.splits = list(splits)
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Download CodeSearchNet and write to sharded JSONL files."""
        output_files: list[Path] = []
        shard_idx = 0
        buffer: list[dict] = []
        total_written = 0
        total_skipped = 0
        stop = False

        # Loop over (language, split) pairs so we always know which language
        # each sample belongs to, regardless of whether the dataset carries
        # a per-sample 'language' column.
        for language in self.languages:
            if stop:
                break
            log.info(f"Loading CodeSearchNet — {language}...")

            for split in self.splits:
                if stop:
                    break
                try:
                    ds = load_dataset(
                        self.DATASET_NAME,
                        language,
                        split=split,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    log.warning(
                        f"  Failed to load {language}/{split}: {e} — skipping"
                    )
                    continue

                log.info(f"  {language}/{split}: {len(ds):,} samples")

                for sample in tqdm(
                    ds,
                    desc=f"CSN {language}/{split}",
                    unit="sample",
                    leave=False,
                ):
                    record = self._format(sample, language)
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

        # Flush remainder
        if buffer:
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)

        log.info(
            f"CodeSearchNet complete — "
            f"written: {total_written:,}, "
            f"skipped: {total_skipped:,}, "
            f"shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _format(self, sample: dict, language: str) -> dict | None:
        """
        Format one CodeSearchNet sample.

        Empirically, func_code_string DOES contain the inline docstring
        (verified against the current HF dataset). If we simply prepend the
        docstring as a comment and append func_code_string, the docstring
        appears twice. We strip the inline docstring out first for Python
        and Ruby (the two languages with inline-string docstrings); for
        C-style languages the docstring is already outside the function
        body as /** ... */ and is not duplicated.
        """
        code = (sample.get("func_code_string") or "").strip()
        if not code:
            code = (sample.get("code") or "").strip()

        docstring = (
            sample.get("func_documentation_string")
            or sample.get("docstring")
            or ""
        ).strip()

        # Strip the inline docstring from the code body so it doesn't appear
        # twice (once as prepended comment, once inline).
        if language in ("python", "ruby") and docstring:
            code = _strip_inline_docstring(code, docstring)

        if len(code) < self.min_code_length:
            return None
        if len(docstring) < self.min_docstring_length:
            return None

        comment_char = _COMMENT_CHAR.get(language, "//")
        docstring_lines = "\n".join(
            f"{comment_char} {line}" for line in docstring.split("\n")
        )
        text = f"{docstring_lines}\n{code}"

        return {
            "text": text,
            "source": self.SOURCE_TAG,
            "language": language,
            "repo": sample.get("repository_name", "") or sample.get("repo", ""),
            "path": sample.get("func_path_in_repository", "") or sample.get("path", ""),
        }

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard."""
        path = self.output_dir / f"codesearchnet_{shard_idx:04d}.jsonl"
        with open(path, "wb") as f:
            for record in records:
                f.write(orjson.dumps(record))
                f.write(b"\n")
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} samples → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("codesearchnet_*.jsonl"))
        total_samples = 0
        total_chars = 0
        lang_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_samples += 1
                    total_chars += len(record.get("text", ""))
                    lang = record.get("language", "unknown")
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

        return {
            "shards": len(shards),
            "samples": total_samples,
            "total_chars": total_chars,
            "avg_chars_per_sample": total_chars // max(total_samples, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "by_language": lang_counts,
        }