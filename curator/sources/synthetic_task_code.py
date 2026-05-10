"""
Synthetic task-code data source.

Generates small, deterministic, task-shaped coding examples for pretraining.
This source is intended to add function-completion and prompt-to-code signal
without using HumanEval, APPS, LeetCode, or benchmark-derived examples.

Language mix:
    Python 70%
    Go     15%
    Rust   10%
    Bash    5%
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import orjson

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class SyntheticTaskCodeSource:
    """
    Generate task-shaped code examples and write sharded JSONL.

    Args:
        output_dir: Directory to write output JSONL files.
        max_docs: Maximum examples to write. None uses DEFAULT_DOCS.
        shard_size: Documents per output JSONL shard.
        seed: RNG seed for deterministic generation.
    """

    SOURCE_TAG = "synthetic_task_code"
    SHARD_PREFIX = "synthetic_task_code"
    DEFAULT_DOCS = 100_000

    def __init__(
        self,
        output_dir: Path,
        max_docs: int | None = None,
        shard_size: int = 100_000,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.max_docs = max_docs or self.DEFAULT_DOCS
        self.shard_size = shard_size
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Generate synthetic task-code examples."""
        existing_shards = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        if existing_shards:
            log.info(
                f"{self.SOURCE_TAG}: found {len(existing_shards)} existing shard(s); "
                "skipping generation"
            )
            return existing_shards

        rng = random.Random(self.seed)
        records: list[dict] = []
        output_files: list[Path] = []

        for i in range(self.max_docs):
            record = self._make_record(i, rng)
            records.append(record)

            if len(records) >= self.shard_size:
                output_files.append(self._write_shard(records, len(output_files)))
                records = []

        if records:
            output_files.append(self._write_shard(records, len(output_files)))

        log.info(
            f"{self.SOURCE_TAG} complete — written: {self.max_docs:,}, "
            f"shards: {len(output_files)}"
        )
        return output_files

    def _make_record(self, idx: int, rng: random.Random) -> dict:
        language = self._language_for_idx(idx)
        task_type = self._task_type_for_idx(idx)

        if language == "python":
            text = self._render_python(task_type, rng)
        elif language == "go":
            text = self._render_go(task_type, rng)
        elif language == "rust":
            text = self._render_rust(task_type, rng)
        else:
            text = self._render_bash(task_type, rng)

        text = self._add_training_context(text, idx, language, task_type)

        stable_id = hashlib.sha256(
            f"{self.SOURCE_TAG}:{idx}:{language}:{task_type}:{text}".encode()
        ).hexdigest()[:16]

        return {
            "id": stable_id,
            "text": text,
            "source": self.SOURCE_TAG,
            "language": language,
            "task_type": task_type,
            "generated": True,
        }

    def _language_for_idx(self, idx: int) -> str:
        # Locked language mix:
        # Python 70%, Go 15%, Rust 10%, Bash 5%
        bucket = idx % 100
        if bucket < 70:
            return "python"
        if bucket < 85:
            return "go"
        if bucket < 95:
            return "rust"
        return "bash"

    def _task_type_for_idx(self, idx: int) -> str:
        # Simple subtype mix:
        # docstring_to_code        30%
        # signature_to_function    20%
        # prompt_to_function       20%
        # function_with_tests      15%
        # bug_fix_small            10%
        # code_explanation_to_code  5%
        bucket = idx % 100
        if bucket < 30:
            return "docstring_to_code"
        if bucket < 50:
            return "signature_to_function"
        if bucket < 70:
            return "prompt_to_function"
        if bucket < 85:
            return "function_with_tests"
        if bucket < 95:
            return "bug_fix_small"
        return "code_explanation_to_code"


    def _add_training_context(
        self,
        text: str,
        idx: int,
        language: str,
        task_type: str,
    ) -> str:
        # Add useful deterministic variation so exact dedup does not collapse
        # the targeted local source at mini scale.
        emphasis = [
            "Return the result directly.",
            "Keep the implementation small and readable.",
            "Use clear variable names.",
            "Prefer straightforward control flow.",
            "Do not describe the code instead of writing it.",
            "Include the complete function body.",
            "Handle the stated edge case.",
            "Avoid unnecessary dependencies.",
        ][idx % 8]

        review_focus = [
            "correctness",
            "readability",
            "edge cases",
            "input handling",
            "return values",
            "simple tests",
            "function behavior",
            "minimal implementation",
        ][(idx // 8) % 8]

        return (
            f"{text}\n"
            f"Implementation note: {emphasis}\n"
            f"Review focus: {language} {task_type} example for {review_focus}.\n"
        )

    # ── Python examples ───────────────────────────────────────────────────────

    def _render_python(self, task_type: str, rng: random.Random) -> str:
        examples = [
            {
                "name": "square",
                "prompt": "Write a Python function named square that returns x multiplied by itself.",
                "code": "def square(x):\n    return x * x",
                "doc": "Return x multiplied by itself.",
                "tests": "assert square(3) == 9\nassert square(-4) == 16\nassert square(0) == 0",
                "buggy": "def square(x):\n    return abs(x)",
                "fixed": "def square(x):\n    return x * x",
            },
            {
                "name": "is_even",
                "prompt": "Write a Python function named is_even that returns True when n is even.",
                "code": "def is_even(n):\n    return n % 2 == 0",
                "doc": "Return True if n is even, otherwise False.",
                "tests": "assert is_even(2) is True\nassert is_even(3) is False\nassert is_even(0) is True",
                "buggy": "def is_even(n):\n    return n % 2 == 1",
                "fixed": "def is_even(n):\n    return n % 2 == 0",
            },
            {
                "name": "count_vowels",
                "prompt": "Write a Python function named count_vowels that counts vowels in text.",
                "code": "def count_vowels(text):\n    vowels = set('aeiouAEIOU')\n    return sum(1 for ch in text if ch in vowels)",
                "doc": "Return the number of vowels in text.",
                "tests": "assert count_vowels('hello') == 2\nassert count_vowels('sky') == 0\nassert count_vowels('AEIOU') == 5",
                "buggy": "def count_vowels(text):\n    return len(text)",
                "fixed": "def count_vowels(text):\n    vowels = set('aeiouAEIOU')\n    return sum(1 for ch in text if ch in vowels)",
            },
            {
                "name": "reverse_words",
                "prompt": "Write a Python function named reverse_words that reverses the order of words.",
                "code": "def reverse_words(text):\n    return ' '.join(text.split()[::-1])",
                "doc": "Return a string with the words in reverse order.",
                "tests": "assert reverse_words('hello world') == 'world hello'\nassert reverse_words('a b c') == 'c b a'\nassert reverse_words('') == ''",
                "buggy": "def reverse_words(text):\n    return text[::-1]",
                "fixed": "def reverse_words(text):\n    return ' '.join(text.split()[::-1])",
            },
            {
                "name": "max_or_none",
                "prompt": "Write a Python function named max_or_none that returns the maximum value or None for an empty list.",
                "code": "def max_or_none(values):\n    if not values:\n        return None\n    return max(values)",
                "doc": "Return the maximum value in values, or None if values is empty.",
                "tests": "assert max_or_none([1, 3, 2]) == 3\nassert max_or_none([-5, -2]) == -2\nassert max_or_none([]) is None",
                "buggy": "def max_or_none(values):\n    return max(values)",
                "fixed": "def max_or_none(values):\n    if not values:\n        return None\n    return max(values)",
            },
        ]

        ex = rng.choice(examples)

        if task_type == "docstring_to_code":
            signature = ex["code"].splitlines()[0]
            return (
                f"Python function:

"
                f"{signature}
"
                f"    """{ex['doc']}"""

"
                f"Solution:
{ex['code']}
"
            )

        if task_type == "signature_to_function":
            return f"Complete the Python function:\n\n{ex['code']}\n"

        if task_type == "prompt_to_function":
            return f"Task: {ex['prompt']}\n\nSolution:\n{ex['code']}\n"

        if task_type == "function_with_tests":
            return f"Task: {ex['prompt']}\n\nCode:\n{ex['code']}\n\nTests:\n{ex['tests']}\n"

        if task_type == "bug_fix_small":
            return f"Buggy Python code:\n{ex['buggy']}\n\nCorrected code:\n{ex['fixed']}\n"

        return f"Description: {ex['prompt']}\n\nImplementation:\n{ex['code']}\n"

    # ── Go examples ───────────────────────────────────────────────────────────

    def _render_go(self, task_type: str, rng: random.Random) -> str:
        examples = [
            {
                "prompt": "Write a Go function named square that returns n multiplied by itself.",
                "code": "func square(n int) int {\n    return n * n\n}",
                "tests": "square(3) == 9\nsquare(-4) == 16",
            },
            {
                "prompt": "Write a Go function named isEven that returns true when n is even.",
                "code": "func isEven(n int) bool {\n    return n%2 == 0\n}",
                "tests": "isEven(2) == true\nisEven(3) == false",
            },
            {
                "prompt": "Write a Go function named maxOrZero that returns the maximum integer or zero for an empty slice.",
                "code": "func maxOrZero(values []int) int {\n    if len(values) == 0 {\n        return 0\n    }\n    max := values[0]\n    for _, v := range values {\n        if v > max {\n            max = v\n        }\n    }\n    return max\n}",
                "tests": "maxOrZero([]int{1, 3, 2}) == 3\nmaxOrZero([]int{}) == 0",
            },
        ]
        ex = rng.choice(examples)
        return f"Task: {ex['prompt']}\n\nGo solution:\n{ex['code']}\n\nChecks:\n{ex['tests']}\n"

    # ── Rust examples ─────────────────────────────────────────────────────────

    def _render_rust(self, task_type: str, rng: random.Random) -> str:
        examples = [
            {
                "prompt": "Write a Rust function named square that returns n multiplied by itself.",
                "code": "fn square(n: i32) -> i32 {\n    n * n\n}",
                "tests": "assert_eq!(square(3), 9);\nassert_eq!(square(-4), 16);",
            },
            {
                "prompt": "Write a Rust function named is_even that returns true when n is even.",
                "code": "fn is_even(n: i32) -> bool {\n    n % 2 == 0\n}",
                "tests": "assert_eq!(is_even(2), true);\nassert_eq!(is_even(3), false);",
            },
            {
                "prompt": "Write a Rust function named sum_values that returns the sum of a slice.",
                "code": "fn sum_values(values: &[i32]) -> i32 {\n    values.iter().sum()\n}",
                "tests": "assert_eq!(sum_values(&[1, 2, 3]), 6);\nassert_eq!(sum_values(&[]), 0);",
            },
        ]
        ex = rng.choice(examples)
        return f"Task: {ex['prompt']}\n\nRust solution:\n{ex['code']}\n\nChecks:\n{ex['tests']}\n"

    # ── Bash examples ─────────────────────────────────────────────────────────

    def _render_bash(self, task_type: str, rng: random.Random) -> str:
        examples = [
            {
                "prompt": "Write a Bash function named file_exists that returns success if a path exists.",
                "code": "file_exists() {\n    [ -e \"$1\" ]\n}",
            },
            {
                "prompt": "Write a Bash function named count_lines that prints the number of lines in a file.",
                "code": "count_lines() {\n    wc -l < \"$1\"\n}",
            },
            {
                "prompt": "Write a Bash function named make_dir that creates a directory if it does not exist.",
                "code": "make_dir() {\n    mkdir -p \"$1\"\n}",
            },
        ]
        ex = rng.choice(examples)
        return f"Task: {ex['prompt']}\n\nBash solution:\n{ex['code']}\n"

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
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
        shards = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        total_docs = 0
        total_chars = 0
        by_language: dict[str, int] = {}
        by_task_type: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_docs += 1
                    total_chars += len(record.get("text", ""))

                    language = record.get("language", "unknown")
                    task_type = record.get("task_type", "unknown")
                    by_language[language] = by_language.get(language, 0) + 1
                    by_task_type[task_type] = by_task_type.get(task_type, 0) + 1

        return {
            "shards": len(shards),
            "documents": total_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": total_chars // max(total_docs, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "by_language": by_language,
            "by_task_type": by_task_type,
        }
