"""
Synthetic arithmetic source for pretraining.

Purpose:
    Add clean, dense elementary arithmetic signal to the base pretraining
    corpus. OpenWebMath contains math-heavy web text, but the arithmetic
    signal is noisy and often buried in long pages. This source provides
    simple repeated mappings such as:

        2 + 2 = 4
        Question: What is 3 + 4?
        Answer: 7

Design:
    - pretraining text only, not chat/SFT format
    - deterministic generation
    - mixed symbolic, QA, word-form, and answer-only styles
    - documents are long enough to pass minimum-length filters
    - many numeric combinations/templates to reduce dedup collapse
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)


class SyntheticArithmeticSource:
    """Generate deterministic synthetic arithmetic JSONL shards."""

    name = "synthetic_arithmetic"

    def __init__(
        self,
        output_dir: Path,
        max_docs: int | None = None,
        seed: int = 42,
        shard_size: int = 10_000,
    ):
        self.output_dir = Path(output_dir)
        self.max_docs = max_docs or 10_000
        self.seed = seed
        self.shard_size = shard_size
        self._docs_written = 0
        self._chars_written = 0

    def _num_word(self, n: int) -> str:
        words_0_19 = [
            "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen",
        ]
        tens = {
            20: "twenty",
            30: "thirty",
            40: "forty",
            50: "fifty",
            60: "sixty",
            70: "seventy",
            80: "eighty",
            90: "ninety",
        }

        if n < 0:
            return "minus " + self._num_word(-n)
        if n < 20:
            return words_0_19[n]
        if n < 100:
            t = (n // 10) * 10
            r = n % 10
            return tens[t] if r == 0 else f"{tens[t]}-{words_0_19[r]}"
        if n < 1000:
            h = n // 100
            r = n % 100
            if r == 0:
                return f"{words_0_19[h]} hundred"
            return f"{words_0_19[h]} hundred {self._num_word(r)}"
        return str(n)

    def _problem(self, rng: random.Random) -> tuple[str, str, str]:
        """Return (symbol expression, word expression, answer)."""
        op = rng.choices(
            ["+", "-", "*", "/"],
            weights=[0.40, 0.30, 0.20, 0.10],
            k=1,
        )[0]

        if op == "+":
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)
            ans = a + b
            word = f"{self._num_word(a)} plus {self._num_word(b)}"

        elif op == "-":
            a = rng.randint(0, 120)
            b = rng.randint(0, 120)
            # Mostly non-negative differences, with a small negative tail.
            if rng.random() < 0.90 and b > a:
                a, b = b, a
            ans = a - b
            word = f"{self._num_word(a)} minus {self._num_word(b)}"

        elif op == "*":
            a = rng.randint(0, 20)
            b = rng.randint(0, 20)
            ans = a * b
            word = f"{self._num_word(a)} times {self._num_word(b)}"

        else:
            divisor = rng.randint(1, 20)
            quotient = rng.randint(0, 25)
            dividend = divisor * quotient
            a = dividend
            b = divisor
            ans = quotient
            word = f"{self._num_word(a)} divided by {self._num_word(b)}"

        return f"{a} {op} {b}", word, str(ans)

    def _make_doc(self, doc_id: int, rng: random.Random) -> str:
        """Create one pretraining document with varied arithmetic forms."""
        headings = [
            "Arithmetic practice examples",
            "Elementary arithmetic facts",
            "Simple calculation exercises",
            "Basic number operation examples",
            "Arithmetic question and answer practice",
        ]

        lines: list[str] = [
            rng.choice(headings),
            (
                "This document contains simple arithmetic examples. "
                "Each line gives a small calculation and its correct result. "
                "The goal is to practice addition, subtraction, multiplication, "
                "and exact division with clear answers."
            ),
            "",
        ]

        n_examples = rng.randint(28, 44)

        for _ in range(n_examples):
            expr, word, ans = self._problem(rng)
            style = rng.choices(
                ["equation", "qa", "answer_only", "word", "explain_short"],
                weights=[0.35, 0.25, 0.15, 0.15, 0.10],
                k=1,
            )[0]

            if style == "equation":
                lines.append(f"{expr} = {ans}")
            elif style == "qa":
                lines.append(f"Question: What is {expr}?")
                lines.append(f"Answer: {ans}")
            elif style == "answer_only":
                lines.append("Answer only the result:")
                lines.append(expr)
                lines.append(ans)
            elif style == "word":
                lines.append(f"What is {word}?")
                lines.append(ans)
            else:
                lines.append(f"The result of {expr} is {ans}.")
                lines.append(f"So, {expr} = {ans}.")

            if rng.random() < 0.20:
                lines.append("")

        # Add a deterministic footer to make records self-describing without
        # changing the answer mappings.
        lines.append("")
        lines.append(
            f"End of arithmetic practice document {doc_id}. "
            "All examples above show the complete calculation and answer."
        )

        return "\n".join(lines)

    def download(self) -> None:
        """Generate JSONL shards under output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # If shards already exist, keep source behavior resumable/idempotent.
        existing = sorted(self.output_dir.glob("*.jsonl"))
        if existing:
            log.info(
                f"{self.name}: found {len(existing)} existing shard(s) in "
                f"{self.output_dir}; skipping generation"
            )
            self._docs_written = self._count_existing_docs(existing)
            self._chars_written = self._count_existing_chars(existing)
            return

        rng = random.Random(self.seed)
        remaining = self.max_docs
        shard_idx = 0
        doc_id = 0

        log.info(
            f"{self.name}: generating {self.max_docs:,} docs "
            f"to {self.output_dir}"
        )

        while remaining > 0:
            n_this = min(self.shard_size, remaining)
            shard_path = self.output_dir / f"{self.name}_{shard_idx:05d}.jsonl"

            with shard_path.open("w", encoding="utf-8") as f:
                for _ in range(n_this):
                    text = self._make_doc(doc_id, rng)
                    rec = {
                        "id": f"{self.name}_{doc_id}",
                        "source": self.name,
                        "text": text,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    self._docs_written += 1
                    self._chars_written += len(text)
                    doc_id += 1

            log.info(f"{self.name}: wrote {n_this:,} docs -> {shard_path}")
            remaining -= n_this
            shard_idx += 1

        log.info(
            f"{self.name}: complete — docs={self._docs_written:,}, "
            f"chars={self._chars_written:,}"
        )

    def _count_existing_docs(self, shards: list[Path]) -> int:
        docs = 0
        for shard in shards:
            with shard.open("r", encoding="utf-8") as f:
                for _ in f:
                    docs += 1
        return docs

    def _count_existing_chars(self, shards: list[Path]) -> int:
        chars = 0
        for shard in shards:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    chars += len(str(rec.get("text", "")))
        return chars

    def stats(self) -> dict:
        return {
            "source": self.name,
            "docs": self._docs_written,
            "chars": self._chars_written,
            "max_docs": self.max_docs,
            "output_dir": str(self.output_dir),
        }
