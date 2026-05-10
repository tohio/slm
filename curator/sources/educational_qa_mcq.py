"""
Synthetic educational QA/MCQ data source.

Generates small, deterministic educational question-answer examples for
pretraining. This source is designed to add QA, MCQ, answer-selection, cloze,
and explanation-format signal without using MMLU, ARC, HellaSwag, TruthfulQA,
HumanEval, APPS, GSM8K, or benchmark-derived examples.

The initial implementation is local/template-driven. Later versions can replace
or extend this with source-grounded generation from approved educational text.
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import orjson

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class EducationalQAMCQSource:
    """
    Generate educational QA/MCQ examples and write sharded JSONL.

    Args:
        output_dir: Directory to write output JSONL files.
        max_docs: Maximum examples to write. None uses DEFAULT_DOCS.
        shard_size: Documents per output JSONL shard.
        seed: RNG seed for deterministic generation.
    """

    SOURCE_TAG = "educational_qa_mcq"
    SHARD_PREFIX = "educational_qa_mcq"
    DEFAULT_DOCS = 100_000

    EXAMPLES = [
        {
            "subject": "science",
            "question": "What force pulls objects toward Earth?",
            "answer": "gravity",
            "explanation": "Gravity is the force that attracts objects with mass toward each other, including objects near Earth.",
            "choices": ["magnetism", "gravity", "friction", "evaporation"],
            "correct": "B",
        },
        {
            "subject": "science",
            "question": "What do plants use to make food from sunlight?",
            "answer": "photosynthesis",
            "explanation": "Photosynthesis is the process plants use to convert sunlight, water, and carbon dioxide into sugars.",
            "choices": ["condensation", "photosynthesis", "erosion", "respiration"],
            "correct": "B",
        },
        {
            "subject": "science",
            "question": "What is the basic unit of life?",
            "answer": "the cell",
            "explanation": "Cells are the smallest units that carry out the basic functions of living organisms.",
            "choices": ["atom", "cell", "planet", "mineral"],
            "correct": "B",
        },
        {
            "subject": "math_concepts",
            "question": "What is the perimeter of a square with side length 4?",
            "answer": "16",
            "explanation": "A square has four equal sides, so the perimeter is 4 + 4 + 4 + 4 = 16.",
            "choices": ["8", "12", "16", "20"],
            "correct": "C",
        },
        {
            "subject": "math_concepts",
            "question": "What is a prime number?",
            "answer": "a number greater than 1 with exactly two positive factors",
            "explanation": "A prime number has exactly two positive factors: 1 and itself.",
            "choices": [
                "a number divisible by every number",
                "a number greater than 1 with exactly two positive factors",
                "a number less than zero",
                "a number with no factors",
            ],
            "correct": "B",
        },
        {
            "subject": "history_geography",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "explanation": "Paris is the capital city of France.",
            "choices": ["Madrid", "Paris", "Rome", "Berlin"],
            "correct": "B",
        },
        {
            "subject": "history_geography",
            "question": "Which ocean is the largest on Earth?",
            "answer": "the Pacific Ocean",
            "explanation": "The Pacific Ocean is the largest ocean by area.",
            "choices": ["Atlantic Ocean", "Indian Ocean", "Pacific Ocean", "Arctic Ocean"],
            "correct": "C",
        },
        {
            "subject": "computer_science",
            "question": "What is a variable in programming?",
            "answer": "a named place to store a value",
            "explanation": "A variable gives a name to a value so a program can refer to it later.",
            "choices": [
                "a type of monitor",
                "a named place to store a value",
                "a network cable",
                "a compiler error",
            ],
            "correct": "B",
        },
        {
            "subject": "computer_science",
            "question": "What does a function do in a program?",
            "answer": "it groups reusable instructions",
            "explanation": "A function groups instructions so they can be called and reused.",
            "choices": [
                "stores only images",
                "groups reusable instructions",
                "turns off the computer",
                "removes all variables",
            ],
            "correct": "B",
        },
        {
            "subject": "general_knowledge",
            "question": "Why do people use calendars?",
            "answer": "to organize dates and events",
            "explanation": "Calendars help track days, months, appointments, deadlines, and events.",
            "choices": [
                "to measure temperature",
                "to organize dates and events",
                "to cook food",
                "to store electricity",
            ],
            "correct": "B",
        },
        {
            "subject": "reading_common_sense",
            "question": "If someone carries an umbrella on a rainy day, what are they likely trying to do?",
            "answer": "stay dry",
            "explanation": "An umbrella blocks rain, so carrying one on a rainy day helps a person stay dry.",
            "choices": ["stay dry", "make bread", "charge a phone", "paint a wall"],
            "correct": "A",
        },
    ]

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
        """Generate educational QA/MCQ examples."""
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
        qa_type = self._qa_type_for_idx(idx)
        example = rng.choice(self.EXAMPLES)
        text = self._render(example, qa_type, rng)
        text = self._add_learning_context(text, example, qa_type, idx)

        stable_id = hashlib.sha256(
            f"{self.SOURCE_TAG}:{idx}:{qa_type}:{text}".encode()
        ).hexdigest()[:16]

        return {
            "id": stable_id,
            "text": text,
            "source": self.SOURCE_TAG,
            "qa_type": qa_type,
            "subject": example["subject"],
            "generated": True,
            "benchmark_excluded": True,
        }

    def _qa_type_for_idx(self, idx: int) -> str:
        # Locked format mix:
        # short_qa             35%
        # multiple_choice_qa   35%
        # qa_with_explanation  20%
        # cloze_completion     10%
        bucket = idx % 100
        if bucket < 35:
            return "short_qa"
        if bucket < 70:
            return "multiple_choice_qa"
        if bucket < 90:
            return "qa_with_explanation"
        return "cloze_completion"

    def _add_learning_context(
        self,
        text: str,
        ex: dict,
        qa_type: str,
        idx: int,
    ) -> str:
        # Add compact educational context so the local QA/MCQ source has
        # useful variation instead of exact repeated templates.
        levels = [
            "basic recall",
            "short explanation",
            "concept check",
            "review question",
            "practice item",
            "quick assessment",
        ]
        goals = [
            "state the answer clearly",
            "connect the answer to the concept",
            "avoid unnecessary detail",
            "use the provided facts",
            "focus on the key idea",
            "keep the response concise",
        ]
        level = levels[idx % len(levels)]
        goal = goals[(idx // len(levels)) % len(goals)]
        subject = ex.get("subject", "general")

        return (
            f"{text}\n"
            f"Learning context: {level} in {subject}.\n"
            f"Answering goal: {goal}. Format: {qa_type}.\n"
        )

    def _render(self, ex: dict, qa_type: str, rng: random.Random) -> str:
        if qa_type == "short_qa":
            formats = [
                f"Question: {ex['question']}\nAnswer: {ex['answer']}.",
                f"Q: {ex['question']}\nA: {ex['answer']}.",
                f"{ex['question']}\n{ex['answer']}.",
            ]
            return rng.choice(formats)

        if qa_type == "multiple_choice_qa":
            labels = ["A", "B", "C", "D"]
            choices = ex["choices"]
            choice_lines = "\n".join(
                f"{label}. {choice}" for label, choice in zip(labels, choices)
            )
            correct_idx = labels.index(ex["correct"])
            correct_answer = choices[correct_idx]
            formats = [
                (
                    f"Question: {ex['question']}\n\n"
                    f"{choice_lines}\n\n"
                    f"Answer: {ex['correct']}. {correct_answer}"
                ),
                (
                    f"Choose the correct answer.\n"
                    f"{ex['question']}\n\n"
                    f"{choice_lines}\n\n"
                    f"Correct answer: {ex['correct']}. {correct_answer}"
                ),
            ]
            return rng.choice(formats)

        if qa_type == "qa_with_explanation":
            formats = [
                (
                    f"Question: {ex['question']}\n"
                    f"Answer: {ex['answer']}.\n"
                    f"Explanation: {ex['explanation']}"
                ),
                (
                    f"Q: {ex['question']}\n"
                    f"A: {ex['answer']}.\n"
                    f"Why: {ex['explanation']}"
                ),
            ]
            return rng.choice(formats)

        # cloze_completion
        cloze = self._to_cloze(ex)
        return f"{cloze}\nAnswer: {ex['answer']}."

    def _to_cloze(self, ex: dict) -> str:
        question = ex["question"]
        answer = ex["answer"]

        if "capital of France" in question:
            return "The capital of France is ___."
        if "largest on Earth" in question:
            return "The largest ocean on Earth is ___."
        if "force pulls objects toward Earth" in question:
            return "The force that pulls objects toward Earth is ___."
        if "plants use to make food" in question:
            return "Plants use ___ to make food from sunlight."
        if "basic unit of life" in question:
            return "The basic unit of life is ___."
        if "perimeter of a square" in question:
            return "The perimeter of a square with side length 4 is ___."
        if "prime number" in question:
            return "A prime number is ___."
        if "variable in programming" in question:
            return "In programming, a variable is ___."
        if "function do in a program" in question:
            return "In a program, a function ___."
        if "calendars" in question:
            return "People use calendars to ___."
        if "umbrella" in question:
            return "A person carrying an umbrella on a rainy day is likely trying to ___."

        return f"The answer is ___ for this question: {question}"

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
        by_qa_type: dict[str, int] = {}
        by_subject: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_docs += 1
                    total_chars += len(record.get("text", ""))

                    qa_type = record.get("qa_type", "unknown")
                    subject = record.get("subject", "unknown")
                    by_qa_type[qa_type] = by_qa_type.get(qa_type, 0) + 1
                    by_subject[subject] = by_subject.get(subject, 0) + 1

        return {
            "shards": len(shards),
            "documents": total_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": total_chars // max(total_docs, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "by_qa_type": by_qa_type,
            "by_subject": by_subject,
        }
