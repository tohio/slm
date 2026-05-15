from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from curator.scripts.groq_synthetic import GroqSyntheticSource


class SyntheticArithmeticSource(GroqSyntheticSource):
    SOURCE_TAG = "synthetic_arithmetic"
    SHARD_PREFIX = "synthetic_arithmetic"
    DEFAULT_DOCS = 100_000
    DEFAULT_BATCH_SIZE = 32

    FORMATS = [
        "qa_arithmetic",
        "word_problem",
        "comparison_arithmetic",
        "single_equation",
        "two_step_clear",
    ]

    def __init__(
        self,
        output_dir: Path,
        max_docs: int | None = None,
        max_chars: int | None = None,
        shard_size: int = 5_000,
        seed: int = 42,
    ):
        super().__init__(
            output_dir=output_dir,
            max_docs=max_docs,
            max_chars=max_chars,
            shard_size=shard_size,
            seed=seed,
        )

    def _record_metadata(self, row: dict, idx: int) -> dict:
        return {
            "category": "arithmetic",
            "format": row.get("format", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            "prompt_template": "synthetic_arithmetic_groq_v5_schema",
            "benchmark_excluded": True,
        }

    def _build_prompt(self, batch_count: int, start_index: int, rng: random.Random) -> str:
        specs = []
        for offset in range(batch_count):
            specs.append({
                "id": start_index + offset,
                "format": rng.choice(self.FORMATS),
                "difficulty": rng.choice(["single_step", "single_step", "two_step_clear"]),
                "range": rng.choice([
                    "0-20",
                    "0-100",
                    "small integers",
                    "multiplication table",
                    "division exact",
                ]),
            })

        specs_json = json.dumps(specs, indent=2)
        return f"""
Generate unique elementary arithmetic pretraining records.

Return a JSON object that matches the configured schema:
- top-level key: records
- records is an array of objects
- each record has question, answer, format, difficulty, metadata
- do not return a text field
- question and answer must be single-line strings
- the Python adapter will build final multiline training text

Rules:
- Each record must contain exactly one arithmetic question and one short answer.
- Include the final answer explicitly in the answer field.
- Use original numbers and wording.
- Allowed: single-step arithmetic, simple word problems, simple comparisons,
  simple solve-for-x equations, and clear two-step arithmetic.
- Do not include multiple separate questions in one record.
- Do not use blanks like "__".
- Do not create trick equations, ambiguous equations, or malformed equations.
- Do not use long GSM8K-style reasoning chains or benchmark-style wording.
- Do not include explanations.

Specs:
{specs_json}
""".strip()

    def _normalise_record(self, row: dict[str, Any], idx: int) -> dict[str, Any] | None:
        question = row.get("question")
        answer = row.get("answer")

        if isinstance(question, str) and isinstance(answer, str):
            question = self._clean_single_line(question)
            answer = self._clean_single_line(answer)
            if not question or not answer:
                return None

            row = dict(row)
            row["text"] = f"Question: {question}\nAnswer: {answer}"

        return super()._normalise_record(row=row, idx=idx)

    def _clean_single_line(self, value: str) -> str:
        value = self._clean_generated_text(value)
        value = " ".join(value.split())
        return value.strip()

    def _quality_ok(self, text: str, metadata: dict[str, Any]) -> bool:
        if not super()._quality_ok(text=text, metadata=metadata):
            return False

        if text.count("Question:") != 1:
            return False
        if text.count("Answer:") != 1:
            return False
        if "__" in text:
            return False

        lowered = text.lower()
        if "explanation:" in lowered:
            return False
        if "explain" in lowered:
            return False

        question = text.split("Question:", 1)[1].split("Answer:", 1)[0].strip()
        answer = text.split("Answer:", 1)[1].strip()

        if not question or not answer:
            return False
        if len(answer) > 40:
            return False

        return True

    def _structured_response_schema(self, batch_count: int | None = None) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["records"],
            "properties": {
                "records": {
                    "type": "array",
                    **({"minItems": batch_count, "maxItems": batch_count} if batch_count else {}),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "question",
                            "answer",
                            "format",
                            "difficulty",
                            "metadata",
                        ],
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "format": {"type": "string"},
                            "difficulty": {"type": "string"},
                            "metadata": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {},
                            },
                        },
                    },
                },
            },
        }
