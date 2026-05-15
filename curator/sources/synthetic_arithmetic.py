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
    DEFAULT_BATCH_SIZE = 128

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
            "prompt_template": "synthetic_arithmetic_groq_v3",
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

Purpose:
- Reinforce concise arithmetic question-answer behavior.
- Keep records simple, correct, and easy to verify.

Critical JSON rules:
- Do not return a "text" field.
- Return separate "question" and "answer" fields.
- The "question" and "answer" values must be single-line JSON strings.
- Do not put newline characters inside any JSON string value.
- The Python adapter will build the final multiline training text.

Rules:
- Each record must contain exactly one arithmetic question and one short answer.
- Include the final answer explicitly in the "answer" field.
- Use original numbers and wording.
- Allowed:
  - single-step addition, subtraction, multiplication, division
  - simple word problems
  - simple comparisons
  - simple solve-for-x equations with one unknown
  - clear two-step arithmetic where the answer is still short
- Do not include multiple separate questions in one record.
- Do not use blanks like "__".
- Do not create trick equations, ambiguous equations, or malformed equations.
- Do not use long GSM8K-style reasoning chains or benchmark-style wording.
- Do not include explanations.
- Return JSON only. No markdown fences. No assistant chatter.

Return JSON using exactly this shape:
{{
  "records": [
    {{
      "question": "2 + 3 =",
      "answer": "5",
      "format": "qa_arithmetic",
      "difficulty": "single_step",
      "metadata": {{}}
    }}
  ]
}}

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
