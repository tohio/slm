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

    # Arithmetic records are intentionally short. Full 125m generation needs a
    # larger batch size than the generic Groq default to avoid excessive API calls.
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
            "prompt_template": "synthetic_arithmetic_groq_v2",
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

Rules:
- Each record must contain exactly one Question and exactly one Answer.
- Use this exact text structure:
  Question: <one arithmetic question>
  Answer: <short numeric answer or yes/no answer>
- Include the final answer explicitly.
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
- Do not include explanations unless the format explicitly asks for a very short one.
- Return JSON only. No markdown fences. No assistant chatter.

Return JSON using exactly this shape:
{{
  "records": [
    {{
      "text": "Question: .\\nAnswer: .",
      "format": "qa_arithmetic",
      "difficulty": "single_step",
      "metadata": {{}}
    }}
  ]
}}

Specs:
{specs_json}
""".strip()

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
        if "explain" in lowered and "explanation:" in lowered:
            return False

        return True