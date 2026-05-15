from __future__ import annotations

import json
import random
from pathlib import Path

from curator.scripts.groq_synthetic import GroqSyntheticSource


class SyntheticArithmeticSource(GroqSyntheticSource):
    SOURCE_TAG = "synthetic_arithmetic"
    SHARD_PREFIX = "synthetic_arithmetic"
    DEFAULT_DOCS = 100_000
    DEFAULT_BATCH_SIZE = 16

    FORMATS = [
        "bare_equation_full",
        "bare_equation_completion",
        "qa_arithmetic",
        "word_problem",
        "comparison_arithmetic",
        "multi_step_simple",
        "explain_short_solution",
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
            "prompt_template": "synthetic_arithmetic_groq_v1",
            "benchmark_excluded": True,
        }

    def _build_prompt(self, batch_count: int, start_index: int, rng: random.Random) -> str:
        specs = []
        for offset in range(batch_count):
            specs.append({
                "id": start_index + offset,
                "format": rng.choice(self.FORMATS),
                "difficulty": rng.choice(["single_step", "single_step", "two_step", "word_problem"]),
                "range": rng.choice(["0-20", "0-100", "mixed small integers", "multiplication table", "division exact"]),
            })

        specs_json = json.dumps(specs, indent=2)
        return f"""
Generate unique elementary arithmetic pretraining records.

Rules:
- Use original numbers and wording.
- Keep answers correct.
- Include the final numeric answer explicitly.
- Avoid GSM8K-style long grade-school benchmark problems.
- Vary equation forms, short QA, word problems, comparisons, and two-step examples.

Return JSON only, no markdown fences, using exactly this shape:
{{
  "records": [
    {{
      "text": "Question: ...\\nAnswer: ...",
      "format": "qa_arithmetic",
      "difficulty": "single_step",
      "metadata": {{}}
    }}
  ]
}}

Specs:
{specs_json}
""".strip()
