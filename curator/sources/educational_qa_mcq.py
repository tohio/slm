from __future__ import annotations

import json
import random
from pathlib import Path

from curator.scripts.groq_synthetic import GroqSyntheticSource


class EducationalQAMCQSource(GroqSyntheticSource):
    SOURCE_TAG = "educational_qa_mcq"
    SHARD_PREFIX = "educational_qa_mcq"
    DEFAULT_DOCS = 100_000
    DEFAULT_BATCH_SIZE = 12

    SUBJECTS = [
        "science",
        "math_concepts",
        "computer_science",
        "history_geography",
        "general_knowledge",
        "reading_common_sense",
    ]
    QA_TYPES = [
        "short_qa",
        "multiple_choice_qa",
        "qa_with_explanation",
        "cloze_completion",
        "compare_and_contrast",
        "common_misconception",
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
            "category": "educational_qa",
            "subject": row.get("subject", "unknown"),
            "qa_type": row.get("qa_type", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            "prompt_template": "educational_qa_mcq_groq_v1",
            "benchmark_excluded": True,
        }

    def _build_prompt(self, batch_count: int, start_index: int, rng: random.Random) -> str:
        specs = []
        for offset in range(batch_count):
            specs.append({
                "id": start_index + offset,
                "subject": rng.choice(self.SUBJECTS),
                "qa_type": rng.choice(self.QA_TYPES),
                "difficulty": rng.choice(["elementary", "middle", "high_school", "intro_college"]),
            })

        specs_json = json.dumps(specs, indent=2)
        return f"""
Generate unique educational pretraining records.

Rules:
- Do not copy or paraphrase benchmark datasets such as MMLU, ARC, HellaSwag, TruthfulQA, GSM8K, HumanEval, APPS, or exam collections.
- Use original questions, answers, and explanations.
- Include the correct answer clearly.
- Vary wording, subjects, entities, and format heavily.

Return JSON only, no markdown fences, using exactly this shape:
{{
  "records": [
    {{
      "text": "Question: ...\\nAnswer: ...\\nExplanation: ...",
      "subject": "science",
      "qa_type": "short_qa",
      "difficulty": "middle",
      "metadata": {{}}
    }}
  ]
}}

Specs:
{specs_json}
""".strip()
