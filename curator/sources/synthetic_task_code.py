from __future__ import annotations

import json
import random
from pathlib import Path

from curator.scripts.groq_synthetic import GroqSyntheticSource


class SyntheticTaskCodeSource(GroqSyntheticSource):
    SOURCE_TAG = "synthetic_task_code"
    SHARD_PREFIX = "synthetic_task_code"
    DEFAULT_DOCS = 100_000
    DEFAULT_BATCH_SIZE = 8

    LANGUAGES = ["python", "python", "python", "python", "go", "rust", "bash"]
    TASK_TYPES = [
        "docstring_to_code",
        "signature_to_function",
        "prompt_to_function",
        "function_with_tests",
        "bug_fix_small",
        "code_explanation_to_code",
        "edge_case_implementation",
        "parser_or_formatter",
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
            "category": "task_code",
            "language": row.get("language", "unknown"),
            "task_type": row.get("task_type", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            "prompt_template": "synthetic_task_code_groq_v1",
            "benchmark_excluded": True,
        }

    def _build_prompt(self, batch_count: int, start_index: int, rng: random.Random) -> str:
        specs = []
        for offset in range(batch_count):
            specs.append({
                "id": start_index + offset,
                "language": rng.choice(self.LANGUAGES),
                "task_type": rng.choice(self.TASK_TYPES),
                "difficulty": rng.choice(["easy", "medium", "medium", "hard"]),
                "topic": rng.choice([
                    "strings", "lists", "maps", "sets", "files", "validation",
                    "parsing", "sorting", "aggregation", "date/time", "CLI input",
                    "error handling", "unit conversion", "small algorithms",
                ]),
            })

        specs_json = json.dumps(specs, indent=2)
        return f"""
Generate unique synthetic coding pretraining records.

Rules:
- Do not use HumanEval, APPS, LeetCode, CodeContests, interview benchmark tasks, or benchmark-derived wording.
- Make every example original and suitable for a small language model.
- Each record text must include a natural language task and a correct solution.
- Prefer complete functions, small tests, edge cases, and clear formatting.
- Vary names, domains, APIs, and wording so exact dedup does not collapse the batch.

Return JSON only, no markdown fences, using exactly this shape:
{{
  "records": [
    {{
      "text": "Task: ...\\n\\nSolution:\\n...",
      "language": "python",
      "task_type": "prompt_to_function",
      "difficulty": "medium",
      "metadata": {{}}
    }}
  ]
}}

Specs:
{specs_json}
""".strip()
