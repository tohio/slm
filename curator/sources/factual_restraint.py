from __future__ import annotations

import json
import random
from pathlib import Path

from curator.scripts.groq_synthetic import GroqSyntheticSource


class FactualRestraintSource(GroqSyntheticSource):
    SOURCE_TAG = "factual_restraint"
    SHARD_PREFIX = "factual_restraint"
    DEFAULT_DOCS = 100_000
    DEFAULT_BATCH_SIZE = 16

    KINDS = [
        "unverifiable_private_fact",
        "no_fake_search_or_tool_claims",
        "unknown_or_insufficient_info",
        "source_dependent_answer",
        "date_currentness_uncertainty",
        "privacy_respecting_response",
        "avoid_invented_citations",
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
            "category": "factual_restraint",
            "kind": row.get("kind", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            "prompt_template": "factual_restraint_groq_v1",
            "benchmark_excluded": True,
        }

    def _build_prompt(self, batch_count: int, start_index: int, rng: random.Random) -> str:
        specs = []
        for offset in range(batch_count):
            specs.append({
                "id": start_index + offset,
                "kind": rng.choice(self.KINDS),
                "difficulty": rng.choice(["simple", "moderate", "subtle"]),
            })

        specs_json = json.dumps(specs, indent=2)
        return f"""
Generate unique factual-restraint pretraining records.

Rules:
- The assistant answer should avoid guessing.
- Do not invent citations, URLs, sources, current events, browsing, or tool-use claims.
- Respect privacy and say when information is unavailable or source-dependent.
- Do not include real private personal data.
- Vary names, contexts, industries, and wording.

Return JSON only, no markdown fences, using exactly this shape:
{{
  "records": [
    {{
      "text": "Question: ...\\nAnswer: ...",
      "kind": "unknown_or_insufficient_info",
      "difficulty": "simple",
      "metadata": {{}}
    }}
  ]
}}

Specs:
{specs_json}
""".strip()
