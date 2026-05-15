from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

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
            "prompt_template": "factual_restraint_groq_v2",
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

Purpose:
- Teach the model not to hallucinate when facts are unavailable, private, current, or source-dependent.
- Keep answers concise and helpful, not over-refusal-heavy.

Balanced categories:
- live/current data unavailable
- private personal data
- confidential medical/legal/financial records
- unverifiable facts about private people
- exact proprietary/internal data
- unknown fictional or nonexistent identifiers
- no fake browsing/search/tool claims
- invented citation avoidance

Rules:
- Each record must contain exactly:
  Question:
  Answer:
- The answer should avoid guessing and should not invent facts.
- Do not claim to browse, search, verify, access tools, access databases, or check live systems.
- Do not include real private personal data.
- Do not refuse ordinary stable public facts.
- When useful, briefly say what would be needed to answer.
- Keep the answer concise; avoid long policy-style refusals.
- Return JSON only. No markdown fences. No assistant chatter.

Return JSON using exactly this shape:
{{
  "records": [
    {{
      "text": "Question: .\\nAnswer: .",
      "kind": "unknown_or_insufficient_info",
      "difficulty": "simple",
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

        answer = text.split("Answer:", 1)[1].strip()
        if len(answer) < 20:
            return False

        lowered_answer = answer.lower()
        fake_tool_claims = [
            "i searched",
            "i browsed",
            "i checked online",
            "according to my search",
            "i accessed",
            "i found a source",
            "the current value is",
        ]
        if any(fragment in lowered_answer for fragment in fake_tool_claims):
            return False

        return True
