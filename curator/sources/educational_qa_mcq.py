from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

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
        "stable_geography",
        "vocabulary_context",
        "reading_reasoning",
        "common_misconception",
    ]

    QA_TYPES = [
        "short_qa",
        "multiple_choice_qa",
        "qa_with_explanation",
        "cloze_completion",
        "common_misconception",
        "cause_effect_reasoning",
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
            "prompt_template": "educational_qa_mcq_groq_v4_jsonl",
            "benchmark_excluded": True,
        }

    def _build_prompt(self, batch_count: int, start_index: int, rng: random.Random) -> str:
        specs = []
        for offset in range(batch_count):
            specs.append({
                "id": start_index + offset,
                "subject": rng.choice(self.SUBJECTS),
                "qa_type": rng.choice(self.QA_TYPES),
                "difficulty": rng.choice(["elementary", "middle", "high_school"]),
            })

        specs_json = json.dumps(specs, indent=2)
        return f"""
Generate unique educational QA pretraining records.

Purpose:
- Teach stable educational facts, concepts, and reasoning.
- Prefer clear explanations over trivia.

Critical output rules:
- Return JSONL only.
- Return exactly one compact JSON object per line.
- Do not return an outer object.
- Do not return a "records" array.
- Do not return markdown fences.
- Do not return assistant chatter.
- Do not return a "text" field.
- Each JSON object must have separate "question", "answer", and "explanation" fields.
- Each field value must be a single-line JSON string.
- Do not put newline characters inside any JSON string value.
- The Python adapter will build the final multiline training text.

Allowed subjects:
- arithmetic and math concepts
- basic science
- introductory computer science concepts
- stable geography only when facts are non-controversial
- vocabulary/context reasoning
- common misconceptions with clear corrections
- cause/effect reasoning

Rules:
- Each record must contain one question, one answer, and one explanation.
- The explanation must teach the reasoning, not merely restate the answer.
- Use stable, non-current, non-controversial facts.
- Avoid trivial life-routine filler.
- Avoid current events, live data, prices, laws, recent statistics, politics, and medical advice.
- Avoid disputed rankings, measurement-dependent claims, or facts that require a caveat.
- Avoid private people or made-up claims about real people.
- Do not copy or paraphrase MMLU, ARC, HellaSwag, TruthfulQA, GSM8K, exam collections,
  or other benchmark datasets.

Each output line must look like this:
{{"question":"Why do plants need sunlight?","answer":"Plants use sunlight as an energy source for photosynthesis.","explanation":"During photosynthesis, plants use light energy to turn carbon dioxide and water into sugars they can use for growth.","subject":"science","qa_type":"short_qa","difficulty":"middle","metadata":{{}}}}

Specs:
{specs_json}
""".strip()

    def _normalise_record(self, row: dict[str, Any], idx: int) -> dict[str, Any] | None:
        question = row.get("question")
        answer = row.get("answer")
        explanation = row.get("explanation")

        if isinstance(question, str) and isinstance(answer, str) and isinstance(explanation, str):
            question = self._clean_single_line(question)
            answer = self._clean_single_line(answer)
            explanation = self._clean_single_line(explanation)
            if not question or not answer or not explanation:
                return None

            row = dict(row)
            row["text"] = (
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Explanation: {explanation}"
            )

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
        if text.count("Explanation:") != 1:
            return False

        explanation = text.split("Explanation:", 1)[1].strip()
        if len(explanation) < 40:
            return False

        lowered = text.lower()
        forbidden = [
            "right now",
            "today",
            "latest",
            "current",
            "stock price",
            "weather",
            "first thing you should do when you wake up",
            "get out of bed",
            "which river is longer, the nile or the amazon",
            "longest river",
        ]
        if any(fragment in lowered for fragment in forbidden):
            return False

        return True