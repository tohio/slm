from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any

from curator.scripts.groq_synthetic import GroqSyntheticSource


class SyntheticTaskCodeSource(GroqSyntheticSource):
    SOURCE_TAG = "synthetic_task_code"
    SHARD_PREFIX = "synthetic_task_code"
    DEFAULT_DOCS = 100_000
    DEFAULT_BATCH_SIZE = 8

    TASK_TYPES = [
        "prompt_to_function",
        "docstring_to_function",
        "function_with_tests",
        "parser_or_formatter",
        "aggregation_utility",
        "collection_transform",
    ]

    TOPICS = [
        "strings",
        "lists",
        "dictionaries",
        "sets",
        "simple parsing",
        "sorting",
        "aggregation",
        "date formatting",
        "unit conversion",
        "numeric utilities",
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
            "language": "python",
            "task_type": row.get("task_type", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            "topic": row.get("topic", "unknown"),
            "prompt_template": "synthetic_task_code_groq_v3",
            "benchmark_excluded": True,
        }

    def _build_prompt(self, batch_count: int, start_index: int, rng: random.Random) -> str:
        specs = []
        for offset in range(batch_count):
            specs.append({
                "id": start_index + offset,
                "language": "python",
                "task_type": rng.choice(self.TASK_TYPES),
                "difficulty": rng.choice(["easy", "easy", "medium"]),
                "topic": rng.choice(self.TOPICS),
            })

        specs_json = json.dumps(specs, indent=2)
        return f"""
Generate unique Python-only synthetic coding pretraining records.

Purpose:
- Create clean task -> implementation -> tests examples for a small language model.
- Prefer boring, correct, simple utility functions over clever or production-hard tasks.

Critical JSON rules:
- Do not return a "text" field.
- Return a single-line "task" field.
- Return a "solution_lines" field as a JSON array of single-line strings.
- Do not put newline characters inside any JSON string value.
- The Python adapter will join solution_lines with newlines and build final training text.

Allowed task families:
- list transformations
- dictionary transformations
- string formatting and simple parsing
- set operations
- counting and aggregation
- simple math or unit conversion
- simple date parsing/formatting with fixed examples
- in-memory JSON/CSV-like parsing without file or network I/O

Hard rules:
- Python only.
- Use only the Python standard library.
- Every solution must define at least one function.
- Every solution must include at least one assert test in solution_lines.
- Every solution must be syntactically valid Python when solution_lines are joined with newlines.
- Do not use pass, TODO, ellipses, placeholders, or comment-only solutions.
- Do not generate bug-fix tasks.
- Do not generate security/auth/password/encryption/sanitization tasks.
- Do not generate email validation, Unicode/grapheme, concurrency, threads, async, networking,
  filesystem, subprocess, database, external API, CLI-input, or production-hardening tasks.
- Do not use HumanEval, APPS, LeetCode, CodeContests, interview benchmark tasks,
  benchmark-derived wording, or named benchmark problems.
- Return JSON only. No markdown fences. No assistant chatter.

Return JSON using exactly this shape:
{{
  "records": [
    {{
      "task": "Write a Python function that returns the even numbers from a list while preserving order.",
      "solution_lines": [
        "def extract_evens(numbers):",
        "    return [n for n in numbers if n % 2 == 0]",
        "",
        "# Tests",
        "assert extract_evens([1, 2, 3, 4]) == [2, 4]",
        "assert extract_evens([1, 3, 5]) == []"
      ],
      "language": "python",
      "task_type": "collection_transform",
      "difficulty": "easy",
      "topic": "lists",
      "metadata": {{}}
    }}
  ]
}}

Specs:
{specs_json}
""".strip()

    def _normalise_record(self, row: dict[str, Any], idx: int) -> dict[str, Any] | None:
        task = row.get("task")
        solution_lines = row.get("solution_lines")
        if solution_lines is None:
            solution_lines = row.get("solution")

        if isinstance(task, str):
            task = self._clean_single_line(task)
        else:
            task = ""

        solution = self._normalise_solution(solution_lines)

        if task and solution:
            row = dict(row)
            row["text"] = f"Task: {task}\n\nSolution:\n{solution}"
            row["language"] = "python"

        return super()._normalise_record(row=row, idx=idx)

    def _clean_single_line(self, value: str) -> str:
        value = self._clean_generated_text(value)
        value = " ".join(value.split())
        return value.strip()

    def _normalise_solution(self, value: Any) -> str:
        if isinstance(value, list):
            lines = []
            for item in value:
                if not isinstance(item, str):
                    return ""
                line = self._clean_generated_text(item)
                line = line.replace("\r", "").replace("\n", " ")
                lines.append(line.rstrip())
            return "\n".join(lines).strip()

        if isinstance(value, str):
            value = self._clean_generated_text(value)
            return value.strip()

        return ""

    def _quality_ok(self, text: str, metadata: dict[str, Any]) -> bool:
        if not super()._quality_ok(text=text, metadata=metadata):
            return False

        if "Task:" not in text or "Solution:" not in text:
            return False

        if text.count("Task:") != 1 or text.count("Solution:") != 1:
            return False

        language = str(metadata.get("language", "")).lower()
        if language != "python":
            return False

        lowered = text.lower()
        forbidden = [
            "fix the",
            "fix this",
            "bug",
            "off-by-one",
            "race condition",
            "unicode",
            "combining character",
            "email validation",
            "validate email",
            "security",
            "authentication",
            "password",
            "cryptographic",
            "encryption",
            "sanitize",
            "thread",
            "async",
            "network",
            "socket",
            "database",
            "subprocess",
            "file io",
            "read a file",
            "write a file",
            "cli input",
        ]
        if any(fragment in lowered for fragment in forbidden):
            return False

        solution = text.split("Solution:", 1)[1].strip()
        if not solution:
            return False

        placeholder_fragments = [
            "todo",
            "pass",
            "not implemented",
            "your code here",
            "placeholder",
            "...",
        ]
        lowered_solution = solution.lower()
        if any(fragment in lowered_solution for fragment in placeholder_fragments):
            return False

        meaningful_lines = [
            line for line in solution.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not meaningful_lines:
            return False

        try:
            parsed = ast.parse(solution)
        except SyntaxError:
            return False

        if not parsed.body:
            return False

        has_function = any(isinstance(node, ast.FunctionDef) for node in parsed.body)
        has_assert = any(isinstance(node, ast.Assert) for node in ast.walk(parsed))
        if not has_function or not has_assert:
            return False

        return True
