#!/usr/bin/env python3
"""
Prepare raw code-completion SFT data.

This stage converts existing code SFT conversation data into raw causal
completion examples:

    prompt:
        imports
        def foo(...):
            """docstring"""

    completion:
            indented function body only

The output is used by finetune/train_code_completion.py, which masks prompt
tokens and trains only on completion/body tokens.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


BAD_COMPLETION_STARTS = (
    "def ",
    "class ",
    "import ",
    "from ",
    "#",
    "print(",
    "```",
    "if __name__",
)

BODY_SIGNAL_RE = re.compile(
    r"\b(return|for|if|while|with|try|raise|yield|append|extend|result|total|count|sum|len)\b"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_code_fences(text: str) -> str:
    text = text.strip()
    blocks = re.findall(
        r"```(?:[a-zA-Z0-9_+\-.#]+)?\s*\n(.*?)```",
        text,
        flags=re.DOTALL,
    )
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        return max(blocks, key=len)
    return text


def get_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages") or record.get("conversations") or []
    if not isinstance(messages, list):
        return []

    normalized: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value") or ""

        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        if role and content:
            normalized.append({"role": str(role), "content": str(content)})

    return normalized


def first_user_text(record: dict[str, Any]) -> str:
    for msg in get_messages(record):
        if msg["role"] == "user":
            return msg["content"].strip()
    return ""


def last_assistant_text(record: dict[str, Any]) -> str:
    for msg in reversed(get_messages(record)):
        if msg["role"] == "assistant":
            return msg["content"].strip()
    return ""


def sanitize_docstring(text: str, max_chars: int = 700) -> str:
    text = text.strip()
    text = text.replace('"""', chr(39) * 3)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()

    return text or "Complete the function."


def collect_imports(tree: ast.Module, lines: list[str], before_lineno: int) -> str:
    imports: list[str] = []

    for node in tree.body:
        if getattr(node, "lineno", 10**9) >= before_lineno:
            break

        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.end_lineno is not None:
            imports.extend(lines[node.lineno - 1 : node.end_lineno])

    return "\n".join(imports).strip()


def get_signature_lines(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> list[str] | None:
    start = node.lineno - 1
    max_scan = min(len(lines), start + 12)
    collected: list[str] = []
    paren_balance = 0

    for i in range(start, max_scan):
        line = lines[i]
        collected.append(line)
        paren_balance += line.count("(") - line.count(")")

        if line.rstrip().endswith(":") and paren_balance <= 0:
            break

    if not collected:
        return None

    first = collected[0].lstrip()
    if not first.startswith(("def ", "async def ")):
        return None

    if not collected[-1].rstrip().endswith(":"):
        return None

    return collected


def is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(getattr(stmt, "value", None), ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def body_completion_after_prompt(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> tuple[list[str], str] | None:
    """Return function prompt lines and body completion."""
    if not node.body or node.end_lineno is None:
        return None

    first_stmt = node.body[0]
    has_docstring = is_docstring_stmt(first_stmt)

    if has_docstring:
        prompt_end_lineno = first_stmt.end_lineno or first_stmt.lineno
        body_start_lineno = prompt_end_lineno + 1
        prompt_lines = lines[node.lineno - 1 : prompt_end_lineno]
    else:
        signature_lines = get_signature_lines(node, lines)
        if signature_lines is None:
            return None

        indent = signature_lines[0][: len(signature_lines[0]) - len(signature_lines[0].lstrip())]
        body_indent = indent + "    "
        prompt_lines = signature_lines + [
            body_indent + '"""',
            body_indent + "__DOCSTRING_PLACEHOLDER__",
            body_indent + '"""',
        ]
        body_start_lineno = first_stmt.lineno

    if body_start_lineno > node.end_lineno:
        return None

    body_lines = lines[body_start_lineno - 1 : node.end_lineno]
    completion = "\n".join(body_lines).rstrip() + "\n"

    return prompt_lines, completion


def valid_completion(completion: str) -> bool:
    if not completion.strip():
        return False

    stripped = completion.lstrip()

    if stripped.startswith(BAD_COMPLETION_STARTS):
        return False

    if "The answer is (" in completion or "The correct answer is" in completion:
        return False

    first_nonempty = next((line for line in completion.splitlines() if line.strip()), "")
    if not first_nonempty.startswith(("    ", "\t")):
        return False

    if not BODY_SIGNAL_RE.search(completion):
        return False

    return True


def build_prompt(
    imports: str,
    prompt_lines: list[str],
    fallback_docstring: str,
) -> str:
    rendered: list[str] = []
    fallback = sanitize_docstring(fallback_docstring)

    for line in prompt_lines:
        if "__DOCSTRING_PLACEHOLDER__" not in line:
            rendered.append(line)
            continue

        indent = line[: len(line) - len(line.lstrip())]
        rendered.extend(indent + part for part in fallback.splitlines())

    body = "\n".join(rendered).rstrip() + "\n"

    if imports:
        return imports + "\n\n" + body
    return body


def extract_records_from_row(record: dict[str, Any]) -> list[dict[str, Any]]:
    user_text = first_user_text(record)
    assistant_text = last_assistant_text(record)

    if not user_text or not assistant_text:
        return []

    code = strip_code_fences(assistant_text)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    lines = code.splitlines()
    records: list[dict[str, Any]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        if node.decorator_list:
            continue

        if node.end_lineno is None:
            continue

        if node.end_lineno - node.lineno > 120:
            continue

        pair = body_completion_after_prompt(node, lines)
        if pair is None:
            continue

        prompt_lines, completion = pair

        if not valid_completion(completion):
            continue

        imports = collect_imports(tree, lines, before_lineno=node.lineno)
        prompt = build_prompt(imports, prompt_lines, user_text)

        records.append(
            {
                "prompt": prompt,
                "completion": completion,
                "name": node.name,
                "source": record.get("source", "unknown"),
                "sft_type": record.get("sft_type", "unknown"),
            }
        )

    return records


def extract_records(rows: list[dict[str, Any]], max_records: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        for rec in extract_records_from_row(row):
            key = rec["prompt"] + "\n---\n" + rec["completion"]
            if key in seen:
                continue
            seen.add(key)
            out.append(rec)

    random.Random(seed).shuffle(out)

    if max_records and len(out) > max_records:
        out = out[:max_records]

    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "by_source": dict(Counter(row.get("source", "unknown") for row in rows)),
        "by_sft_type": dict(Counter(row.get("sft_type", "unknown") for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw code-completion SFT data")
    parser.add_argument("--size", default="125m")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-train-records", type=int, default=20000)
    parser.add_argument("--max-val-records", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir or f"data/runs/{args.size}/sft_code")
    output_dir = Path(args.output_dir or f"data/runs/{args.size}/code_completion")

    train_rows = read_jsonl(input_dir / "train.jsonl")
    val_rows = read_jsonl(input_dir / "val.jsonl")

    train_records = extract_records(train_rows, args.max_train_records, args.seed)
    val_records = extract_records(val_rows, args.max_val_records, args.seed + 1)

    # If val extraction is too small, split a small validation set from train.
    if len(val_records) < 100 and len(train_records) > 1000:
        n_val = min(max(100, len(train_records) // 50), 1000)
        val_records = train_records[:n_val]
        train_records = train_records[n_val:]

    if len(train_records) < 100:
        raise SystemExit(f"Too few train records extracted: {len(train_records)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)

    stats = {
        "size": args.size,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train": summarize(train_records),
        "val": summarize(val_records),
        "format": {
            "prompt": "raw function signature/docstring prefix",
            "completion": "indented function body only",
        },
    }

    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"Raw code-completion train: {len(train_records):,}")
    print(f"Raw code-completion val:   {len(val_records):,}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
