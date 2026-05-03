#!/usr/bin/env python
"""
Small behavior sanity eval for SLM checkpoints.

Examples:
  python eval/sanity_eval.py --model results/slm-125m-dpo/final
  python eval/sanity_eval.py --model tohio/slm-125m-chat --trust-remote-code
  python eval/sanity_eval.py --model results/slm-125m/final --prompts eval/sanity_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Result:
    id: str
    category: str
    passed: bool
    output: str
    failures: list[str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def too_repetitive_token_ids(tokens: list[int], max_repeat_run: int = 8) -> bool:
    if not tokens:
        return False

    run = 1
    for prev, cur in zip(tokens, tokens[1:]):
        if prev == cur:
            run += 1
            if run >= max_repeat_run:
                return True
        else:
            run = 1
    return False


def too_repetitive_text(text: str, max_word_repeat_run: int = 6) -> bool:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return False

    run = 1
    for prev, cur in zip(words, words[1:]):
        if prev == cur:
            run += 1
            if run >= max_word_repeat_run:
                return True
        else:
            run = 1
    return False


def resolve_tokenizer_path(model_id: str) -> str:
    path = Path(model_id)
    if path.exists():
        if (path / "tokenizer_config.json").exists():
            return str(path)
        if (path / "tokenizer" / "tokenizer_config.json").exists():
            return str(path / "tokenizer")
    return model_id


def load_model_and_tokenizer(model_id: str, trust_remote_code: bool, device: str):
    # Local repo custom class path is safer for local checkpoints if available.
    if Path(model_id).exists() and not trust_remote_code:
        try:
            from model.model import SLMForCausalLM

            model = SLMForCausalLM.from_pretrained(model_id)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )

    model = model.to(device)
    model.eval()

    tokenizer_path = resolve_tokenizer_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer


def generate(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    device: str,
    max_new_tokens: int,
    repetition_penalty: float,
) -> tuple[str, list[int]]:
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)

    endofturn_id = tokenizer.convert_tokens_to_ids("<|endofturn|>")
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(endofturn_id, int) and endofturn_id >= 0 and endofturn_id not in eos_ids:
        eos_ids.append(endofturn_id)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )

    input_len = inputs["input_ids"].shape[1]
    new_ids = output[0][input_len:].tolist()

    # Trim explicit stop tokens for checks.
    trimmed = list(new_ids)
    for stop_id in eos_ids:
        if stop_id in trimmed:
            trimmed = trimmed[: trimmed.index(stop_id)]

    decoded = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
    return decoded, trimmed


def evaluate_case(case: dict[str, Any], output: str, token_ids: list[int]) -> Result:
    failures = []

    norm_output = normalize(output)

    if not output.strip():
        failures.append("empty output")

    for needle in case.get("must_contain", []):
        if not contains(output, needle):
            failures.append(f"missing required text: {needle!r}")

    must_contain_any = case.get("must_contain_any", [])
    if must_contain_any and not any(contains(output, item) for item in must_contain_any):
        failures.append(f"missing one of required alternatives: {must_contain_any!r}")

    for needle in case.get("must_not_contain", []):
        if contains(output, needle):
            failures.append(f"contains forbidden text: {needle!r}")

    if too_repetitive_token_ids(token_ids):
        failures.append("repeated token run detected")

    if too_repetitive_text(output):
        failures.append("repeated word run detected")

    # Basic continuation/hallucination guard for short factual questions.
    category = case.get("category", "")
    if category in {"simple_factual", "stop_behavior"}:
        word_count = len(norm_output.split())
        if word_count > 35:
            failures.append(f"too long for {category}: {word_count} words")

    return Result(
        id=case["id"],
        category=category,
        passed=not failures,
        output=output,
        failures=failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SLM sanity behavior eval.")
    parser.add_argument("--model", required=True, help="Local checkpoint path or Hub model id")
    parser.add_argument("--prompts", default="eval/sanity_prompts.jsonl", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    cases = load_jsonl(args.prompts)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
    )

    results: list[Result] = []
    for case in cases:
        output, token_ids = generate(
            model=model,
            tokenizer=tokenizer,
            messages=case["messages"],
            device=args.device,
            max_new_tokens=int(case.get("max_new_tokens", 80)),
            repetition_penalty=args.repetition_penalty,
        )
        result = evaluate_case(case, output, token_ids)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {result.id} ({result.category})")
        print(output)
        if result.failures:
            for failure in result.failures:
                print(f"  - {failure}")

    passed = sum(r.passed for r in results)
    total = len(results)
    print("\n" + "=" * 80)
    print(f"Sanity eval: {passed}/{total} passed")

    by_category: dict[str, list[Result]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)

    for category, items in sorted(by_category.items()):
        cat_passed = sum(r.passed for r in items)
        print(f"{category}: {cat_passed}/{len(items)}")

    if args.json_out:
        payload = {
            "model": args.model,
            "passed": passed,
            "total": total,
            "results": [
                {
                    "id": r.id,
                    "category": r.category,
                    "passed": r.passed,
                    "output": r.output,
                    "failures": r.failures,
                }
                for r in results
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())