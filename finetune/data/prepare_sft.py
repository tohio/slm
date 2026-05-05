"""
Download and format SFT datasets for chat and code fine-tuning.

Formats all data into the SLM conversation format:
    [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ]

Chat template application:
    No "text" field is produced. train_sft.py renames "conversations" →
    "messages" at load time; trl's SFTTrainer then auto-detects the
    conversational format and applies tokenizer.apply_chat_template()
    internally. Pre-formatting here would bypass the tokenizer's baked-in
    chat template (including {% generation %} tags) and break
    assistant_only_loss=True.

Stage 1 — Chat SFT:
    Dataset: teknium/OpenHermes-2.5
    Size:    ~1M examples
    Output:  data/sft/chat/train.jsonl + val.jsonl

Stage 2 — Code SFT:
    Dataset: ise-uiuc/Magicoder-OSS-Instruct-75K
    Size:    ~75k examples before filtering
    Output:  data/sft/code/train.jsonl + val.jsonl

    Code SFT keeps examples whose assistant responses contain actual code.
    Obvious prose-only / explanation-only examples are dropped so this stage
    teaches the model to emit code when code is requested.

Output format — one conversation per line:
    {
        "conversations": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "source": "openhermes",
        "sft_type": "general_assistant"
    }

Usage:
    python finetune/data/prepare_sft.py --stage both
    python finetune/data/prepare_sft.py --stage chat
    python finetune/data/prepare_sft.py --stage code
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.data.response_control import build_response_control_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
SFT_DIR  = DATA_DIR / "sft"

# Default system prompts
DEFAULT_SYSTEM = "You are a helpful, harmless, and honest assistant."
CODE_SYSTEM = (
    "You are an expert programming assistant. "
    "When code is requested, write code directly and avoid unnecessary explanation. "
    "When explanation is requested, explain clearly in prose and do not rewrite the code."
)

# Handcrafted explanation examples are intentionally oversampled because the
# Magicoder code stage is dominated by code-output tasks. A small repeated
# prose-only explanation signal helps preserve task-mode distinction:
#   explain code -> explain in prose
#   write code   -> write code
HANDCRAFTED_CODE_EXPLANATION_REPEAT = 20


# Generated response-control chat examples are built in finetune/data/response_control.py.


def build_handcrafted_response_control_records() -> list[dict]:
    """Return generated response-control chat examples.

    Kept under the old function name so prepare_chat() does not need broader
    wiring changes. The source field is "response_control".
    """
    return build_response_control_records(
        system=DEFAULT_SYSTEM,
        max_examples=5000,
    )


# ── Handcrafted code-explanation examples ─────────────────────────────────────

HANDCRAFTED_CODE_EXPLANATIONS = [
    {
        "prompt": "Explain what this Python function does:\n\ndef square(x):\n    return x * x",
        "answer": "It returns the square of x by multiplying x by itself.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef is_even(n):\n    return n % 2 == 0",
        "answer": "It returns True when n is evenly divisible by 2, otherwise it returns False.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef first_item(items):\n    return items[0]",
        "answer": "It returns the first element from the items list.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef count_positive(numbers):\n    return sum(1 for n in numbers if n > 0)",
        "answer": "It counts how many numbers in the input are greater than zero.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef reverse_string(text):\n    return text[::-1]",
        "answer": "It returns a new string with the characters of text in reverse order.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef safe_get(mapping, key, default=None):\n    return mapping.get(key, default)",
        "answer": "It looks up key in the mapping and returns default if the key is missing.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef remove_duplicates(items):\n    return list(dict.fromkeys(items))",
        "answer": "It removes duplicate values while preserving the original order of the items.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef clamp(value, low, high):\n    return max(low, min(value, high))",
        "answer": "It restricts value to stay between low and high.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False",
        "answer": "It checks every pair of numbers and returns True if any pair is closer than the threshold.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef trailing_zeroes_in_factorial(num):\n    count = 0\n    while num >= 5:\n        num //= 5\n        count += num\n    return count",
        "answer": "It counts how many trailing zeroes appear in num factorial by counting factors of 5.",
    },
    {
        "prompt": "Explain this JavaScript function:\n\nfunction add(a, b) {\n  return a + b;\n}",
        "answer": "It returns the sum of a and b.",
    },
    {
        "prompt": "Explain this JavaScript function:\n\nfunction isEmpty(arr) {\n  return arr.length === 0;\n}",
        "answer": "It returns true when the array has no elements.",
    },
    {
        "prompt": "Explain this JavaScript function:\n\nfunction getName(user) {\n  return user.name;\n}",
        "answer": "It returns the name property from the user object.",
    },
    {
        "prompt": "Explain this TypeScript reducer case:\n\ncase 'CUSTOMER_CLEAR_INFO':\n  return { ...state, info: {} };",
        "answer": "It returns a new state object with the info field reset to an empty object.",
    },
    {
        "prompt": "Explain this Rust function:\n\nfn double(x: i32) -> i32 {\n    x * 2\n}",
        "answer": "It returns x multiplied by 2.",
    },
    {
        "prompt": "Explain this Rust expression:\n\nnumbers.iter().filter(|n| **n > 0).count()",
        "answer": "It iterates over numbers, keeps only positive values, and counts how many there are.",
    },
    {
        "prompt": "Explain this SQL query:\n\nSELECT COUNT(*) FROM users WHERE active = true;",
        "answer": "It counts the number of active users.",
    },
    {
        "prompt": "Explain this Bash command:\n\ncp source.txt backup.txt",
        "answer": "It copies source.txt to a new file named backup.txt.",
    },
    {
        "prompt": "Explain this Python class method:\n\ndef available_seats(self):\n    return self.capacity - self.reserved",
        "answer": "It returns the number of seats that are still available by subtracting reserved seats from capacity.",
    },
    {
        "prompt": "Explain what this Python code does:\n\nresult = [x * x for x in numbers]",
        "answer": "It creates a new list containing the square of each value in numbers.",
    },
    {
        "prompt": "Explain what this Python code does:\n\nwith open(path) as f:\n    lines = f.readlines()",
        "answer": "It opens the file at path and reads all lines into a list.",
    },
    {
        "prompt": "Explain this Python conditional:\n\nif not items:\n    return []",
        "answer": "It checks whether items is empty and returns an empty list when it is.",
    },
    {
        "prompt": "Explain this Python loop:\n\nfor item in items:\n    print(item)",
        "answer": "It iterates through each item in items and prints it.",
    },
    {
        "prompt": "Explain this Python dictionary update:\n\ncounts[word] = counts.get(word, 0) + 1",
        "answer": "It increments the count for word, starting from 0 if the word is not already present.",
    },
    {
        "prompt": "Explain this Python statement:\n\nreturn sorted(items, key=lambda item: item.name)",
        "answer": "It returns items sorted by each item's name attribute.",
    },
    {
        "prompt": "Explain this C# property:\n\npublic string Name { get; set; }",
        "answer": "It declares a public string property named Name with automatic getter and setter methods.",
    },
    {
        "prompt": "Explain this C# condition:\n\nif (items.Count == 0) return false;",
        "answer": "It returns false when the items collection is empty.",
    },
    {
        "prompt": "Explain this Java method:\n\npublic int size() {\n    return items.size();\n}",
        "answer": "It returns the number of elements in the items collection.",
    },
    {
        "prompt": "Explain this JavaScript arrow function:\n\nconst double = x => x * 2;",
        "answer": "It defines a function that returns its input multiplied by 2.",
    },
    {
        "prompt": "Explain this Python exception handler:\n\ntry:\n    value = int(text)\nexcept ValueError:\n    value = 0",
        "answer": "It tries to convert text to an integer and falls back to 0 if conversion fails.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef merge_dicts(a, b):\n    return {**a, **b}",
        "answer": "It returns a new dictionary containing keys from a and b, with b overriding duplicate keys.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef get_extension(filename):\n    return filename.rsplit('.', 1)[-1]",
        "answer": "It returns the part of filename after the final dot.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9",
        "answer": "It converts a Fahrenheit temperature to Celsius.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef non_empty(strings):\n    return [s for s in strings if s]",
        "answer": "It returns only the strings that are not empty.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef word_count(text):\n    return len(text.split())",
        "answer": "It returns the number of whitespace-separated words in text.",
    },
]


def build_handcrafted_code_explanation_records() -> list[dict]:
    """Return small prose-only examples for code-explanation behavior."""
    records = []

    for example in HANDCRAFTED_CODE_EXPLANATIONS:
        records.append({
            "conversations": [
                {"role": "system", "content": CODE_SYSTEM},
                {"role": "user", "content": example["prompt"].strip()},
                {"role": "assistant", "content": example["answer"].strip()},
            ],
            "source": "handcrafted_code_explanation",
            "sft_type": "code_explanation",
            "normalized": False,
        })

    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records):,} records to {path}")


# ── Chat SFT — OpenHermes-2.5 ─────────────────────────────────────────────────

# Per-stage defaults for validation fraction. These are the sources of truth;
# CLI --val-fraction overrides them only when explicitly passed.
DEFAULT_VAL_FRACTION = {
    "chat": 0.02,
    "code": 0.05,
}


def prepare_chat(val_fraction: float) -> None:
    """
    Download and format OpenHermes-2.5 for chat SFT.

    OpenHermes-2.5 is a high-quality synthetic instruction dataset
    generated by GPT-4. Contains diverse instruction-following examples
    across coding, reasoning, creative writing, and general knowledge.

    Format: each example has a "conversations" field with role/value pairs.
    """
    from datasets import load_dataset

    out_dir    = SFT_DIR / "chat"
    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    if train_path.exists() and val_path.exists():
        log.info(f"Chat SFT data already exists at {out_dir}")
        return

    log.info("Loading OpenHermes-2.5...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    log.info(f"OpenHermes-2.5: {len(dataset):,} examples")

    records = []
    skipped = 0

    for example in dataset:
        conversations = example.get("conversations") or []
        if not conversations:
            skipped += 1
            continue

        messages = []

        # Add system prompt — use `or ""` to handle None values
        system = (example.get("system_prompt") or "").strip()
        messages.append({
            "role": "system",
            "content": system if system else DEFAULT_SYSTEM,
        })

        valid = True
        for turn in conversations:
            # Use `or ""` to handle None values in role and content fields
            role    = (turn.get("from") or turn.get("role") or "").lower()
            content = (turn.get("value") or turn.get("content") or "").strip()

            if not content:
                valid = False
                break

            if role in ("human", "user"):
                messages.append({"role": "user", "content": content})
            elif role in ("gpt", "assistant"):
                messages.append({"role": "assistant", "content": content})

        if not valid or len(messages) < 2:
            skipped += 1
            continue

        # Must end with assistant turn
        if messages[-1]["role"] != "assistant":
            skipped += 1
            continue

        records.append({
            "conversations": messages,
            "source": "openhermes",
            "sft_type": "general_assistant",
        })

    handcrafted_records = build_handcrafted_response_control_records()
    records.extend(handcrafted_records)
    log.info(f"Added generated response-control chat examples: {len(handcrafted_records):,}")
    log.info(f"Processed: {len(records):,} kept, {skipped:,} skipped")

    # Split
    n_val = max(1000, int(len(records) * val_fraction))
    random.seed(42)
    random.shuffle(records)
    val_records   = records[:n_val]
    train_records = records[n_val:]

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    log.info(f"Chat SFT: {len(train_records):,} train, {len(val_records):,} val")


# ── Code SFT — Magicoder-OSS-Instruct ─────────────────────────────────────────

def prepare_code(val_fraction: float) -> None:
    """
    Download and format Magicoder-OSS-Instruct-75K for code SFT.

    Magicoder generates coding problems inspired by real open-source code,
    then generates solutions. Each example is a single-turn instruction/response
    pair.

    This stage keeps only examples whose assistant response contains real code.
    Obvious prose-only / explanation-only examples are dropped. For code-output
    task types, fenced code is normalized to code-only targets so code SFT
    teaches the model to produce code directly when code is requested.
    """
    from datasets import load_dataset

    out_dir    = SFT_DIR / "code"
    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    if train_path.exists() and val_path.exists():
        log.info(f"Code SFT data already exists at {out_dir}")
        return

    log.info("Loading Magicoder-OSS-Instruct-75K...")
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    log.info(f"Magicoder: {len(dataset):,} examples")

    records = []
    skipped_reasons = Counter()
    type_counts = Counter()
    normalized_count = 0

    for example in dataset:
        # Use `or ""` to handle None values
        instruction = (example.get("problem") or "").strip()
        solution    = (example.get("solution") or "").strip()

        if not instruction:
            skipped_reasons["missing_instruction"] += 1
            continue

        if not solution:
            skipped_reasons["missing_solution"] += 1
            continue

        sft_type = classify_code_sft_type(instruction, solution)

        if sft_type == "code_explanation":
            if looks_like_mostly_code(solution):
                skipped_reasons["code_explanation_is_code"] += 1
                continue
        else:
            if is_prose_heavy_without_code(solution):
                skipped_reasons["prose_only"] += 1
                continue

            if not looks_like_code(solution):
                skipped_reasons["no_code_detected"] += 1
                continue

        normalized_solution, normalized = normalize_code_solution(solution, sft_type)

        if not normalized_solution:
            skipped_reasons["empty_after_normalization"] += 1
            continue

        if normalized:
            normalized_count += 1

        type_counts[sft_type] += 1

        messages = [
            {"role": "system",    "content": CODE_SYSTEM},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": normalized_solution},
        ]

        records.append({
            "conversations": messages,
            "source": "magicoder",
            "sft_type": sft_type,
            "normalized": normalized,
        })

    handcrafted_records = build_handcrafted_function_completion_records()
    records.extend(handcrafted_records)
    type_counts["function_completion"] += len(handcrafted_records)
    log.info(
        f"Added handcrafted function-completion examples: "
        f"{len(handcrafted_records):,}"
    )

    handcrafted_explanation_base = build_handcrafted_code_explanation_records()
    handcrafted_explanation_records = (
        handcrafted_explanation_base * HANDCRAFTED_CODE_EXPLANATION_REPEAT
    )
    records.extend(handcrafted_explanation_records)
    type_counts["code_explanation"] += len(handcrafted_explanation_records)
    log.info(
        f"Added handcrafted code-explanation examples: "
        f"{len(handcrafted_explanation_records):,} "
        f"({len(handcrafted_explanation_base):,} unique × "
        f"{HANDCRAFTED_CODE_EXPLANATION_REPEAT})"
    )

    skipped = sum(skipped_reasons.values())
    log.info(f"Processed: {len(records):,} kept, {skipped:,} skipped")
    log.info(f"Normalized code-only outputs: {normalized_count:,}")
    if skipped_reasons:
        log.info("Skipped by reason:")
        for reason, count in skipped_reasons.most_common():
            log.info(f"  {reason}: {count:,}")
    if type_counts:
        log.info("Kept by sft_type:")
        for sft_type, count in type_counts.most_common():
            log.info(f"  {sft_type}: {count:,}")

    if not records:
        raise RuntimeError(
            "Code SFT filtering removed all examples. Relax looks_like_code() "
            "or inspect Magicoder schema changes."
        )

    # Split
    n_val = max(500, int(len(records) * val_fraction))
    n_val = min(n_val, max(1, len(records) - 1))
    random.seed(42)
    random.shuffle(records)
    val_records   = records[:n_val]
    train_records = records[n_val:]

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    log.info(f"Code SFT: {len(train_records):,} train, {len(val_records):,} val")

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT datasets")
    parser.add_argument(
        "--stage",
        choices=["chat", "code", "both"],
        default="both",
        help="Which SFT stage to prepare",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help=(
            "Override validation fraction. If unset, uses stage-specific "
            f"defaults: {DEFAULT_VAL_FRACTION}"
        ),
    )
    args = parser.parse_args()

    if args.stage in ("chat", "both"):
        log.info("=== Preparing Chat SFT data (OpenHermes-2.5) ===")
        frac = args.val_fraction if args.val_fraction is not None else DEFAULT_VAL_FRACTION["chat"]
        prepare_chat(val_fraction=frac)

    if args.stage in ("code", "both"):
        log.info("=== Preparing Code SFT data (Magicoder-OSS-Instruct) ===")
        frac = args.val_fraction if args.val_fraction is not None else DEFAULT_VAL_FRACTION["code"]
        prepare_code(val_fraction=frac)

    log.info("SFT data preparation complete.")
    log.info(f"Output: {SFT_DIR}")


if __name__ == "__main__":
    main()
