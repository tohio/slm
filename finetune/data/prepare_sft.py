"""
finetune/data/prepare_sft.py
-----------------------------
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
CODE_SYSTEM    = "You are an expert programming assistant. When code is requested, write code directly and avoid unnecessary explanation."

# Per-stage defaults for validation fraction. These are the sources of truth;
# CLI --val-fraction overrides them only when explicitly passed.
DEFAULT_VAL_FRACTION = {
    "chat": 0.02,
    "code": 0.05,
}


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records):,} records to {path}")


# ── Code SFT filtering helpers ────────────────────────────────────────────────

CODE_KEYWORDS = (
    "def ",
    "class ",
    "return ",
    "import ",
    "from ",
    "for ",
    "while ",
    "if ",
    "elif ",
    "else:",
    "try:",
    "except ",
    "with ",
    "lambda ",
    "async ",
    "await ",
)

PROSE_ONLY_STARTS = (
    "the given code",
    "the provided code",
    "this code",
    "this function",
    "this program",
    "to solve this problem",
    "to implement this",
    "here is an explanation",
    "here's an explanation",
    "in this solution",
    "we need to",
    "you can",
)

CODE_OUTPUT_TYPES = {"code_generation", "function_completion", "code_repair"}

CODE_BLOCK_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+\-.#]+)?\s*\n(.*?)```",
    re.DOTALL,
)


def has_code_fence(text: str) -> bool:
    return "```" in text


def count_indented_code_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if line.startswith(("    ", "	")) and not stripped.startswith(("-", "*")):
            count += 1
    return count


def count_code_keyword_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(keyword) for keyword in CODE_KEYWORDS):
            count += 1
    return count


def has_programming_syntax(text: str) -> bool:
    patterns = [
        r"\w+\s*=\s*[^=]",
        r"\w+\([^)]*\)",
        r"[{};]",
        r"==|!=|<=|>=|->|=>",
        r"\[[^\]]+\]",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def looks_like_code(text: str) -> bool:
    """Return True when an assistant response contains substantial code.

    This intentionally uses broad signals because Magicoder contains multiple
    languages and formats. The goal is not perfect language detection; the goal
    is to drop obvious prose-only examples from the code SFT stage.
    """
    if not text or len(text.strip()) < 20:
        return False

    keyword_lines = count_code_keyword_lines(text)
    indented_lines = count_indented_code_lines(text)

    if has_code_fence(text) and (keyword_lines >= 1 or has_programming_syntax(text)):
        return True

    if keyword_lines >= 2:
        return True

    if indented_lines >= 2 and has_programming_syntax(text):
        return True

    if "def " in text and "return" in text:
        return True

    if "class " in text and ("def " in text or "return" in text):
        return True

    return False


def is_prose_heavy_without_code(text: str) -> bool:
    stripped = text.strip().lower()
    starts_like_explanation = stripped.startswith(PROSE_ONLY_STARTS)
    return starts_like_explanation and not looks_like_code(text)


STRICT_FUNCTION_COMPLETION_PHRASES = (
    "return only the function body",
    "return only function body",
    "return only the method body",
    "return only method body",
    "function body only",
    "method body only",
    "complete the function body",
    "complete this function body",
    "complete the method body",
    "complete this method body",
    "implement the function body",
    "implement this function body",
    "implement the method body",
    "implement this method body",
    "fill in the function body",
    "fill in this function body",
    "fill in the method body",
    "fill in this method body",
    "replace the pass statement",
    "replace pass with",
)


def is_strict_function_completion_prompt(prompt: str) -> bool:
    """Return True only for body/completion-style code prompts.

    Magicoder often says "complete the implementation" for broad tasks that
    expect a full function, class, script, or module. Those are useful code
    generation examples, but they are not HumanEval-style function-completion
    examples. Keep `function_completion` narrow so it teaches body-only /
    missing-code behavior instead of full-code generation.
    """
    prompt = prompt.lower()

    if any(phrase in prompt for phrase in STRICT_FUNCTION_COMPLETION_PHRASES):
        return True

    body_pattern = re.compile(
        r"\b(complete|implement|fill in|write)\b.{0,120}\b(function|method)\b.{0,120}\bbody\b",
        re.DOTALL,
    )
    if body_pattern.search(prompt):
        return True

    return_only_body_pattern = re.compile(
        r"\breturn only\b.{0,80}\b(function|method)\b.{0,40}\bbody\b",
        re.DOTALL,
    )
    if return_only_body_pattern.search(prompt):
        return True

    missing_code_pattern = re.compile(
        r"\b(fill in|complete)\b.{0,80}\b(missing code|todo|pass statement)\b",
        re.DOTALL,
    )
    if missing_code_pattern.search(prompt):
        return True

    return False


def classify_code_sft_type(instruction: str, solution: str) -> str:
    prompt = instruction.lower()
    answer = solution.lower()

    # Explanation and repair prompts are task-mode specific and should not be
    # swallowed by broad completion wording.
    if "explain" in prompt or "what does" in prompt or "describe" in prompt:
        return "code_explanation"

    if "fix" in prompt or "debug" in prompt or "bug" in prompt:
        return "code_repair"

    # Keep function_completion strict. Broad prompts like "complete the
    # implementation of this class/method" usually expect full code and should
    # remain code_generation.
    if is_strict_function_completion_prompt(prompt):
        return "function_completion"

    if "```" in answer or looks_like_code(solution):
        return "code_generation"

    return "code_other"


def extract_fenced_code(text: str) -> str | None:
    """Return the largest fenced code block from a response, without fences."""
    blocks = [match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text)]
    blocks = [block for block in blocks if block]
    if not blocks:
        return None
    return max(blocks, key=len).strip()


def normalize_code_solution(solution: str, sft_type: str) -> tuple[str, bool]:
    """Normalize assistant output for code-output SFT examples.

    Magicoder often emits a fenced code block followed by prose explanation.
    For code_generation, function_completion, and code_repair, keep only the
    largest fenced code block when present. This teaches code-output tasks to
    produce code directly. For code_explanation, keep the original prose.
    """
    solution = solution.strip()

    if sft_type not in CODE_OUTPUT_TYPES:
        return solution, False

    fenced = extract_fenced_code(solution)
    if fenced:
        return fenced, True

    return solution, False


# ── Chat SFT — OpenHermes-2.5 ─────────────────────────────────────────────────

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

        if is_prose_heavy_without_code(solution):
            skipped_reasons["prose_only"] += 1
            continue

        if not looks_like_code(solution):
            skipped_reasons["no_code_detected"] += 1
            continue

        sft_type = classify_code_sft_type(instruction, solution)
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