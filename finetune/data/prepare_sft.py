"""
finetune/data/prepare_sft.py
-----------------------------
Downloads and formats SFT datasets into NeMo Aligner's expected JSONL format.

Chat stage datasets:
  - OpenAssistant OASST1  (multi-turn, human-labeled)
  - Dolly 15k             (single-turn instruction following)

Code stage datasets:
  - CodeSearchNet Python  (docstring → code pairs)

NeMo Aligner SFT expects ShareGPT-style JSONL:
  {
    "conversations": [
      {"from": "system",    "value": "You are a helpful assistant."},
      {"from": "human",     "value": "What is Python?"},
      {"from": "assistant", "value": "Python is a high-level..."}
    ]
  }

Usage:
  python prepare_sft.py --stage chat --output-dir /data/sft/chat
  python prepare_sft.py --stage code --output-dir /data/sft/code
  python prepare_sft.py --stage both
"""

import argparse
import json
import logging
import random
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sft.prepare")

SYSTEM_PROMPT_GENERAL = "You are a helpful, honest, and harmless AI assistant."
SYSTEM_PROMPT_CODE = (
    "You are an expert programming assistant. "
    "Write clean, correct, well-commented code. "
    "Explain your solutions clearly."
)

VAL_SPLIT = 0.05   # 5% validation


# ─────────────────────────────────────────────────────────────────────────────
# Chat datasets
# ─────────────────────────────────────────────────────────────────────────────

def build_oasst1_pairs(dataset) -> list[dict]:
    """
    Extract (prompt, response) pairs from the OASST1 message tree.

    The raw OpenAssistant/oasst1 dataset is a flat list of messages with
    parent_id references — not pre-flattened instruction/output pairs.
    Each message has: message_id, parent_id, role ('prompter'|'assistant'),
    text, rank (0 = best among siblings), lang.

    We extract the top-rated single-turn pairs:
      parent = prompter message (root or child)
      child  = assistant reply with rank=0 (highest rated)
    and filter to English only.
    """
    # Build lookup: message_id → message
    messages = {m["message_id"]: m for m in dataset}

    pairs = []
    for msg in dataset:
        # Only process top-rated assistant responses in English
        if msg["role"] != "assistant":
            continue
        if msg.get("lang", "en") != "en":
            continue
        if msg.get("rank", 999) != 0:
            continue

        parent = messages.get(msg["parent_id"])
        if parent is None or parent["role"] != "prompter":
            continue

        prompt = parent["text"].strip()
        response = msg["text"].strip()

        if len(prompt) < 10 or len(response) < 20:
            continue

        pairs.append({
            "conversations": [
                {"from": "system",    "value": SYSTEM_PROMPT_GENERAL},
                {"from": "human",     "value": prompt},
                {"from": "assistant", "value": response},
            ]
        })

    return pairs


def format_dolly(example: dict) -> dict | None:
    """Format Dolly 15k into ShareGPT format."""
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()

    if not instruction or not response:
        return None

    # Incorporate context into the human turn if present
    human_turn = instruction
    if context:
        human_turn = f"Context:\n{context}\n\nInstruction:\n{instruction}"

    return {
        "conversations": [
            {"from": "system",    "value": SYSTEM_PROMPT_GENERAL},
            {"from": "human",     "value": human_turn},
            {"from": "assistant", "value": response},
        ]
    }


def prepare_chat(output_dir: Path, seed: int = 42):
    """Download and format chat SFT datasets."""
    logger.info("Preparing chat SFT datasets...")
    random.seed(seed)
    all_examples = []

    # OpenAssistant OASST1
    logger.info("Loading OpenAssistant OASST1...")
    try:
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        # Log schema of first message to help debug if schema changes
        if len(oasst) > 0:
            logger.info(f"  OASST1 message keys: {list(oasst[0].keys())}")
        oasst_pairs = build_oasst1_pairs(oasst)
        all_examples.extend(oasst_pairs)
        logger.info(f"  OASST1: {len(oasst_pairs):,} examples")
        if len(oasst_pairs) == 0:
            logger.warning(
                "  OASST1 produced 0 examples — dataset schema may have changed. "
                "Check message keys logged above."
            )
    except Exception as e:
        logger.warning(f"Failed to load OASST1: {e}")

    # Dolly 15k
    logger.info("Loading Dolly 15k...")
    dolly_start = len(all_examples)
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        for ex in dolly:
            formatted = format_dolly(ex)
            if formatted:
                all_examples.append(formatted)
        logger.info(f"  Dolly: {len(all_examples) - dolly_start:,} examples")
    except Exception as e:
        logger.warning(f"Failed to load Dolly: {e}")

    if not all_examples:
        raise RuntimeError("No chat examples loaded. Check network access.")

    write_splits(all_examples, output_dir, seed)
    logger.info(f"Chat SFT data written to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Code datasets
# ─────────────────────────────────────────────────────────────────────────────

def format_codesearchnet(example: dict) -> dict | None:
    """
    Format CodeSearchNet into instruction → code pairs.
    Uses the docstring as the instruction, function body as the response.
    """
    docstring = example.get("func_documentation_string", "").strip()
    code = example.get("whole_func_string", "").strip()

    if not docstring or not code:
        return None
    if len(docstring) < 20 or len(code) < 50:
        return None

    instruction = f"Write a Python function based on this description:\n\n{docstring}"
    response = f"```python\n{code}\n```"

    return {
        "conversations": [
            {"from": "system",    "value": SYSTEM_PROMPT_CODE},
            {"from": "human",     "value": instruction},
            {"from": "assistant", "value": response},
        ]
    }


def format_code_explanation(example: dict) -> dict | None:
    """
    Generate 'explain this code' pairs from CodeSearchNet.
    Provides diversity — model learns to explain code, not just write it.
    """
    code = example.get("whole_func_string", "").strip()
    docstring = example.get("func_documentation_string", "").strip()

    if not code or not docstring or len(code) < 50:
        return None

    instruction = f"Explain what this Python code does:\n\n```python\n{code}\n```"
    response = docstring

    return {
        "conversations": [
            {"from": "system",    "value": SYSTEM_PROMPT_CODE},
            {"from": "human",     "value": instruction},
            {"from": "assistant", "value": response},
        ]
    }


def prepare_code(output_dir: Path, seed: int = 42):
    """Download and format code SFT datasets."""
    logger.info("Preparing code SFT datasets...")
    random.seed(seed)
    all_examples = []

    logger.info("Loading CodeSearchNet (Python)...")
    try:
        csn = load_dataset("code_search_net", "python", split="train")
        for ex in csn:
            # Generate both write and explain pairs for diversity
            write_ex = format_codesearchnet(ex)
            if write_ex:
                all_examples.append(write_ex)

            explain_ex = format_code_explanation(ex)
            if explain_ex:
                all_examples.append(explain_ex)

        logger.info(f"  CodeSearchNet: {len(all_examples):,} examples")
    except Exception as e:
        logger.warning(f"Failed to load CodeSearchNet: {e}")

    if not all_examples:
        raise RuntimeError("No code examples loaded. Check network access.")

    write_splits(all_examples, output_dir, seed)
    logger.info(f"Code SFT data written to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def write_splits(examples: list[dict], output_dir: Path, seed: int):
    """Shuffle and write train/val splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    random.shuffle(examples)

    n_val = max(1, int(len(examples) * VAL_SPLIT))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    for split, data in [("train", train_examples), ("val", val_examples)]:
        output_file = output_dir / f"{split}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"  {split}: {len(data):,} examples → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT datasets")
    parser.add_argument("--stage", required=True, choices=["chat", "code", "both"])
    parser.add_argument("--output-dir", help="Output directory (overrides default per-stage path)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage in ("chat", "both"):
        out = Path(args.output_dir) if args.output_dir else Path("/data/sft/chat")
        prepare_chat(out, args.seed)

    if args.stage in ("code", "both"):
        out = Path(args.output_dir) if args.output_dir else Path("/data/sft/code")
        prepare_code(out, args.seed)


if __name__ == "__main__":
    main()