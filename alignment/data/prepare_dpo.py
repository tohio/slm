"""
alignment/data/prepare_dpo.py
------------------------------
Downloads and formats preference datasets for DPO training.

Datasets used:
  - Anthropic HH-RLHF  (helpfulness + harmlessness preference pairs)
  - UltraFeedback      (GPT-4 rated preference pairs, high quality)

NeMo Aligner DPO expects JSONL with:
  {
    "prompt":   "User: What is...\nAssistant:",
    "chosen":   "Python is a...",
    "rejected": "idk lol"
  }

Usage:
  python prepare_dpo.py --output-dir /data/dpo
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
logger = logging.getLogger("dpo.prepare")

VAL_SPLIT = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Dataset formatters
# ─────────────────────────────────────────────────────────────────────────────

def format_hh_rlhf(example: dict) -> dict | None:
    """
    Format Anthropic HH-RLHF dataset.
    The dataset provides chosen and rejected conversation strings directly.
    We extract the final assistant turn from each.
    """
    chosen = example.get("chosen", "").strip()
    rejected = example.get("rejected", "").strip()

    if not chosen or not rejected:
        return None

    # HH-RLHF format: "Human: ...\n\nAssistant: ..."
    # Extract everything up to the last Assistant turn as prompt
    if "\n\nAssistant:" not in chosen:
        return None

    prompt_end = chosen.rfind("\n\nAssistant:")
    prompt = chosen[:prompt_end + len("\n\nAssistant:")]
    chosen_response = chosen[prompt_end + len("\n\nAssistant:"):].strip()

    # Get rejected response
    if "\n\nAssistant:" not in rejected:
        return None
    rej_end = rejected.rfind("\n\nAssistant:")
    rejected_response = rejected[rej_end + len("\n\nAssistant:"):].strip()

    # Filter trivially short or identical responses
    if len(chosen_response) < 20 or len(rejected_response) < 20:
        return None
    if chosen_response == rejected_response:
        return None

    return {
        "prompt":   prompt,
        "chosen":   chosen_response,
        "rejected": rejected_response,
    }


def _get_ultrafeedback_score(completion: dict) -> float | None:
    """
    Extract score from a UltraFeedback completion dict.
    Handles both schema versions:
      - Old: completion["overall_score"]
      - New: completion["annotations"]["overall_score"]
    Returns None if score cannot be parsed.
    """
    # Try new schema first
    score = completion.get("annotations", {}).get("overall_score", None)
    if score is None:
        # Fall back to old schema
        score = completion.get("overall_score", None)
    if score is None:
        return None
    try:
        return float(score)
    except (ValueError, TypeError):
        return None


def format_ultrafeedback(example: dict) -> dict | None:
    """
    Format UltraFeedback dataset.
    Contains GPT-4 rated completions — we take highest vs lowest rated.
    Handles both old and new dataset schemas.
    """
    instruction = example.get("instruction", "").strip()
    completions = example.get("completions", [])

    if not instruction or len(completions) < 2:
        return None

    # Sort by overall score — take best and worst
    scored = []
    for c in completions:
        score = _get_ultrafeedback_score(c)
        response = c.get("response", "").strip()
        if score is not None and response:
            scored.append((score, response))

    if len(scored) < 2:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, chosen = scored[0]
    worst_score, rejected = scored[-1]

    # Only use pairs with meaningful score separation
    if (best_score - worst_score) < 1.0:
        return None

    if len(chosen) < 20 or len(rejected) < 20:
        return None

    prompt = f"Human: {instruction}\n\nAssistant:"

    return {
        "prompt":   prompt,
        "chosen":   chosen,
        "rejected": rejected,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dpo(output_dir: Path, seed: int = 42):
    random.seed(seed)
    all_examples = []

    # Anthropic HH-RLHF
    logger.info("Loading Anthropic HH-RLHF (helpfulness subset)...")
    try:
        hh = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
        before = len(all_examples)
        for ex in hh:
            formatted = format_hh_rlhf(ex)
            if formatted:
                all_examples.append(formatted)
        logger.info(f"  HH-RLHF: {len(all_examples) - before:,} examples")
    except Exception as e:
        logger.warning(f"Failed to load HH-RLHF: {e}")

    # UltraFeedback
    logger.info("Loading UltraFeedback...")
    try:
        uf = load_dataset("openbmb/UltraFeedback", split="train")

        # Log schema of first example to help debug if formatter produces 0 results
        if len(uf) > 0:
            sample_completion = uf[0].get("completions", [{}])[0]
            logger.info(f"  UltraFeedback completion keys: {list(sample_completion.keys())}")

        before = len(all_examples)
        for ex in uf:
            formatted = format_ultrafeedback(ex)
            if formatted:
                all_examples.append(formatted)

        added = len(all_examples) - before
        logger.info(f"  UltraFeedback: {added:,} examples")
        if added == 0:
            logger.warning(
                "  UltraFeedback produced 0 examples — dataset schema may have changed. "
                "Check completion keys logged above."
            )
    except Exception as e:
        logger.warning(f"Failed to load UltraFeedback: {e}")

    if not all_examples:
        raise RuntimeError("No DPO examples loaded. Check dataset availability and schema.")

    # Shuffle and split
    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * VAL_SPLIT))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)

    for split, data in [("train", train_examples), ("val", val_examples)]:
        out_file = output_dir / f"{split}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"  {split}: {len(data):,} examples → {out_file}")

    logger.info(
        f"DPO dataset prepared: {len(train_examples):,} train, {len(val_examples):,} val"
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO preference datasets")
    parser.add_argument("--output-dir", default="/data/dpo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dpo(Path(args.output_dir), seed=args.seed)


if __name__ == "__main__":
    main()