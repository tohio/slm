"""
alignment/data/prepare_dpo.py
------------------------------
Download and format DPO preference datasets.

Blends three complementary sources:
    1. Anthropic/hh-rlhf        — 170k human preference pairs
    2. Intel/orca_dpo_pairs      — synthetic Orca preference pairs
    3. argilla/dpo-mix-7k        — curated high quality mix

Output format — one preference pair per line:
    {
        "prompt":   "<|system|>...<|user|>...<|endofturn|><|assistant|>",
        "chosen":   "<assistant response that was preferred>",
        "rejected": "<assistant response that was not preferred>",
        "source":   "hh-rlhf | orca | argilla"
    }

This format is directly consumed by trl DPOTrainer.

Usage:
    python alignment/data/prepare_dpo.py
    python alignment/data/prepare_dpo.py --source all
    python alignment/data/prepare_dpo.py --source hh-rlhf
"""

import argparse
import json
import logging
import os
import random
import sys
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
DPO_DIR = DATA_DIR / "dpo"

DEFAULT_SYSTEM = "You are a helpful, harmless, and honest assistant."


def format_prompt(system: str, user: str) -> str:
    """Format a prompt in the SLM chat template."""
    system = system.strip() or DEFAULT_SYSTEM
    return f"<|system|>{system}<|endofturn|><|user|>{user.strip()}<|endofturn|><|assistant|>"


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records):,} records to {path}")


# ── Source 1: Anthropic/hh-rlhf ───────────────────────────────────────────────

def prepare_hh_rlhf(max_examples: int = 50_000) -> list[dict]:
    """
    Format Anthropic/hh-rlhf preference pairs.

    hh-rlhf contains human-written conversations where annotators
    chose between two assistant responses. Covers helpfulness and
    harmlessness preference signals.

    Format: each example has 'chosen' and 'rejected' full conversation strings.
    We extract the last user/assistant turn as the preference pair.
    """
    from datasets import load_dataset

    log.info("Loading Anthropic/hh-rlhf...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    log.info(f"  hh-rlhf: {len(dataset):,} examples")

    records = []
    skipped = 0

    for example in dataset:
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")

        if not chosen or not rejected:
            skipped += 1
            continue

        # Extract the final exchange — last Human/Assistant pair
        parsed = _parse_hh_rlhf(chosen, rejected)
        if parsed is None:
            skipped += 1
            continue

        prompt, chosen_resp, rejected_resp = parsed

        if not chosen_resp or not rejected_resp:
            skipped += 1
            continue

        if chosen_resp == rejected_resp:
            skipped += 1
            continue

        records.append({
            "prompt": prompt,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
            "source": "hh-rlhf",
        })

        if len(records) >= max_examples:
            break

    log.info(f"  hh-rlhf: {len(records):,} kept, {skipped:,} skipped")
    return records


def _parse_hh_rlhf(chosen: str, rejected: str) -> tuple | None:
    """
    Parse the hh-rlhf conversation format into (prompt, chosen, rejected).

    hh-rlhf format:
        Human: <user turn>\n\nAssistant: <response>\n\nHuman: ...\n\nAssistant: <final>
    """
    import re

    def extract_turns(text):
        turns = re.split(r"\n\nHuman: |\n\nAssistant: ", text)
        return [t.strip() for t in turns if t.strip()]

    chosen_turns = extract_turns(chosen)
    rejected_turns = extract_turns(rejected)

    if len(chosen_turns) < 2:
        return None

    # Build prompt from all turns except the last assistant response
    # The last turn in chosen is the preferred response
    chosen_response = chosen_turns[-1]
    rejected_response = rejected_turns[-1] if rejected_turns else ""

    # Reconstruct the conversation up to the last user message
    conversation_turns = chosen_turns[:-1]
    if not conversation_turns:
        return None

    # Format as SLM chat template
    messages = []
    messages.append({"role": "system", "content": DEFAULT_SYSTEM})
    for i, turn in enumerate(conversation_turns):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": turn})

    # Build prompt string
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            parts.append(f"<|system|>{msg['content']}<|endofturn|>")
        elif msg["role"] == "user":
            parts.append(f"<|user|>{msg['content']}<|endofturn|>")
        elif msg["role"] == "assistant":
            parts.append(f"<|assistant|>{msg['content']}<|endofturn|>")
    parts.append("<|assistant|>")
    prompt = "".join(parts)

    return prompt, chosen_response, rejected_response


# ── Source 2: Intel/orca_dpo_pairs ────────────────────────────────────────────

def prepare_orca_dpo(max_examples: int = 30_000) -> list[dict]:
    """
    Format Intel/orca_dpo_pairs preference pairs.

    Orca DPO pairs are high-quality synthetic preference pairs generated
    by comparing GPT-4 responses (chosen) vs GPT-3.5 responses (rejected)
    on complex reasoning tasks.
    """
    from datasets import load_dataset

    log.info("Loading Intel/orca_dpo_pairs...")
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    log.info(f"  orca_dpo_pairs: {len(dataset):,} examples")

    records = []
    skipped = 0

    for example in dataset:
        system = example.get("system", DEFAULT_SYSTEM).strip()
        question = example.get("question", "").strip()
        chosen = example.get("chosen", "").strip()
        rejected = example.get("rejected", "").strip()

        if not question or not chosen or not rejected:
            skipped += 1
            continue

        if chosen == rejected:
            skipped += 1
            continue

        prompt = format_prompt(system, question)

        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "orca",
        })

        if len(records) >= max_examples:
            break

    log.info(f"  orca: {len(records):,} kept, {skipped:,} skipped")
    return records


# ── Source 3: argilla/dpo-mix-7k ──────────────────────────────────────────────

def prepare_argilla_dpo() -> list[dict]:
    """
    Format argilla/dpo-mix-7k preference pairs.

    A carefully curated mix of 7k high-quality DPO pairs from multiple
    sources, filtered for quality. Small but high signal.
    """
    from datasets import load_dataset

    log.info("Loading argilla/dpo-mix-7k...")
    dataset = load_dataset("argilla/dpo-mix-7k", split="train")
    log.info(f"  dpo-mix-7k: {len(dataset):,} examples")

    records = []
    skipped = 0

    for example in dataset:
        system = example.get("system", DEFAULT_SYSTEM) or DEFAULT_SYSTEM
        instruction = example.get("instruction", "").strip()
        chosen = example.get("chosen", "").strip()
        rejected = example.get("rejected", "").strip()

        # Handle nested format
        if isinstance(chosen, list):
            chosen = chosen[-1].get("content", "") if chosen else ""
        if isinstance(rejected, list):
            rejected = rejected[-1].get("content", "") if rejected else ""

        if not instruction or not chosen or not rejected:
            skipped += 1
            continue

        if chosen == rejected:
            skipped += 1
            continue

        prompt = format_prompt(system, instruction)

        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "argilla",
        })

    log.info(f"  argilla: {len(records):,} kept, {skipped:,} skipped")
    return records


# ── Blend and split ────────────────────────────────────────────────────────────

def blend_and_split(
    records: list[dict],
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split into train/val."""
    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(500, int(len(records) * val_fraction))
    return records[n_val:], records[:n_val]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare DPO datasets")
    parser.add_argument(
        "--source",
        choices=["all", "hh-rlhf", "orca", "argilla"],
        default="all",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    train_path = DPO_DIR / "train.jsonl"
    val_path = DPO_DIR / "val.jsonl"

    if train_path.exists() and val_path.exists():
        log.info(f"DPO data already exists at {DPO_DIR}")
        return

    all_records = []

    if args.source in ("all", "hh-rlhf"):
        all_records.extend(prepare_hh_rlhf())

    if args.source in ("all", "orca"):
        all_records.extend(prepare_orca_dpo())

    if args.source in ("all", "argilla"):
        all_records.extend(prepare_argilla_dpo())

    log.info(f"Total records: {len(all_records):,}")

    # Source breakdown
    from collections import Counter
    source_counts = Counter(r["source"] for r in all_records)
    for source, count in source_counts.items():
        log.info(f"  {source:<15} {count:>8,}  ({100*count/len(all_records):.1f}%)")

    train_records, val_records = blend_and_split(all_records, args.val_fraction)

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    # Save stats
    stats = {
        "total": len(all_records),
        "train": len(train_records),
        "val": len(val_records),
        "sources": dict(source_counts),
    }
    with open(DPO_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("DPO data preparation complete.")


if __name__ == "__main__":
    main()