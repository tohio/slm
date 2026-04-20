"""
alignment/data/prepare_dpo.py
------------------------------
Download and format DPO preference datasets.

Blends three complementary sources:
    1. Anthropic/hh-rlhf        — 170k pairs upstream, capped to 50k (shuffled)
    2. Intel/orca_dpo_pairs      — ~12k synthetic Orca preference pairs
    3. argilla/dpo-mix-7k        — ~7k curated high-quality pairs

Output format — conversational format for trl DPOTrainer:
    {
        "prompt":   [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
        "chosen":   [{"role": "assistant", "content": "preferred response"}],
        "rejected": [{"role": "assistant", "content": "rejected response"}],
        "source":   "hh-rlhf | orca | argilla"
    }

trl DPOTrainer detects list inputs and uses apply_chat_template, which
tokenizes the full conversation consistently — avoiding BPE boundary
mismatch warnings that occur with plain string prompts.

Length filtering (defense in depth):
    trl 0.29 supports DPOConfig.max_prompt_length (applied by the data
    collator at load time). However, trl's truncation may still drop the
    start of an overlong prompt, and responses exceeding max_length also
    get truncated. We additionally filter here using the actual SLM
    tokenizer: drop any pair where
        len(prompt) + max(len(chosen), len(rejected))
    exceeds MAX_TOTAL_TOKENS (2048 = smallest model size's DPOConfig.max_length).
    This means train-time truncation never fires on the prepared dataset,
    the filtered dataset serves all three model sizes without re-preparation,
    and the whole contract survives the eventual trl 1.0 upgrade (where
    max_prompt_length is removed and the filter becomes load-bearing).

Usage:
    python alignment/data/prepare_dpo.py
    python alignment/data/prepare_dpo.py --source all
    python alignment/data/prepare_dpo.py --source hh-rlhf
    python alignment/data/prepare_dpo.py --force            # re-run even if output exists
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
DPO_DIR  = DATA_DIR / "dpo"

DEFAULT_SYSTEM = "You are a helpful, harmless, and honest assistant."

# Token-budget ceiling for prompt + max(chosen, rejected). Set to the smallest
# of the three model sizes' DPO max_length (125m=350m=2048, 1b=4096) so one
# prepared dataset serves all sizes. DPO rarely needs >2048 context.
MAX_TOTAL_TOKENS = 2048

# Upstream cap for hh-rlhf (170k pairs; capping for blend balance with orca/argilla).
HH_RLHF_CAP = 50_000


def make_prompt(system: str, user: str) -> list[dict]:
    """Return prompt as a list of message dicts for trl conversational format."""
    return [
        {"role": "system", "content": (system or DEFAULT_SYSTEM).strip()},
        {"role": "user",   "content": user.strip()},
    ]


def make_response(content: str) -> list[dict]:
    """Return a single assistant message dict."""
    return [{"role": "assistant", "content": content.strip()}]


def extract_text(value) -> str:
    """
    Safely extract a string from a field that may be:
      - str: return as-is
      - list of dicts with 'content': return last content value
      - list of str: return last element
      - None: return ""
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return ""
        last = value[-1]
        if isinstance(last, dict):
            return last.get("content", "") or ""
        return str(last)
    return str(value)


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records):,} records to {path}")


# ── Length filter ──────────────────────────────────────────────────────────────

def load_tokenizer_for_filter():
    """
    Load the SLM tokenizer for length counting. Uses the same tokenizer that
    train_dpo.py will use at training time, so counts are exact.
    """
    from transformers import PreTrainedTokenizerFast

    tokenizer_path = DATA_DIR / "tokenizer"
    if not (tokenizer_path / "tokenizer_config.json").exists():
        raise FileNotFoundError(
            f"tokenizer_config.json not found at {tokenizer_path}. "
            f"Run: python tokenizer/train_tokenizer.py"
        )
    return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))


def apply_length_filter(
    records: list[dict],
    tokenizer,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> list[dict]:
    """
    Drop records where len(prompt) + max(len(chosen), len(rejected)) exceeds
    the token budget. Tokenizes via apply_chat_template to match what trl
    DPOTrainer does at training time — so counts are the real counts trl
    will see, not approximations.

    Tracks drop reasons per source so we can surface them in logs.
    """
    from collections import Counter

    kept = []
    dropped_by_source = Counter()
    total_by_source   = Counter()

    for rec in records:
        total_by_source[rec["source"]] += 1

        # Tokenize prompt once (shared by chosen/rejected)
        prompt_ids = tokenizer.apply_chat_template(
            rec["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )
        # For responses, we tokenize only the assistant content — not through
        # apply_chat_template, since that would re-add system/user. This slightly
        # underestimates vs. trl's internal tokenization (which may add special
        # tokens around the response), but the underestimate is ≤ 4 tokens and
        # we're filtering with a safety margin anyway.
        chosen_content   = rec["chosen"][0]["content"]
        rejected_content = rec["rejected"][0]["content"]
        chosen_ids   = tokenizer.encode(chosen_content,   add_special_tokens=False)
        rejected_ids = tokenizer.encode(rejected_content, add_special_tokens=False)

        total = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
        # 16-token safety margin for trl's added special tokens around responses
        if total + 16 > max_total_tokens:
            dropped_by_source[rec["source"]] += 1
            continue

        kept.append(rec)

    log.info(f"Length filter (max_total_tokens={max_total_tokens}):")
    for source in sorted(total_by_source):
        total = total_by_source[source]
        dropped = dropped_by_source[source]
        pct = 100 * dropped / total if total else 0
        log.info(f"  {source:<15} dropped {dropped:>6,}/{total:<6,} ({pct:.1f}%)")
    log.info(f"  total kept: {len(kept):,} / {len(records):,} "
             f"({100 * len(kept) / len(records):.1f}%)")
    return kept


# ── Source 1: Anthropic/hh-rlhf ───────────────────────────────────────────────

def prepare_hh_rlhf(cap: int = HH_RLHF_CAP, seed: int = 42) -> list[dict]:
    """
    hh-rlhf upstream is ~170k pairs. We cap at `cap` for blend balance with
    orca (~12k) and argilla (~7k). Dataset is shuffled before capping so the
    subset is representative, not a biased slice of the head.
    """
    from datasets import load_dataset

    log.info("Loading Anthropic/hh-rlhf...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    log.info(f"  hh-rlhf: {len(dataset):,} examples upstream")

    # Shuffle with a fixed seed so runs are reproducible but not biased
    # toward the dataset's natural ordering.
    dataset = dataset.shuffle(seed=seed)

    records = []
    skipped = 0

    for example in dataset:
        chosen   = extract_text(example.get("chosen"))
        rejected = extract_text(example.get("rejected"))

        if not chosen or not rejected:
            skipped += 1
            continue

        parsed = _parse_hh_rlhf(chosen, rejected)
        if parsed is None:
            skipped += 1
            continue

        prompt_msgs, chosen_resp, rejected_resp = parsed

        if not chosen_resp or not rejected_resp or chosen_resp == rejected_resp:
            skipped += 1
            continue

        records.append({
            "prompt":   prompt_msgs,
            "chosen":   make_response(chosen_resp),
            "rejected": make_response(rejected_resp),
            "source":   "hh-rlhf",
        })

        if len(records) >= cap:
            break

    log.info(f"  hh-rlhf: {len(records):,} kept (cap={cap:,}), {skipped:,} skipped")
    return records


def _parse_hh_rlhf(chosen: str, rejected: str) -> tuple | None:
    """Parse hh-rlhf into (prompt_messages, chosen_response, rejected_response)."""
    import re

    def extract_turns(text):
        turns = re.split(r"\n\nHuman: |\n\nAssistant: ", text)
        return [t.strip() for t in turns if t.strip()]

    chosen_turns   = extract_turns(chosen)
    rejected_turns = extract_turns(rejected)

    if len(chosen_turns) < 2:
        return None

    chosen_response   = chosen_turns[-1]
    rejected_response = rejected_turns[-1] if rejected_turns else ""
    conversation_turns = chosen_turns[:-1]

    if not conversation_turns:
        return None

    messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
    for i, turn in enumerate(conversation_turns):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": turn})

    return messages, chosen_response, rejected_response


# ── Source 2: Intel/orca_dpo_pairs ────────────────────────────────────────────

def prepare_orca_dpo() -> list[dict]:
    """
    orca_dpo_pairs is ~12k pairs upstream — small enough that no cap is needed.
    """
    from datasets import load_dataset

    log.info("Loading Intel/orca_dpo_pairs...")
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    log.info(f"  orca_dpo_pairs: {len(dataset):,} examples")

    records = []
    skipped = 0

    for example in dataset:
        system   = extract_text(example.get("system")) or DEFAULT_SYSTEM
        question = extract_text(example.get("question")).strip()
        chosen   = extract_text(example.get("chosen")).strip()
        rejected = extract_text(example.get("rejected")).strip()

        if not question or not chosen or not rejected or chosen == rejected:
            skipped += 1
            continue

        records.append({
            "prompt":   make_prompt(system, question),
            "chosen":   make_response(chosen),
            "rejected": make_response(rejected),
            "source":   "orca",
        })

    log.info(f"  orca: {len(records):,} kept, {skipped:,} skipped")
    return records


# ── Source 3: argilla/dpo-mix-7k ──────────────────────────────────────────────

def prepare_argilla_dpo() -> list[dict]:
    from datasets import load_dataset

    log.info("Loading argilla/dpo-mix-7k...")
    dataset = load_dataset("argilla/dpo-mix-7k", split="train")
    log.info(f"  dpo-mix-7k: {len(dataset):,} examples")

    records = []
    skipped = 0

    for example in dataset:
        system      = extract_text(example.get("system")) or DEFAULT_SYSTEM
        instruction = extract_text(example.get("instruction")).strip()
        chosen      = extract_text(example.get("chosen")).strip()
        rejected    = extract_text(example.get("rejected")).strip()

        if not instruction or not chosen or not rejected or chosen == rejected:
            skipped += 1
            continue

        records.append({
            "prompt":   make_prompt(system, instruction),
            "chosen":   make_response(chosen),
            "rejected": make_response(rejected),
            "source":   "argilla",
        })

    log.info(f"  argilla: {len(records):,} kept, {skipped:,} skipped")
    return records


# ── Blend and split ────────────────────────────────────────────────────────────

def blend_and_split(
    records: list[dict],
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
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
        help="Which source(s) to prepare",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=MAX_TOTAL_TOKENS,
        help=(
            "Drop pairs where len(prompt) + max(len(chosen), len(rejected)) "
            "exceeds this. Default is the smallest model-size max_length."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output files already exist",
    )
    args = parser.parse_args()

    train_path = DPO_DIR / "train.jsonl"
    val_path   = DPO_DIR / "val.jsonl"

    if train_path.exists() and val_path.exists() and not args.force:
        log.info(
            f"DPO data already exists at {DPO_DIR}. "
            f"Use --force to regenerate."
        )
        return

    # Tokenizer is required for length filtering. Load it once, up front, so
    # the run fails fast if the tokenizer isn't available.
    log.info("Loading tokenizer for length filtering...")
    tokenizer = load_tokenizer_for_filter()
    log.info(f"  vocab_size: {tokenizer.vocab_size:,}")

    all_records = []

    if args.source in ("all", "hh-rlhf"):
        all_records.extend(prepare_hh_rlhf())
    if args.source in ("all", "orca"):
        all_records.extend(prepare_orca_dpo())
    if args.source in ("all", "argilla"):
        all_records.extend(prepare_argilla_dpo())

    log.info(f"Total records before length filter: {len(all_records):,}")

    # Apply length filter with the same tokenizer trl will use at train time.
    all_records = apply_length_filter(all_records, tokenizer, args.max_total_tokens)

    from collections import Counter
    source_counts = Counter(r["source"] for r in all_records)
    for source, count in source_counts.items():
        pct = 100 * count / len(all_records) if all_records else 0
        log.info(f"  {source:<15} {count:>8,}  ({pct:.1f}%)")

    train_records, val_records = blend_and_split(all_records, args.val_fraction)

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    stats = {
        "total":            len(all_records),
        "train":            len(train_records),
        "val":              len(val_records),
        "sources":          dict(source_counts),
        "max_total_tokens": args.max_total_tokens,
        "hh_rlhf_cap":      HH_RLHF_CAP,
    }
    with open(DPO_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("DPO data preparation complete.")


if __name__ == "__main__":
    main()