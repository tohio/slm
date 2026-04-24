"""
scripts/sanity_train.py
------------------------
Self-contained sanity check for the SLM model + training code.

Removes tokenizer and curated data as variables: tokenizes FineWeb-Edu with
Mistral's tokenizer (vocab=32000, matches our architecture) and trains
either the mini or the 125m architecture on it. If learning fails here,
the issue is in model/ or the training loop. If it succeeds, both are
fine and any failure on the real pipeline is in the curator or the SLM
tokenizer.

Usage:
    python scripts/sanity_train.py                                  # 125m, 2.5B tokens
    python scripts/sanity_train.py --arch mini                      # mini, default 2.5B (override w/ --target-tokens)
    python scripts/sanity_train.py --arch mini --target-tokens 500000000
    python scripts/sanity_train.py --save                           # keep the trained model

Delete this file and the `sanity-train*` Makefile targets when no longer
needed — nothing else in the codebase depends on it.
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Architectures (mirror gpt_mini.yaml and gpt_125m.yaml) ──────────────────
MODEL_ARCH_MINI = {
    "hidden_size":             384,
    "num_hidden_layers":       6,
    "num_attention_heads":     6,
    "num_key_value_heads":     2,
    "max_position_embeddings": 1024,
    "rope_theta":              500_000.0,
    "rms_norm_eps":            1e-5,
    "initializer_range":       0.02,
    "tie_word_embeddings":     True,
}

MODEL_ARCH_125M = {
    "hidden_size":             768,
    "num_hidden_layers":       12,
    "num_attention_heads":     12,
    "num_key_value_heads":     4,
    "max_position_embeddings": 2048,
    "rope_theta":              500_000.0,
    "rms_norm_eps":            1e-5,
    "initializer_range":       0.02,
    "tie_word_embeddings":     True,
}

ARCH_REGISTRY = {
    "mini": MODEL_ARCH_MINI,
    "125m": MODEL_ARCH_125M,
}

REFERENCE_TOKENIZER = "mistralai/Mistral-7B-v0.1"
REFERENCE_DATASET   = "HuggingFaceFW/fineweb-edu"
REFERENCE_SUBSET    = "sample-10BT"

# QA probes — loss-based factual knowledge tests. For each pair, the model
# should assign lower loss to the correct continuation than the wrong one.
QA_PROBES = [
    ("The capital of France is", " Paris", " London"),
    ("Two plus two equals",       " four",  " five"),
    ("The sun rises in the",      " east",  " west"),
    ("Water freezes at zero degrees", " Celsius", " Fahrenheit"),
    ("The opposite of hot is",    " cold",  " warm"),
]

# Generation showcase — one greedy 50-token continuation to inspect by eye.
GEN_PROMPT = "The"


# ── Tokenization ─────────────────────────────────────────────────────────────

def tokenize_fineweb(tokenizer, target_tokens: int, output_path: Path) -> int:
    """Stream FineWeb-Edu, tokenize, write to bin file. Return tokens written."""
    from datasets import load_dataset

    log.info(f"Streaming {REFERENCE_DATASET}/{REFERENCE_SUBSET}...")
    ds = load_dataset(REFERENCE_DATASET, name=REFERENCE_SUBSET, split="train", streaming=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(target_tokens,))

    eos = tokenizer.eos_token_id or 2
    written = 0
    last_log = time.time()

    for example in ds:
        ids = tokenizer.encode(example["text"], add_special_tokens=False)
        ids.append(eos)
        end = min(written + len(ids), target_tokens)
        arr[written:end] = ids[: end - written]
        written = end
        if time.time() - last_log > 10:
            log.info(f"  Tokenized {written:,}/{target_tokens:,} "
                     f"({100 * written / target_tokens:.1f}%)")
            last_log = time.time()
        if written >= target_tokens:
            break

    arr.flush()
    log.info(f"Wrote {written:,} tokens to {output_path}")
    return written


# ── Dataset ──────────────────────────────────────────────────────────────────

class PackedTokensDataset(Dataset):
    """Memmap-backed dataset that returns fixed-length sequences."""

    def __init__(self, bin_path: Path, seq_len: int, n_tokens: int):
        self.data    = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        # Trim to multiple of seq_len; reserve last 1% for eval
        usable = (n_tokens // seq_len) * seq_len
        n_eval = max(seq_len * 4, usable // 100)
        n_eval = (n_eval // seq_len) * seq_len
        self.train_end = usable - n_eval
        self.eval_end  = usable

    def __len__(self):
        return self.train_end // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = np.array(self.data[start : start + self.seq_len], dtype=np.int64)
        return torch.from_numpy(chunk)

    def eval_batches(self, batch_size: int):
        """Yield eval batches from the held-out tail."""
        n = (self.eval_end - self.train_end) // self.seq_len
        positions = [self.train_end + i * self.seq_len for i in range(n)]
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i : i + batch_size]
            seqs = [
                np.array(self.data[p : p + self.seq_len], dtype=np.int64)
                for p in batch_positions
            ]
            yield torch.from_numpy(np.stack(seqs))


# ── Training ─────────────────────────────────────────────────────────────────

def cosine_lr(step: int, max_steps: int, peak_lr: float, warmup: int) -> float:
    if step < warmup:
        return peak_lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, dataset, batch_size: int, device) -> float:
    model.eval()
    losses = []
    for batch in dataset.eval_batches(batch_size):
        batch = batch.to(device)
        out = model(input_ids=batch, labels=batch)
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(model, dataset, args, device, arch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    use_autocast   = device.type == "cuda"

    seq_len = arch["max_position_embeddings"]
    log.info(f"Training: {args.max_steps:,} steps, batch={args.batch_size}, "
             f"seq_len={seq_len}, "
             f"~{args.batch_size * seq_len:,} tokens/step")

    log.info("Baseline eval (step 0)...")
    eval_loss = evaluate(model, dataset, args.batch_size, device)
    log.info(f"  eval_loss: {eval_loss:.4f}")

    model.train()
    step = 0
    t0 = time.time()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            batch = batch.to(device, non_blocking=True)
            lr = cosine_lr(step, args.max_steps, args.lr, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            if use_autocast:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    out = model(input_ids=batch, labels=batch)
                    loss = out.loss
            else:
                out = model(input_ids=batch, labels=batch)
                loss = out.loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                log.info(
                    f"  step {step:>6,}/{args.max_steps:,} | "
                    f"loss {loss.item():.4f} | "
                    f"grad_norm {grad_norm.item():.2f} | "
                    f"lr {lr:.2e} | "
                    f"{step / max(elapsed, 1):.1f} steps/s"
                )

            if step > 0 and step % args.eval_every == 0:
                eval_loss = evaluate(model, dataset, args.batch_size, device)
                log.info(f"  step {step:>6,} | eval_loss {eval_loss:.4f}")

            step += 1

    final_eval = evaluate(model, dataset, args.batch_size, device)
    log.info(f"Final eval_loss: {final_eval:.4f}")
    return final_eval


# ── QA evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def qa_loss(model, tokenizer, prefix: str, completion: str, device) -> float:
    """Compute average loss on `completion` tokens given `prefix` context."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids   = tokenizer.encode(prefix + completion, add_special_tokens=False)
    completion_ids = full_ids[len(prefix_ids):]
    if not completion_ids:
        return float("inf")

    input_ids = torch.tensor([full_ids], device=device)
    labels = input_ids.clone()
    # Only score the completion tokens
    labels[0, : len(prefix_ids)] = -100
    out = model(input_ids=input_ids, labels=labels)
    return out.loss.item()


@torch.no_grad()
def run_qa_suite(model, tokenizer, device):
    log.info("=" * 60)
    log.info("QA probes (loss-based: lower loss on correct = ✓)")
    log.info("=" * 60)
    correct = 0
    for prefix, right, wrong in QA_PROBES:
        loss_right = qa_loss(model, tokenizer, prefix, right, device)
        loss_wrong = qa_loss(model, tokenizer, prefix, wrong, device)
        ok = loss_right < loss_wrong
        marker = "✓" if ok else "✗"
        if ok:
            correct += 1
        log.info(
            f"  {marker} {prefix!r:40s} "
            f"{right!r}={loss_right:.3f}  vs  {wrong!r}={loss_wrong:.3f}"
        )
    log.info(f"QA score: {correct}/{len(QA_PROBES)}")

    # Generation showcase
    log.info("=" * 60)
    log.info(f"Generation from {GEN_PROMPT!r} (greedy, 50 tokens):")
    log.info("=" * 60)
    ids = tokenizer.encode(GEN_PROMPT, add_special_tokens=False, return_tensors="pt").to(device)
    out = model.generate(ids, max_new_tokens=50, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    log.info(f"  {text!r}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SLM sanity training (model + training code only)")
    p.add_argument("--arch",          type=str, default="125m",
                   choices=list(ARCH_REGISTRY.keys()),
                   help="Architecture to use: mini (21.7M) or 125m")
    p.add_argument("--target-tokens", type=int, default=2_500_000_000)
    p.add_argument("--batch-size",    type=int, default=16,
                   help="H200 default; lower for smaller GPUs")
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup-steps",  type=int, default=200)
    p.add_argument("--log-every",     type=int, default=50)
    p.add_argument("--eval-every",    type=int, default=2000)
    p.add_argument("--scratch-dir",   type=Path, default=Path("/tmp/slm-sanity"))
    p.add_argument("--save",          action="store_true",
                   help="Save trained model to results/sanity-<arch>/ instead of discarding")
    p.add_argument("--reuse-tokens",  action="store_true",
                   help="Skip tokenization if scratch bin already exists")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch   = ARCH_REGISTRY[args.arch]
    log.info(f"Device: {device}")
    log.info(f"Architecture: {args.arch} "
             f"(hidden={arch['hidden_size']}, layers={arch['num_hidden_layers']}, "
             f"seq_len={arch['max_position_embeddings']})")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    log.info(f"Loading {REFERENCE_TOKENIZER}...")
    tokenizer = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER, use_fast=True)
    log.info(f"  vocab_size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == 32000, (
        f"Expected Mistral tokenizer vocab=32000, got {tokenizer.vocab_size}. "
        f"Architecture mismatch — model uses vocab_size=32000."
    )

    # ── Tokenize FineWeb-Edu ─────────────────────────────────────────────────
    bin_path = args.scratch_dir / "fineweb_mistral.bin"
    if args.reuse_tokens and bin_path.exists():
        log.info(f"Reusing existing tokens at {bin_path}")
        n_tokens = bin_path.stat().st_size // 2  # uint16 = 2 bytes
        n_tokens = min(n_tokens, args.target_tokens)
    else:
        n_tokens = tokenize_fineweb(tokenizer, args.target_tokens, bin_path)

    # ── Model ────────────────────────────────────────────────────────────────
    from model import SLMConfig, SLMForCausalLM
    config = SLMConfig(vocab_size=32000, **arch)
    model = SLMForCausalLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {n_params:,} parameters ({n_params / 1e6:.1f}M)")

    # ── Dataset ──────────────────────────────────────────────────────────────
    seq_len = arch["max_position_embeddings"]
    dataset = PackedTokensDataset(bin_path, seq_len=seq_len, n_tokens=n_tokens)
    tokens_per_step = args.batch_size * seq_len
    args.max_steps  = n_tokens // tokens_per_step
    log.info(f"Dataset: {len(dataset):,} train sequences, max_steps={args.max_steps:,}")

    # ── Train ────────────────────────────────────────────────────────────────
    train(model, dataset, args, device, arch)

    # ── QA suite ─────────────────────────────────────────────────────────────
    run_qa_suite(model, tokenizer, device)

    # ── Save (optional) ──────────────────────────────────────────────────────
    if args.save:
        out_dir = Path(f"results/sanity-{args.arch}")
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir / "tokenizer"))
        log.info(f"Saved to {out_dir}")
    else:
        log.info("(Use --save to keep the trained model.)")


if __name__ == "__main__":
    main()