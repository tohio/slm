"""
Throwaway continued-pretraining experiment for synthetic arithmetic.

Purpose:
    Validate whether a small synthetic arithmetic continued-pretraining pass
    improves arithmetic sanity behavior before adding a real synthetic source
    to the curation pipeline.

This script intentionally does NOT modify:
    - pretrain/train.py
    - Makefile
    - config/data_mix.py
    - curator/
    - data/tokenized/
    - results/slm-125m/final

It writes:
    data/cpt_math/train.jsonl
    data/cpt_math/val.jsonl
    data/tokenized_cpt_math/train.bin
    data/tokenized_cpt_math/train.json
    data/tokenized_cpt_math/val.bin
    data/tokenized_cpt_math/val.json
    results/slm-125m-cpt-math/final
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import PreTrainedTokenizerFast, TrainingArguments

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import SLMForCausalLM  # noqa: E402
from pretrain.data.dataset import load_train_val  # noqa: E402
from pretrain.train import SLMTrainer, VRAMProbe, validate_tokenizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(False)


def num_word(n: int) -> str:
    words_0_19 = [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen",
    ]
    tens = {
        20: "twenty",
        30: "thirty",
        40: "forty",
        50: "fifty",
        60: "sixty",
        70: "seventy",
        80: "eighty",
        90: "ninety",
    }

    if n < 0:
        return "minus " + num_word(-n)
    if n < 20:
        return words_0_19[n]
    if n < 100:
        t = (n // 10) * 10
        r = n % 10
        return tens[t] if r == 0 else f"{tens[t]}-{words_0_19[r]}"
    if n < 1000:
        h = n // 100
        r = n % 100
        return f"{words_0_19[h]} hundred" if r == 0 else f"{words_0_19[h]} hundred {num_word(r)}"
    return str(n)


def make_problem(rng: random.Random) -> tuple[str, str, str]:
    """Return (symbol_expr, word_expr, answer)."""
    op = rng.choices(
        ["+", "-", "*", "/"],
        weights=[0.38, 0.28, 0.22, 0.12],
        k=1,
    )[0]

    if op == "+":
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        ans = a + b
        word = f"{num_word(a)} plus {num_word(b)}"

    elif op == "-":
        a = rng.randint(0, 120)
        b = rng.randint(0, 120)
        # Mostly non-negative results, but include some negative examples.
        if rng.random() < 0.85 and b > a:
            a, b = b, a
        ans = a - b
        word = f"{num_word(a)} minus {num_word(b)}"

    elif op == "*":
        a = rng.randint(0, 20)
        b = rng.randint(0, 20)
        ans = a * b
        word = f"{num_word(a)} times {num_word(b)}"

    else:
        divisor = rng.randint(1, 20)
        quotient = rng.randint(0, 25)
        dividend = divisor * quotient
        a = dividend
        b = divisor
        ans = quotient
        word = f"{num_word(a)} divided by {num_word(b)}"

    return f"{a} {op} {b}", word, str(ans)


def make_doc(rng: random.Random, examples_per_doc: int) -> tuple[str, str]:
    """Create one CPT document and return (text, cpt_type)."""
    doc_type = rng.choices(
        ["equations", "qa", "answer_only", "word_form", "mixed_lesson"],
        weights=[0.30, 0.25, 0.20, 0.15, 0.10],
        k=1,
    )[0]

    lines: list[str] = []

    if doc_type == "equations":
        for _ in range(examples_per_doc):
            expr, _, ans = make_problem(rng)
            lines.append(f"{expr} = {ans}")

    elif doc_type == "qa":
        for _ in range(examples_per_doc):
            expr, _, ans = make_problem(rng)
            lines.append(f"Question: What is {expr}?\nAnswer: {ans}")

    elif doc_type == "answer_only":
        for _ in range(examples_per_doc):
            expr, _, ans = make_problem(rng)
            lines.append(f"Answer only the result:\n{expr}\n{ans}")

    elif doc_type == "word_form":
        for _ in range(examples_per_doc):
            _, word, ans = make_problem(rng)
            lines.append(f"What is {word}?\n{ans}")

    else:
        lines.append("Arithmetic examples:")
        for _ in range(examples_per_doc):
            expr, word, ans = make_problem(rng)
            style = rng.choice(["equation", "question", "word"])
            if style == "equation":
                lines.append(f"{expr} = {ans}")
            elif style == "question":
                lines.append(f"Question: What is {expr}?\nAnswer: {ans}")
            else:
                lines.append(f"What is {word}?\n{ans}")

    return "\n\n".join(lines), doc_type


def write_jsonl_docs(
    out_dir: Path,
    n_train_docs: int,
    n_val_docs: int,
    examples_per_doc: int,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for split, n_docs in [("train", n_train_docs), ("val", n_val_docs)]:
        path = out_dir / f"{split}.jsonl"
        counts: dict[str, int] = {}

        with path.open("w", encoding="utf-8") as f:
            for _ in range(n_docs):
                text, cpt_type = make_doc(rng, examples_per_doc)
                counts[cpt_type] = counts.get(cpt_type, 0) + 1
                row = {
                    "text": text,
                    "source": "synthetic_arithmetic_cpt",
                    "cpt_type": cpt_type,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        log.info(f"Wrote {n_docs:,} {split} CPT docs to {path}")
        log.info(f"{split} cpt_type counts: {counts}")


def find_eos_id(tokenizer: PreTrainedTokenizerFast) -> int:
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)

    candidates = [
        "<|endoftext|>",
        "<|endofturn|>",
        "</s>",
        "<eos>",
    ]
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return int(token_id)

    raise RuntimeError(
        "Could not determine EOS token id. Check tokenizer special tokens."
    )


def tokenize_jsonl_to_bin(
    jsonl_path: Path,
    bin_path: Path,
    tokenizer: PreTrainedTokenizerFast,
    eos_id: int,
) -> None:
    token_ids: list[int] = []
    n_docs = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row["text"]).strip()
            if not text:
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids.extend(ids)
            token_ids.append(eos_id)
            n_docs += 1

    if not token_ids:
        raise RuntimeError(f"No tokens produced from {jsonl_path}")

    max_id = max(token_ids)
    if max_id > np.iinfo(np.uint16).max:
        raise RuntimeError(
            f"Token id {max_id} exceeds uint16 max. Dataset reader expects uint16."
        )

    arr = np.asarray(token_ids, dtype=np.uint16)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(bin_path)

    meta = {
        "dtype": "uint16",
        "n_tokens": int(arr.size),
        "n_docs": n_docs,
        "vocab_size": int(tokenizer.vocab_size),
        "eos_id": eos_id,
        "source": "synthetic_arithmetic_cpt",
    }
    with bin_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(
        f"Tokenized {jsonl_path} -> {bin_path}: "
        f"{arr.size:,} tokens from {n_docs:,} docs"
    )


def prepare_tokenized_cpt(
    cpt_dir: Path,
    tokenized_dir: Path,
    tokenizer_dir: Path,
    n_train_docs: int,
    n_val_docs: int,
    examples_per_doc: int,
    seed: int,
    force: bool,
) -> None:
    train_bin = tokenized_dir / "train.bin"
    val_bin = tokenized_dir / "val.bin"

    if train_bin.exists() and val_bin.exists() and not force:
        log.info(f"Tokenized CPT data already exists at {tokenized_dir}")
        return

    if force:
        shutil.rmtree(cpt_dir, ignore_errors=True)
        shutil.rmtree(tokenized_dir, ignore_errors=True)

    validate_tokenizer(tokenizer_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    eos_id = find_eos_id(tokenizer)
    log.info(f"Using EOS id: {eos_id}")

    write_jsonl_docs(
        out_dir=cpt_dir,
        n_train_docs=n_train_docs,
        n_val_docs=n_val_docs,
        examples_per_doc=examples_per_doc,
        seed=seed,
    )

    tokenize_jsonl_to_bin(cpt_dir / "train.jsonl", train_bin, tokenizer, eos_id)
    tokenize_jsonl_to_bin(cpt_dir / "val.jsonl", val_bin, tokenizer, eos_id)


def load_base_model(base_model: Path, dtype: torch.dtype):
    log.info(f"Loading base model from {base_model}")

    # The project model has a custom from_pretrained implementation. Try the
    # standard torch_dtype keyword first, then fall back for older signatures.
    try:
        return SLMForCausalLM.from_pretrained(str(base_model), torch_dtype=dtype)
    except TypeError:
        try:
            return SLMForCausalLM.from_pretrained(str(base_model), dtype=dtype)
        except TypeError:
            model = SLMForCausalLM.from_pretrained(str(base_model))
            return model.to(dtype=dtype)


def build_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and args.precision == "bf16"
    use_fp16 = has_cuda and args.precision == "fp16"

    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.eval_micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        adam_epsilon=args.eps,
        max_grad_norm=args.gradient_clip_val,
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",
        lr_scheduler_type=args.lr_scheduler,
        bf16=use_bf16,
        fp16=use_fp16,
        torch_compile=args.torch_compile and has_cuda,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=args.log_steps,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.run_name,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Throwaway continued pretraining on synthetic arithmetic."
    )

    parser.add_argument("--base-model", type=Path, default=Path("results/slm-125m/final"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--cpt-dir", type=Path, default=Path("data/cpt_math"))
    parser.add_argument("--tokenized-dir", type=Path, default=Path("data/tokenized_cpt_math"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/slm-125m-cpt-math"))
    parser.add_argument("--force-tokenize", action="store_true")

    parser.add_argument("--n-train-docs", type=int, default=200_000)
    parser.add_argument("--n-val-docs", type=int, default=5_000)
    parser.add_argument("--examples-per-doc", type=int, default=12)

    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)

    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--eval-micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)

    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")

    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")

    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--run-name", type=str, default="slm-125m-cpt-math")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cuda()

    if not args.base_model.exists():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")
    if not args.tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer dir not found: {args.tokenizer_dir}")

    log.info("=== Throwaway CPT Math Experiment ===")
    log.info(f"Base model:     {args.base_model}")
    log.info(f"CPT jsonl dir:  {args.cpt_dir}")
    log.info(f"Tokenized dir:  {args.tokenized_dir}")
    log.info(f"Output dir:     {args.output_dir}")
    log.info(f"Device:         {'cuda' if torch.cuda.is_available() else 'cpu'}")

    prepare_tokenized_cpt(
        cpt_dir=args.cpt_dir,
        tokenized_dir=args.tokenized_dir,
        tokenizer_dir=args.tokenizer_dir,
        n_train_docs=args.n_train_docs,
        n_val_docs=args.n_val_docs,
        examples_per_doc=args.examples_per_doc,
        seed=args.seed,
        force=args.force_tokenize,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() and args.precision == "bf16" else torch.float32
    model = load_base_model(args.base_model, dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    train_ds, val_ds = load_train_val(
        tokenized_dir=args.tokenized_dir,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    global_batch = (
        args.micro_batch_size
        * args.gradient_accumulation_steps
        * int(os.environ.get("WORLD_SIZE", "1"))
    )
    trained_tokens = args.max_steps * global_batch * args.seq_len
    log.info(f"Train examples: {len(train_ds):,}")
    log.info(f"Val examples:   {len(val_ds):,}")
    log.info(f"Estimated CPT trained tokens: {trained_tokens / 1e9:.3f}B")

    training_args = build_args(args, output_dir=args.output_dir)

    if args.report_to == "wandb":
        import wandb

        wandb.init(
            project="slm",
            name=args.run_name,
            config={
                "base_model": str(args.base_model),
                "tokenized_dir": str(args.tokenized_dir),
                "n_train_docs": args.n_train_docs,
                "n_val_docs": args.n_val_docs,
                "examples_per_doc": args.examples_per_doc,
                "max_steps": args.max_steps,
                "lr": args.lr,
                "global_batch": global_batch,
                "estimated_trained_tokens": trained_tokens,
            },
        )

    trainer = SLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[VRAMProbe()],
    )

    log.info("Running baseline CPT eval...")
    baseline = trainer.evaluate()
    log.info(f"Baseline CPT eval: {baseline}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    log.info("Starting CPT math training...")
    trainer.train()

    final_dir = args.output_dir / "final"
    log.info(f"Saving CPT model to {final_dir}")
    trainer.save_model(str(final_dir))

    if hasattr(model, "config"):
        model.config.save_pretrained(str(final_dir))

    shutil.copytree(args.tokenizer_dir, final_dir / "tokenizer", dirs_exist_ok=True)
    log.info(f"Tokenizer copied to {final_dir / 'tokenizer'}")

    log.info("CPT math complete.")
    log.info(f"Next test checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
