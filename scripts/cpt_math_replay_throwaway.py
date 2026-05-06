"""
Throwaway replay-based continued pretraining experiment.

Goal:
    Test whether a tiny amount of synthetic arithmetic helps without causing
    the distribution collapse seen in arithmetic-only CPT.

This script does NOT modify:
    - pretrain/train.py
    - Makefile
    - config/data_mix.py
    - curator/
    - data/tokenized/
    - results/slm-125m/final

It uses:
    replay data: data/tokenized/{train,val}.bin
    math data:   data/tokenized_cpt_math/{train,val}.bin

Training mix:
    replay_ratio = 1 - math_ratio
    math_ratio   = default 0.02
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset, Subset
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


def load_math_cpt_helpers():
    """Import helper functions from scripts/cpt_math_throwaway.py."""
    helper_path = Path(__file__).resolve().with_name("cpt_math_throwaway.py")
    if not helper_path.exists():
        raise FileNotFoundError(
            f"Missing helper script: {helper_path}\n"
            "Expected scripts/cpt_math_throwaway.py to exist."
        )

    spec = importlib.util.spec_from_file_location("cpt_math_throwaway", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {helper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MixedReplayMathDataset(Dataset):
    """Sample mostly replay examples with a small synthetic math fraction."""

    def __init__(
        self,
        replay_ds: Dataset,
        math_ds: Dataset,
        math_ratio: float,
        seed: int = 42,
        virtual_size: int | None = None,
    ):
        if not 0.0 < math_ratio < 1.0:
            raise ValueError(f"math_ratio must be between 0 and 1, got {math_ratio}")

        self.replay_ds = replay_ds
        self.math_ds = math_ds
        self.math_ratio = math_ratio
        self.seed = seed

        # Trainer stops by max_steps, so virtual_size only needs to be large
        # enough for epoch accounting and dataloader construction.
        self.virtual_size = virtual_size or max(len(replay_ds), len(math_ds))

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Deterministic per index so DataLoader worker count does not change
        # which source is sampled for a given index.
        rng = random.Random(self.seed + int(idx))

        if rng.random() < self.math_ratio:
            source = self.math_ds
        else:
            source = self.replay_ds

        sample_idx = rng.randrange(len(source))
        return source[sample_idx]


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


def load_base_model(base_model: Path, dtype: torch.dtype):
    log.info(f"Loading base model from {base_model}")

    try:
        return SLMForCausalLM.from_pretrained(str(base_model), torch_dtype=dtype)
    except TypeError:
        try:
            return SLMForCausalLM.from_pretrained(str(base_model), dtype=dtype)
        except TypeError:
            model = SLMForCausalLM.from_pretrained(str(base_model))
            return model.to(dtype=dtype)


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and args.precision == "bf16"
    use_fp16 = has_cuda and args.precision == "fp16"

    return TrainingArguments(
        output_dir=str(args.output_dir),
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


def _resolve_eos_id(tokenizer: PreTrainedTokenizerFast) -> int | None:
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)

    for token in ["<|endoftext|>", "<|endofturn|>", "</s>", "<eos>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return int(token_id)

    return None


def smoke_generate(
    model,
    tokenizer_dir: Path,
    device: str,
    max_new_tokens: int = 32,
) -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    eos_id = _resolve_eos_id(tokenizer)

    prompts = [
        "2 + 2 =",
        "3 + 4 =",
        "7 - 3 =",
        "Question: What is 2 + 2?\nAnswer:",
        "Question: What is 3 + 4?\nAnswer:",
    ]

    model.eval()
    log.info("=== Raw completion smoke test ===")

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.05,
        }

        if eos_id is not None:
            generate_kwargs["eos_token_id"] = eos_id
            generate_kwargs["pad_token_id"] = eos_id

        with torch.no_grad():
            out = model.generate(**generate_kwargs)

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        log.info("PROMPT: %r", prompt)
        log.info("OUTPUT: %s", text.replace("\n", "\\n"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Throwaway replay + synthetic arithmetic CPT."
    )

    parser.add_argument("--base-model", type=Path, default=Path("results/slm-125m/final"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))

    parser.add_argument("--replay-tokenized-dir", type=Path, default=Path("data/tokenized"))
    parser.add_argument("--math-cpt-dir", type=Path, default=Path("data/cpt_math"))
    parser.add_argument("--math-tokenized-dir", type=Path, default=Path("data/tokenized_cpt_math"))

    parser.add_argument("--output-dir", type=Path, default=Path("results/slm-125m-cpt-math-replay"))
    parser.add_argument("--force-tokenize", action="store_true")

    parser.add_argument("--n-train-docs", type=int, default=50_000)
    parser.add_argument("--n-val-docs", type=int, default=2_000)
    parser.add_argument("--examples-per-doc", type=int, default=12)

    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--math-ratio", type=float, default=0.02)

    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--eval-micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)

    parser.add_argument("--lr", type=float, default=5e-7)
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
    parser.add_argument("--eval-subset-size", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--run-name", type=str, default="slm-125m-cpt-math-replay")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cuda()

    if not args.base_model.exists():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")
    if not args.replay_tokenized_dir.exists():
        raise FileNotFoundError(f"Replay tokenized dir not found: {args.replay_tokenized_dir}")
    if not args.tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer dir not found: {args.tokenizer_dir}")

    validate_tokenizer(args.tokenizer_dir)

    log.info("=== Throwaway Replay CPT Math Experiment ===")
    log.info(f"Base model:          {args.base_model}")
    log.info(f"Replay tokenized:    {args.replay_tokenized_dir}")
    log.info(f"Math CPT jsonl:      {args.math_cpt_dir}")
    log.info(f"Math tokenized:      {args.math_tokenized_dir}")
    log.info(f"Output dir:          {args.output_dir}")
    log.info(f"Math ratio:          {args.math_ratio}")
    log.info(f"Device:              {'cuda' if torch.cuda.is_available() else 'cpu'}")

    helpers = load_math_cpt_helpers()
    helpers.prepare_tokenized_cpt(
        cpt_dir=args.math_cpt_dir,
        tokenized_dir=args.math_tokenized_dir,
        tokenizer_dir=args.tokenizer_dir,
        n_train_docs=args.n_train_docs,
        n_val_docs=args.n_val_docs,
        examples_per_doc=args.examples_per_doc,
        seed=args.seed,
        force=args.force_tokenize,
    )

    replay_train, replay_val = load_train_val(
        tokenized_dir=args.replay_tokenized_dir,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    math_train, math_val = load_train_val(
        tokenized_dir=args.math_tokenized_dir,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    train_ds = MixedReplayMathDataset(
        replay_ds=replay_train,
        math_ds=math_train,
        math_ratio=args.math_ratio,
        seed=args.seed,
    )

    replay_eval = Subset(
        replay_val,
        range(min(args.eval_subset_size, len(replay_val))),
    )
    math_eval = Subset(
        math_val,
        range(min(args.eval_subset_size, len(math_val))),
    )

    log.info(f"Replay eval subset: {len(replay_eval):,} examples")
    log.info(f"Math eval subset:   {len(math_eval):,} examples")

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and args.precision == "bf16"
        else torch.float16
        if torch.cuda.is_available() and args.precision == "fp16"
        else torch.float32
    )

    model = load_base_model(args.base_model, dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = args.micro_batch_size * args.gradient_accumulation_steps * world_size
    total_tokens = args.max_steps * global_batch * args.seq_len
    math_tokens = total_tokens * args.math_ratio
    replay_tokens = total_tokens * (1.0 - args.math_ratio)

    expected_math_sequences = args.max_steps * global_batch * args.math_ratio
    expected_replay_sequences = args.max_steps * global_batch * (1.0 - args.math_ratio)

    log.info(f"Replay train examples:          {len(replay_train):,}")
    log.info(f"Math train examples:            {len(math_train):,}")
    log.info(f"Global batch seqs:              {global_batch:,}")
    log.info(f"Expected replay sequences:      {expected_replay_sequences:,.1f}")
    log.info(f"Expected math sequences:        {expected_math_sequences:,.1f}")
    log.info(f"Estimated total CPT tokens:     {total_tokens / 1e6:.1f}M")
    log.info(f"Estimated replay CPT tokens:    {replay_tokens / 1e6:.1f}M")
    log.info(f"Estimated math CPT tokens:      {math_tokens / 1e6:.1f}M")

    training_args = build_training_args(args)

    if args.report_to == "wandb":
        import wandb

        wandb.init(
            project="slm",
            name=args.run_name,
            config={
                "base_model": str(args.base_model),
                "replay_tokenized_dir": str(args.replay_tokenized_dir),
                "math_tokenized_dir": str(args.math_tokenized_dir),
                "math_ratio": args.math_ratio,
                "max_steps": args.max_steps,
                "lr": args.lr,
                "global_batch": global_batch,
                "estimated_total_tokens": total_tokens,
                "estimated_math_tokens": math_tokens,
                "estimated_replay_tokens": replay_tokens,
                "eval_subset_size": args.eval_subset_size,
            },
        )

    trainer = SLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset={
            "replay": replay_eval,
            "math": math_eval,
        },
        callbacks=[VRAMProbe()],
    )

    log.info("Running baseline eval on replay and math val...")
    baseline = trainer.evaluate()
    log.info(f"Baseline eval: {baseline}")

    log.info("Raw completions before training:")
    smoke_generate(model, args.tokenizer_dir, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    log.info("Starting replay CPT math training...")
    trainer.train()

    log.info("Running final eval on replay and math val...")
    final_metrics = trainer.evaluate()
    log.info(f"Final eval: {final_metrics}")

    log.info("Raw completions after training:")
    smoke_generate(model, args.tokenizer_dir, device=device)

    final_dir = args.output_dir / "final"
    log.info(f"Saving replay CPT model to {final_dir}")
    trainer.save_model(str(final_dir))

    if hasattr(model, "config"):
        model.config.save_pretrained(str(final_dir))

    shutil.copytree(args.tokenizer_dir, final_dir / "tokenizer", dirs_exist_ok=True)
    log.info(f"Tokenizer copied to {final_dir / 'tokenizer'}")

    log.info("Replay CPT math complete.")
    log.info(f"Checkpoint: {final_dir}")


if __name__ == "__main__":
    main()