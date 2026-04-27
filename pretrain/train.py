"""
pretrain/train.py
-----------------
Pretraining entry point using HuggingFace Trainer.

Trains SLMForCausalLM from scratch on the tokenized memory-mapped
datasets. Supports single-GPU and multi-GPU training via accelerate.

Usage:
    # Single GPU
    python pretrain/train.py --config pretrain/configs/gpt_125m.yaml

    # Multi-GPU (accelerate)
    accelerate launch pretrain/train.py --config pretrain/configs/gpt_125m.yaml

    # Resume from checkpoint
    python pretrain/train.py --config pretrain/configs/gpt_125m.yaml --resume
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR    = Path(os.environ.get("DATA_DIR", "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


# ── GPU performance setup ─────────────────────────────────────────────────────
# Apply GPU-friendly defaults at import time so they take effect before any
# tensors or modules are created. These are no-ops on CPU.
#
# - TF32: H100/H200/B200 lose nothing in quality from TF32 matmul; the FP32
#   fallback is roughly 8× slower for no benefit.
# - SDPA backends: explicitly prefer FlashAttention and memory-efficient
#   kernels over the math fallback. The model's GQA module uses
#   F.scaled_dot_product_attention, so this is the right place to bias
#   PyTorch's kernel selector.
def _configure_cuda_for_performance() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # SDPA backend hints — guard with hasattr because these moved between
    # PyTorch minor versions and we don't want a crash on older builds.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        # Disable the slow math fallback so we notice (via a kernel error)
        # if the fast paths can't be used, instead of silently degrading.
        torch.backends.cuda.enable_math_sdp(False)


_configure_cuda_for_performance()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_tokenizer(tokenizer_dir: Path) -> None:
    """
    Verify the tokenizer directory is complete AND parseable before training.

    Fails hard rather than silently continuing — a missing or corrupt
    tokenizer at this point means the saved checkpoint will be unusable
    by train_sft.py, which requires tokenizer_config.json to load the
    chat_template via PreTrainedTokenizerFast.from_pretrained().

    Checks:
        - tokenizer_dir exists and is non-empty
        - tokenizer_config.json is present and parses as valid JSON
        - slm_tokenizer.json is present and parses as valid JSON

    Args:
        tokenizer_dir: Path to the tokenizer directory.

    Raises:
        RuntimeError: If any required file is missing or unparseable.
    """
    if not tokenizer_dir.exists() or not any(tokenizer_dir.iterdir()):
        raise RuntimeError(
            f"Tokenizer directory missing or empty: {tokenizer_dir}\n"
            f"Run: make tokenizer-download\n"
            f"Or retrain: make tokenizer && make tokenizer-upload"
        )

    required_files = {
        "tokenizer_config.json": (
            "Contains the baked-in chat_template required by train_sft.py. "
            "Retrain the tokenizer: make tokenizer"
        ),
        "slm_tokenizer.json": (
            "Raw BPE tokenizer required by tokenize_data.py. "
            "Retrain the tokenizer: make tokenizer"
        ),
    }

    for filename, hint in required_files.items():
        path = tokenizer_dir / filename
        if not path.exists():
            raise RuntimeError(
                f"Missing tokenizer file: {path}\n{hint}"
            )
        # Parse each JSON to catch truncated or otherwise-corrupt files.
        # A zero-byte or truncated tokenizer_config.json would pass a naive
        # existence check and then crash much later in training.
        try:
            with open(path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Tokenizer file is not valid JSON: {path}\n"
                f"  Error: {e}\n"
                f"  Hint: {hint}"
            ) from e

    log.info(f"Tokenizer validated at {tokenizer_dir}")


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """
    Return the latest checkpoint in output_dir, or None if none exist.

    HF Trainer stores checkpoints as `checkpoint-<step>` subdirectories.
    The latest is the one with the highest step number.
    """
    if not output_dir.exists():
        return None
    candidates = [
        p for p in output_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    if not candidates:
        return None
    # Parse the step from the name. Skip any checkpoint-* dirs with a
    # non-numeric suffix — they shouldn't exist but robust is cheap here.
    numbered = []
    for p in candidates:
        try:
            step = int(p.name.split("-", 1)[1])
            numbered.append((step, p))
        except (IndexError, ValueError):
            continue
    if not numbered:
        return None
    numbered.sort()
    return numbered[-1][1]


def build_training_args(cfg: dict, output_dir: Path, resume: bool):
    from transformers import TrainingArguments

    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]

    # bf16/fp16 only when CUDA is available — avoids warnings on CPU runs
    # (e.g. make pretrain-mini on a CPU curation instance)
    has_cuda = torch.cuda.is_available()
    precision = train_cfg.get("precision", "bf16")
    use_bf16  = has_cuda and precision == "bf16"
    use_fp16  = has_cuda and precision == "fp16"

    # torch.compile — defaults ON because graph compilation is essentially
    # free performance for this workload: static shapes (fixed seq_len, fixed
    # micro_batch), no dynamic control flow, no custom kernels. The 1-2 min
    # compilation cost on the first step amortizes to <1% overhead on a
    # multi-hour run and buys ~1.3-1.5x throughput on H100/H200/B200.
    #
    # Opt out by setting `torch_compile: false` in the training config if
    # you hit a kernel issue or are debugging the model.
    torch_compile = bool(train_cfg.get("torch_compile", True)) and has_cuda

    return TrainingArguments(
        output_dir=str(output_dir),

        # Steps
        max_steps=train_cfg["max_steps"],
        warmup_steps=train_cfg.get("warmup_steps", 2000),

        # Batch size
        per_device_train_batch_size=train_cfg["micro_batch_size"],
        per_device_eval_batch_size=train_cfg["micro_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),

        # Optimizer — fused AdamW is a free 5-10% on modern GPUs (H100+).
        # Falls back to the non-fused implementation automatically on CPU.
        learning_rate=optim_cfg["lr"],
        weight_decay=optim_cfg.get("weight_decay", 0.1),
        adam_beta1=optim_cfg.get("beta1", 0.9),
        adam_beta2=optim_cfg.get("beta2", 0.95),
        adam_epsilon=optim_cfg.get("eps", 1e-8),
        max_grad_norm=train_cfg.get("gradient_clip_val", 1.0),
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",

        # LR schedule
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),

        # Precision — only enable on GPU
        bf16=use_bf16,
        fp16=use_fp16,

        # torch.compile — defaults on (see comment above)
        torch_compile=torch_compile,
        torch_compile_backend=train_cfg.get("torch_compile_backend", "inductor"),
        torch_compile_mode=train_cfg.get("torch_compile_mode", "default"),

        # Evaluation — eval_strategy replaces evaluation_strategy in transformers v5
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 1000),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 1000),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=False,

        # Logging
        logging_strategy="steps",
        logging_steps=train_cfg.get("log_steps", 10),
        report_to=train_cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("name", "slm-pretrain"),

        # Misc
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),

        # Gradient checkpointing
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),

        # DDP — disable unused parameter scan, it adds overhead every step
        # and is never needed for this architecture (no flow control / optional layers)
        ddp_find_unused_parameters=False,
    )


def main():
    parser = argparse.ArgumentParser(description="SLM Pretraining")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Data directory override",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Results directory override",
    )
    args = parser.parse_args()

    cfg        = load_config(args.config)
    model_name = cfg["name"]
    output_dir = args.results_dir / model_name

    log.info(f"=== SLM Pretraining ===")
    log.info(f"Config:     {args.config}")
    log.info(f"Model:      {model_name}")
    log.info(f"Output:     {output_dir}")
    log.info(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        log.info(f"GPU:        {torch.cuda.get_device_name(0)} "
                 f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")

    # ── Resume checkpoint discovery ───────────────────────────────────────────
    # Log which checkpoint we're resuming from BEFORE the model is built, so
    # if anything goes wrong the user can tell at a glance which state was
    # loaded. HF Trainer does log this internally, but it's buried in noise.
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = _find_latest_checkpoint(output_dir)
        if resume_checkpoint is None:
            log.warning(
                f"--resume passed but no checkpoint found in {output_dir}. "
                f"Training will start from scratch."
            )
        else:
            log.info(f"Resuming from checkpoint: {resume_checkpoint}")

    # ── Validate tokenizer before starting ────────────────────────────────────
    # Fail early — a missing or corrupt tokenizer discovered after hours of
    # training produces an unusable checkpoint that can't be loaded by
    # train_sft.py.
    tokenizer_dir = args.data_dir / "tokenizer"
    validate_tokenizer(tokenizer_dir)

    # ── Model ─────────────────────────────────────────────────────────────────
    from model import SLMConfig, SLMForCausalLM

    model_cfg_dict = cfg["model"]
    model_config = SLMConfig(
        vocab_size=model_cfg_dict["vocab_size"],
        hidden_size=model_cfg_dict["hidden_size"],
        intermediate_size=model_cfg_dict.get("intermediate_size"),
        num_hidden_layers=model_cfg_dict["num_hidden_layers"],
        num_attention_heads=model_cfg_dict["num_attention_heads"],
        num_key_value_heads=model_cfg_dict["num_key_value_heads"],
        max_position_embeddings=model_cfg_dict["max_position_embeddings"],
        rope_theta=model_cfg_dict.get("rope_theta", 10000.0),
        rms_norm_eps=model_cfg_dict.get("rms_norm_eps", 1e-5),
        initializer_range=model_cfg_dict.get("initializer_range", 0.02),
        tie_word_embeddings=model_cfg_dict.get("tie_word_embeddings", True),
    )

    log.info(f"Initializing model from scratch...")
    model    = SLMForCausalLM(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    from pretrain.data.dataset import load_train_val

    tokenized_dir = args.data_dir / "tokenized"
    seq_len = model_cfg_dict["max_position_embeddings"]

    log.info(f"Loading datasets from {tokenized_dir}")
    train_ds, val_ds = load_train_val(tokenized_dir=tokenized_dir, seq_len=seq_len)

    log.info(f"Train examples: {len(train_ds):,}")
    log.info(f"Val examples:   {len(val_ds):,}")

    budget = train_ds.token_budget()
    log.info(f"Training tokens: {budget['total_training_tokens'] / 1e9:.2f}B")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = build_training_args(cfg, output_dir, resume=args.resume)

    # Log throughput-relevant settings at INFO so they appear above the
    # training noise. Useful for spot-checking auto-generated configs.
    log.info(
        f"Throughput knobs: "
        f"micro_batch={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, "
        f"bf16={training_args.bf16}, "
        f"optim={training_args.optim}, "
        f"compile={training_args.torch_compile}, "
        f"grad_ckpt={training_args.gradient_checkpointing}"
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    if "wandb" in training_args.report_to:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "slm"),
            name=model_name,
            config={
                "model":           model_cfg_dict,
                "training":        cfg["training"],
                "optimizer":       cfg["optimizer"],
                "n_params":        n_params,
                "n_train_tokens":  budget["total_training_tokens"],
                "n_val_examples":  len(val_ds),
            },
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # ── Baseline eval ─────────────────────────────────────────────────────────
    # Run eval before training begins so the W&B eval curve starts from
    # random-init loss. Without this, the first eval point is ~500-1000
    # steps in — you lose the "loss at step 0 vs step N" comparison that
    # confirms the model is actually learning. Skipped when resuming since
    # the baseline already exists from the original run.
    if not args.resume:
        log.info("Running baseline eval before training (step 0)...")
        baseline = trainer.evaluate()
        log.info(f"Baseline eval: {baseline}")

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint if args.resume else None)

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("Saving final model...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    model_config.save_pretrained(str(final_dir))

    # Copy tokenizer alongside model.
    # tokenizer_dir was already validated above — if we reach this point it is
    # complete, parseable, and contains tokenizer_config.json. No re-check.
    shutil.copytree(tokenizer_dir, final_dir / "tokenizer", dirs_exist_ok=True)
    log.info(f"Tokenizer copied to {final_dir / 'tokenizer'}")

    log.info(f"Model saved to {final_dir}")
    log.info("Pretraining complete.")
    log.info("Next step: make sft")


if __name__ == "__main__":
    main()