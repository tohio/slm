"""
pretrain/train.py
-----------------
Pretraining entry point using HuggingFace Trainer.

Trains SLMForCausalLM from scratch on the tokenized memory-mapped
dataset. Supports single-GPU and multi-GPU training via accelerate.

Usage:
    # Single GPU
    python pretrain/train.py --config pretrain/configs/gpt_125m.yaml

    # Multi-GPU (accelerate)
    accelerate launch pretrain/train.py --config pretrain/configs/gpt_125m.yaml

    # Resume from checkpoint
    python pretrain/train.py --config pretrain/configs/gpt_125m.yaml --resume
"""

import argparse
import logging
import os
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

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_training_args(cfg: dict, output_dir: Path, resume: bool):
    from transformers import TrainingArguments

    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]

    # bf16/fp16 only when CUDA is available — avoids warnings on CPU runs
    # (e.g. make pretrain-mini on a CPU curation instance)
    has_cuda = torch.cuda.is_available()
    precision = train_cfg.get("precision", "bf16")
    use_bf16 = has_cuda and precision == "bf16"
    use_fp16 = has_cuda and precision == "fp16"

    return TrainingArguments(
        output_dir=str(output_dir),

        # Steps
        max_steps=train_cfg["max_steps"],
        warmup_steps=train_cfg.get("warmup_steps", 2000),

        # Batch size
        per_device_train_batch_size=train_cfg["micro_batch_size"],
        per_device_eval_batch_size=train_cfg["micro_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),

        # Optimizer
        learning_rate=optim_cfg["lr"],
        weight_decay=optim_cfg.get("weight_decay", 0.1),
        adam_beta1=optim_cfg.get("beta1", 0.9),
        adam_beta2=optim_cfg.get("beta2", 0.95),
        adam_epsilon=optim_cfg.get("eps", 1e-8),
        max_grad_norm=train_cfg.get("gradient_clip_val", 1.0),

        # LR schedule
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),

        # Precision — only enable on GPU
        bf16=use_bf16,
        fp16=use_fp16,

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
        dataloader_pin_memory=has_cuda,  # pin_memory only useful with CUDA
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

    cfg = load_config(args.config)
    model_name = cfg["name"]
    output_dir = args.results_dir / model_name

    log.info(f"=== SLM Pretraining ===")
    log.info(f"Config:     {args.config}")
    log.info(f"Model:      {model_name}")
    log.info(f"Output:     {output_dir}")
    log.info(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")

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
    model = SLMForCausalLM(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    from pretrain.data.dataset import PretrainingDatasetWithValidation

    bin_path = args.data_dir / "tokenized" / "train.bin"
    seq_len = model_cfg_dict["max_position_embeddings"]

    log.info(f"Loading dataset from {bin_path}")
    splits = PretrainingDatasetWithValidation(
        bin_path=bin_path,
        seq_len=seq_len,
        val_fraction=cfg["data"].get("val_fraction", 0.005),
    )

    log.info(f"Train examples: {len(splits.train):,}")
    log.info(f"Val examples:   {len(splits.val):,}")

    budget = splits.train.token_budget()
    log.info(f"Training tokens: {budget['total_training_tokens'] / 1e9:.2f}B")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = build_training_args(cfg, output_dir, resume=args.resume)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if "wandb" in training_args.report_to:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "slm"),
            name=model_name,
            config={
                "model": model_cfg_dict,
                "training": cfg["training"],
                "optimizer": cfg["optimizer"],
                "n_params": n_params,
                "n_train_tokens": budget["total_training_tokens"],
            },
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits.train,
        eval_dataset=splits.val,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("Saving final model...")
    trainer.save_model(str(output_dir / "final"))
    model_config.save_pretrained(str(output_dir / "final"))

    # Copy tokenizer alongside model — only if tokenizer dir has actual content.
    # Guards against silently propagating an empty tokenizer dir downstream
    # (e.g. if setup-gpu ran before tokenizer was downloaded from S3).
    import shutil
    tokenizer_dir = args.data_dir / "tokenizer"
    if tokenizer_dir.exists() and any(tokenizer_dir.iterdir()):
        shutil.copytree(
            tokenizer_dir,
            output_dir / "final" / "tokenizer",
            dirs_exist_ok=True,
        )
        log.info("Tokenizer copied alongside model")
    else:
        log.warning(
            f"Tokenizer empty or missing at {tokenizer_dir} — skipping copy. "
            f"Run: make tokenizer-download"
        )

    log.info(f"Model saved to {output_dir / 'final'}")
    log.info("Pretraining complete.")
    log.info(f"Next step: make sft")


if __name__ == "__main__":
    main()