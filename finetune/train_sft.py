"""
finetune/train_sft.py
----------------------
Supervised Fine-Tuning using HuggingFace trl SFTTrainer.

Runs two sequential SFT stages:
    Stage 1: Chat SFT on OpenHermes-2.5 (general instruction following)
    Stage 2: Code SFT on Magicoder-OSS-Instruct (coding capability)

Sequential fine-tuning preserves capabilities from earlier stages —
the code SFT uses a lower LR to reduce catastrophic forgetting of
chat capability learned in stage 1.

Answer-only loss: loss is computed only on assistant response tokens,
not on system/user prompt tokens. This focuses the gradient signal
on what the model should generate, not what it reads.

Usage:
    # Chat SFT
    python finetune/train_sft.py --config finetune/configs/sft_chat.yaml

    # Code SFT
    python finetune/train_sft.py --config finetune/configs/sft_code.yaml

    # Multi-GPU
    accelerate launch finetune/train_sft.py --config finetune/configs/sft_chat.yaml

    # Resume
    python finetune/train_sft.py --config finetune/configs/sft_chat.yaml --resume
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

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


def load_dataset_from_jsonl(path: Path):
    from datasets import Dataset
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def build_training_args(cfg: dict, output_dir: Path):
    from transformers import TrainingArguments

    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("epochs", 3),
        max_steps=train_cfg.get("max_steps", -1),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        per_device_train_batch_size=train_cfg["micro_batch_size"],
        per_device_eval_batch_size=train_cfg["micro_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=optim_cfg["lr"],
        weight_decay=optim_cfg.get("weight_decay", 0.01),
        adam_beta1=optim_cfg.get("beta1", 0.9),
        adam_beta2=optim_cfg.get("beta2", 0.98),
        max_grad_norm=train_cfg.get("gradient_clip_val", 1.0),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        bf16=train_cfg.get("precision", "bf16") == "bf16",
        evaluation_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=train_cfg.get("log_steps", 10),
        report_to=train_cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("name", "slm-sft"),
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
    )


def main():
    parser = argparse.ArgumentParser(description="SLM Supervised Fine-Tuning")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to SFT config YAML",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Path to base model (overrides config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["name"]
    output_dir = RESULTS_DIR / model_name
    base_model_path = args.base_model or Path(cfg["model"]["base_model_path"])

    log.info(f"=== SLM Supervised Fine-Tuning ===")
    log.info(f"Config:     {args.config}")
    log.info(f"Name:       {model_name}")
    log.info(f"Base model: {base_model_path}")
    log.info(f"Output:     {output_dir}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from transformers import AutoConfig
    from model import SLMConfig, SLMForCausalLM

    AutoConfig.register("slm", SLMConfig)

    log.info(f"Loading base model from {base_model_path}...")
    model = SLMForCausalLM.from_pretrained(str(base_model_path))
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import PreTrainedTokenizerFast

    tokenizer_path = base_model_path / "tokenizer"
    if not tokenizer_path.exists():
        tokenizer_path = DATA_DIR / "tokenizer"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token = "<PAD>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "<EOS>"
    tokenizer.eos_token_id = 3
    tokenizer.bos_token = "<BOS>"
    tokenizer.bos_token_id = 2

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    train_path = Path(data_cfg["train_path"])
    val_path = Path(data_cfg["val_path"])

    if not train_path.exists():
        log.error(f"Training data not found: {train_path}")
        log.error("Run: python finetune/data/prepare_sft.py")
        sys.exit(1)

    log.info(f"Loading dataset from {train_path}...")
    train_dataset = load_dataset_from_jsonl(train_path)
    val_dataset = load_dataset_from_jsonl(val_path)

    log.info(f"Train: {len(train_dataset):,} examples")
    log.info(f"Val:   {len(val_dataset):,} examples")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = build_training_args(cfg, output_dir)

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg["model"].get("max_seq_length", 2048),
        packing=data_cfg.get("packing", False),
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    if "wandb" in training_args.report_to:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "slm"),
            name=model_name,
            config={
                "base_model": str(base_model_path),
                "n_train": len(train_dataset),
                "n_val": len(val_dataset),
                **cfg,
            },
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting SFT...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("Saving final model...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))

    import shutil
    if tokenizer_path.exists():
        shutil.copytree(tokenizer_path, final_dir / "tokenizer", dirs_exist_ok=True)

    log.info(f"Model saved to {final_dir}")
    log.info("SFT complete.")
    log.info(f"Next step: make dpo" if "chat" in model_name else "make sft-code")


if __name__ == "__main__":
    main()