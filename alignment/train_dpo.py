"""
alignment/train_dpo.py
-----------------------
Direct Preference Optimization (DPO) using HuggingFace trl DPOTrainer.

DPO optimizes the model to prefer chosen responses over rejected responses
without a separate reward model. It treats the SFT model as an implicit
reference policy and directly updates the model weights using a
classification-style loss on preference pairs.

DPO loss:
    L = -E[log σ(β * (log π(y_w|x) - log π_ref(y_w|x))
                  - β * (log π(y_l|x) - log π_ref(y_l|x)))]

where:
    π     = policy model (being trained)
    π_ref = reference model (frozen SFT checkpoint)
    y_w   = chosen (preferred) response
    y_l   = rejected response
    β     = temperature controlling deviation from reference policy

Base model: slm-125m-chat-code (after both SFT stages)
Dataset:    Blended hh-rlhf + orca_dpo_pairs + dpo-mix-7k

Usage:
    python alignment/train_dpo.py --config alignment/configs/dpo_125m.yaml

    # Multi-GPU
    accelerate launch alignment/train_dpo.py --config alignment/configs/dpo_125m.yaml

    # Resume
    python alignment/train_dpo.py --config alignment/configs/dpo_125m.yaml --resume
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
        num_train_epochs=train_cfg.get("epochs", 1),
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
        run_name=cfg.get("name", "slm-dpo"),
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
    )


def main():
    parser = argparse.ArgumentParser(description="SLM DPO Alignment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["name"]
    output_dir = RESULTS_DIR / model_name
    base_model_path = args.base_model or Path(cfg["model"]["base_model_path"])
    beta = cfg["dpo"].get("beta", 0.1)

    log.info(f"=== SLM DPO Alignment ===")
    log.info(f"Config:     {args.config}")
    log.info(f"Name:       {model_name}")
    log.info(f"Base model: {base_model_path}")
    log.info(f"Beta:       {beta}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from transformers import AutoConfig
    from model import SLMConfig, SLMForCausalLM

    AutoConfig.register("slm", SLMConfig)
    model = SLMForCausalLM.from_pretrained(str(base_model_path))
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    train_path = Path(data_cfg["train_path"])
    val_path = Path(data_cfg["val_path"])

    if not train_path.exists():
        log.error(f"DPO data not found: {train_path}")
        log.error("Run: python alignment/data/prepare_dpo.py")
        sys.exit(1)

    train_dataset = load_dataset_from_jsonl(train_path)
    val_dataset = load_dataset_from_jsonl(val_path)
    log.info(f"Train: {len(train_dataset):,} pairs | Val: {len(val_dataset):,} pairs")

    # ── DPOTrainer ────────────────────────────────────────────────────────────
    from trl import DPOTrainer

    training_args = build_training_args(cfg, output_dir)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,         # DPOTrainer creates a frozen copy automatically
        args=training_args,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_length=cfg["model"].get("max_seq_length", 2048),
        max_prompt_length=cfg["dpo"].get("max_prompt_length", 512),
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    if "wandb" in training_args.report_to:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "slm"),
            name=model_name,
            config={"base_model": str(base_model_path), "beta": beta, **cfg},
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting DPO training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save ──────────────────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))

    import shutil
    if tokenizer_path.exists():
        shutil.copytree(tokenizer_path, final_dir / "tokenizer", dirs_exist_ok=True)

    log.info(f"Model saved to {final_dir}")
    log.info("DPO complete. Next: make eval")


if __name__ == "__main__":
    main()