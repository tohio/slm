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

Base model: slm-{size}-chat-code (after both SFT stages)
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


def build_dpo_args(cfg: dict, output_dir: Path, beta: float):
    """
    Build DPOConfig directly — avoids the fragile TrainingArguments.to_dict()
    round-trip which can include keys DPOConfig doesn't accept.

    DPOConfig extends TrainingArguments and adds DPO-specific fields:
    beta, max_length, max_prompt_length.
    """
    from trl import DPOConfig

    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]

    has_cuda = torch.cuda.is_available()
    precision = train_cfg.get("precision", "bf16")
    use_bf16  = has_cuda and precision == "bf16"
    use_fp16  = has_cuda and precision == "fp16"

    return DPOConfig(
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
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=False,
        logging_steps=train_cfg.get("log_steps", 10),
        report_to=train_cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("name", "slm-dpo"),
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        # DPO-specific fields
        beta=beta,
        max_length=cfg["model"].get("max_seq_length", 2048),
    )


def main():
    parser = argparse.ArgumentParser(description="SLM DPO Alignment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name      = cfg["name"]
    output_dir      = RESULTS_DIR / model_name
    # Expand $DATA_DIR/$RESULTS_DIR env vars in config paths
    base_model_path = args.base_model or Path(
        os.path.expandvars(cfg["model"]["base_model_path"])
    )
    beta = cfg["dpo"].get("beta", 0.1)

    log.info(f"=== SLM DPO Alignment ===")
    log.info(f"Config:     {args.config}")
    log.info(f"Name:       {model_name}")
    log.info(f"Base model: {base_model_path}")
    log.info(f"Beta:       {beta}")
    log.info(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from transformers import AutoConfig
    from model import SLMConfig, SLMForCausalLM

    AutoConfig.register("slm", SLMConfig)
    model = SLMForCausalLM.from_pretrained(str(base_model_path))
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load a separate frozen copy as the reference model.
    # DPOTrainer with ref_model=None tries to load from checkpoint using
    # config.architectures[0] which looks up SLMForCausalLM in transformers
    # (not our local module) and fails. Passing an explicit ref_model avoids this.
    ref_model = SLMForCausalLM.from_pretrained(str(base_model_path))
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import PreTrainedTokenizerFast

    tokenizer_path = base_model_path / "tokenizer"
    if not tokenizer_path.exists():
        tokenizer_path = DATA_DIR / "tokenizer"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token    = "<PAD>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token    = "<EOS>"
    tokenizer.eos_token_id = 3
    tokenizer.bos_token    = "<BOS>"
    tokenizer.bos_token_id = 2

    # Set chat template so DPOTrainer can use apply_chat_template for
    # conversational format data (list of message dicts). This ensures
    # consistent tokenization at prompt/response boundaries.
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}<|endofturn|>"
            "{% elif message['role'] == 'user' %}<|user|>{{ message['content'] }}<|endofturn|>"
            "{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}<|endofturn|>"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}"
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_cfg   = cfg["data"]
    # Expand $DATA_DIR env var in dataset paths
    train_path = Path(os.path.expandvars(data_cfg["train_path"]))
    val_path   = Path(os.path.expandvars(data_cfg["val_path"]))

    if not train_path.exists():
        log.error(f"DPO data not found: {train_path}")
        log.error("Run: python alignment/data/prepare_dpo.py")
        sys.exit(1)

    train_dataset = load_dataset_from_jsonl(train_path)
    val_dataset   = load_dataset_from_jsonl(val_path)

    # Optionally truncate for mini validation runs
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset   = val_dataset.select(range(min(max_samples // 10, len(val_dataset))))
        log.info(f"Truncated to {max_samples} train / {len(val_dataset)} val (max_samples set)")

    log.info(f"Train: {len(train_dataset):,} pairs | Val: {len(val_dataset):,} pairs")

    # ── DPO args ──────────────────────────────────────────────────────────────
    # Build DPOConfig directly — avoids fragile TrainingArguments.to_dict() round-trip
    dpo_args = build_dpo_args(cfg, output_dir, beta)

    # ── DPOTrainer ────────────────────────────────────────────────────────────
    from trl import DPOTrainer

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,    # explicit frozen reference model — avoids transformers lookup
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
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