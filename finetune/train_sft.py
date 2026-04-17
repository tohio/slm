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
not on system/user prompt tokens. Handled natively by SFTConfig via
completion_only_loss=True and response_template="<|assistant|>".
DataCollatorForCompletionOnlyLM was removed in trl 0.17+.

Chat template: formatting uses tokenizer.apply_chat_template() via a
formatting_func passed to SFTTrainer — the same code path as inference.
The "text" field from prepare_sft.py is ignored.

Usage:
    # Chat SFT
    python finetune/train_sft.py --config finetune/configs/sft_chat_125m.yaml

    # Code SFT
    python finetune/train_sft.py --config finetune/configs/sft_code_125m.yaml

    # Multi-GPU
    accelerate launch finetune/train_sft.py --config finetune/configs/sft_chat_125m.yaml

    # Resume
    python finetune/train_sft.py --config finetune/configs/sft_chat_125m.yaml --resume
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


def load_tokenizer(tokenizer_path: Path):
    """
    Load the HuggingFace tokenizer saved by train_tokenizer.py.

    Uses PreTrainedTokenizerFast.from_pretrained() to load the full
    tokenizer config including the baked-in chat_template. Do not
    reconstruct from tokenizer.json directly — that bypasses
    tokenizer_config.json and loses the chat template, causing
    apply_chat_template() to fall back to a generic format.
    """
    from transformers import PreTrainedTokenizerFast

    if not (tokenizer_path / "tokenizer_config.json").exists():
        raise FileNotFoundError(
            f"tokenizer_config.json not found at {tokenizer_path}. "
            f"Retrain the tokenizer: python tokenizer/train_tokenizer.py"
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer at {tokenizer_path} has no chat_template. "
            f"Retrain the tokenizer: python tokenizer/train_tokenizer.py"
        )

    return tokenizer


def make_formatting_func(tokenizer):
    """
    Return a formatting function for SFTTrainer that applies the
    tokenizer's baked-in chat template to each example.

    SFTTrainer calls this function on each example before tokenization.
    Using apply_chat_template() here ensures training format matches
    inference format exactly — the same code path used in chat.py and
    generate.py.

    The "text" field from prepare_sft.py is ignored — formatting always
    goes through the tokenizer's chat template.
    """
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
    return formatting_func


def build_sft_args(cfg: dict, output_dir: Path):
    """
    Build SFTConfig directly — avoids the fragile TrainingArguments.to_dict()
    round-trip which can include keys SFTConfig doesn't accept.

    SFTConfig extends TrainingArguments and adds SFT-specific fields:
    max_seq_length, packing, completion_only_loss, response_template.

    Answer-only loss is handled natively via completion_only_loss=True and
    response_template="<|assistant|>" — DataCollatorForCompletionOnlyLM
    was removed in trl 0.17+.
    """
    from trl import SFTConfig

    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]

    has_cuda = torch.cuda.is_available()
    precision = train_cfg.get("precision", "bf16")
    use_bf16  = has_cuda and precision == "bf16"
    use_fp16  = has_cuda and precision == "fp16"

    return SFTConfig(
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
        run_name=cfg.get("name", "slm-sft"),
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        # SFT-specific fields
        max_length=cfg["model"].get("max_seq_length", 2048),
        packing=cfg["data"].get("packing", False),
        # Answer-only loss — compute loss on assistant responses only.
        # trl 0.29 infers the response template from the chat template automatically.
        # DataCollatorForCompletionOnlyLM was removed in trl 0.17+.
        completion_only_loss=True,
    )


def main():
    parser = argparse.ArgumentParser(description="SLM Supervised Fine-Tuning")
    parser.add_argument("--config",     type=Path, required=True, help="Path to SFT config YAML")
    parser.add_argument("--base-model", type=Path, default=None,  help="Override base model path")
    parser.add_argument("--resume",     action="store_true",       help="Resume from latest checkpoint")
    args = parser.parse_args()

    cfg             = load_config(args.config)
    model_name      = cfg["name"]
    output_dir      = RESULTS_DIR / model_name
    base_model_path = args.base_model or Path(
        os.path.expandvars(cfg["model"]["base_model_path"])
    )

    log.info(f"=== SLM Supervised Fine-Tuning ===")
    log.info(f"Config:     {args.config}")
    log.info(f"Name:       {model_name}")
    log.info(f"Base model: {base_model_path}")
    log.info(f"Output:     {output_dir}")
    log.info(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from transformers import AutoConfig
    from model import SLMConfig, SLMForCausalLM

    AutoConfig.register("slm", SLMConfig)

    log.info(f"Loading base model from {base_model_path}...")
    model    = SLMForCausalLM.from_pretrained(str(base_model_path))
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Use from_pretrained() so tokenizer_config.json is loaded — this is what
    # gives us the baked-in chat_template for apply_chat_template().
    # Fall back to DATA_DIR/tokenizer if not found in the model directory.
    tokenizer_path = base_model_path / "tokenizer"
    if not (tokenizer_path / "tokenizer_config.json").exists():
        log.warning(
            f"tokenizer_config.json not found at {tokenizer_path} — "
            f"falling back to {DATA_DIR / 'tokenizer'}"
        )
        tokenizer_path = DATA_DIR / "tokenizer"

    log.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)
    log.info(f"Vocab size: {tokenizer.vocab_size:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_cfg   = cfg["data"]
    train_path = Path(os.path.expandvars(data_cfg["train_path"]))
    val_path   = Path(os.path.expandvars(data_cfg["val_path"]))

    if not train_path.exists():
        log.error(f"Training data not found: {train_path}")
        log.error("Run: python finetune/data/prepare_sft.py")
        sys.exit(1)

    log.info(f"Loading dataset from {train_path}...")
    train_dataset = load_dataset_from_jsonl(train_path)
    val_dataset   = load_dataset_from_jsonl(val_path)

    # Optionally truncate for mini validation runs
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset   = val_dataset.select(range(min(max_samples // 10, len(val_dataset))))
        log.info(f"Truncated to {max_samples} train / {len(val_dataset)} val (max_samples set)")

    log.info(f"Train: {len(train_dataset):,} examples")
    log.info(f"Val:   {len(val_dataset):,} examples")

    # ── SFT args ──────────────────────────────────────────────────────────────
    sft_args = build_sft_args(cfg, output_dir)
    log.info("Answer-only loss enabled (completion_only_loss=True, response_template='<|assistant|>')")

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        formatting_func=make_formatting_func(tokenizer),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting SFT...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("Saving final model...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))

    if tokenizer_path.exists() and any(tokenizer_path.iterdir()):
        shutil.copytree(tokenizer_path, final_dir / "tokenizer", dirs_exist_ok=True)
        log.info("Tokenizer copied alongside model")
    else:
        log.warning(f"Tokenizer empty or missing at {tokenizer_path} — skipping copy")

    log.info(f"Model saved to {final_dir}")
    log.info("SFT complete.")
    log.info(f"Next step: {'make dpo' if 'chat-code' in model_name else 'make sft-code'}")


if __name__ == "__main__":
    main()