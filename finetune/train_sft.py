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

Answer-only loss:
    Uses trl's native assistant_only_loss=True in SFTConfig. This requires
    the chat template to include {% generation %} / {% endgeneration %} tags
    around assistant responses — these are baked into the tokenizer at
    train_tokenizer.py time.

    SFTTrainer automatically applies the chat template when given a
    conversational dataset (with a "messages" field containing
    role/content message dicts). No formatting_func needed.

Best-checkpoint selection:
    load_best_model_at_end=True with metric_for_best_model="eval_loss".
    SFTTrainer reloads the lowest-eval-loss checkpoint before save_model(),
    so results/<name>/final/ is the best checkpoint, not the last.

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

    Validates that the chat template includes {% generation %} /
    {% endgeneration %} tags required by assistant_only_loss=True.
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

    if "{% generation %}" not in tokenizer.chat_template:
        raise ValueError(
            f"Chat template at {tokenizer_path} is missing {{% generation %}} tags. "
            f"Required for assistant_only_loss=True. "
            f"Retrain the tokenizer: python tokenizer/train_tokenizer.py"
        )

    return tokenizer


def build_sft_args(cfg: dict, output_dir: Path):
    """
    Build SFTConfig for trl 0.17+.

    assistant_only_loss=True computes loss only on assistant response tokens.
    Requires {% generation %} / {% endgeneration %} tags in the chat template.
    SFTTrainer applies the chat template automatically for conversational datasets.

    load_best_model_at_end=True with metric_for_best_model="eval_loss" means
    final/ contains the lowest-eval-loss checkpoint, not the last. Constraints:
        - save_strategy must equal eval_strategy (both "steps")
        - save_steps must be a multiple of eval_steps
        - save_total_limit keeps N recent checkpoints PLUS always the best,
          so disk usage is up to save_total_limit + 1 checkpoints.
    """
    from trl import SFTConfig

    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]

    has_cuda = torch.cuda.is_available()
    precision = train_cfg.get("precision", "bf16")
    use_bf16  = has_cuda and precision == "bf16"
    use_fp16  = has_cuda and precision == "fp16"

    # Warmup: prefer warmup_ratio (scales with total steps under varying
    # epochs/dataset sizes). warmup_steps is honoured if set for backward
    # compatibility, but configs should use warmup_ratio.
    warmup_ratio = train_cfg.get("warmup_ratio", 0.0)
    warmup_steps = train_cfg.get("warmup_steps", 0)
    if warmup_ratio and warmup_steps:
        log.warning(
            "Both warmup_ratio and warmup_steps set — HF Trainer uses "
            "warmup_steps when non-zero and ignores warmup_ratio. "
            "Drop warmup_steps from config to use the ratio."
        )

    save_steps = train_cfg.get("save_steps", 200)
    eval_steps = train_cfg.get("eval_steps", 200)
    if save_steps % eval_steps != 0:
        raise ValueError(
            f"save_steps ({save_steps}) must be a multiple of eval_steps "
            f"({eval_steps}) when load_best_model_at_end=True."
        )

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("epochs", 2),
        max_steps=train_cfg.get("max_steps", -1),
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
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
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=train_cfg.get("log_steps", 10),
        report_to=train_cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("name", "slm-sft"),
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        # SFT-specific
        max_length=cfg["model"].get("max_seq_length", 2048),
        packing=cfg["data"].get("packing", False),
        # Answer-only loss — requires {% generation %} tags in chat template
        assistant_only_loss=True,
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
    # Dataset has a "conversations" field with role/content message dicts.
    # trl's is_conversational() only recognises these field names:
    # prompt, chosen, rejected, completion, messages. Rename "conversations"
    # → "messages" so trl auto-detects the conversational format and applies
    # tokenizer.apply_chat_template() internally. assistant_only_loss=True then
    # uses the {% generation %} tags in the template to mask prompt tokens.
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

    if "conversations" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("conversations", "messages")
        val_dataset   = val_dataset.rename_column("conversations", "messages")

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
    log.info("Answer-only loss enabled (assistant_only_loss=True)")
    log.info("Best-checkpoint selection enabled (metric_for_best_model=eval_loss)")

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting SFT...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save ──────────────────────────────────────────────────────────────────
    # load_best_model_at_end=True means trainer.model is now the best
    # checkpoint by eval_loss, not the last. save_model() persists that.
    log.info("Saving best model (lowest eval_loss)...")
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