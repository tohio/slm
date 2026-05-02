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

Packing:
    `data.packing` is read from YAML and passed to SFTConfig. Packing
    concatenates short examples into max_seq_length sequences, dramatically
    improving throughput on conversational datasets where most examples are
    much shorter than the context window. assistant_only_loss is compatible
    with packing in trl 0.28+ — the {% generation %} tags survive packing
    boundaries because the chat template is applied per-message.

Eval batching:
    `training.eval_micro_batch_size` controls per-device eval batch size
    independently of the training micro-batch. Eval doesn't accumulate
    gradients but still materializes full logits for loss, so a larger
    eval batch can spike VRAM. Defaults to half of training micro-batch
    when not specified.

Best-checkpoint selection:
    load_best_model_at_end=True with metric_for_best_model="eval_loss".
    SFTTrainer reloads the lowest-eval-loss checkpoint before save_model(),
    so results/<name>/final/ is the best checkpoint, not the last.

Warmup:
    The YAML stores `warmup_ratio_recipe` (e.g. 0.03 = 3% of total steps).
    We compute the equivalent `warmup_steps` at runtime from the resolved
    total step count and pass that to SFTConfig. We do NOT pass
    warmup_ratio because TRL deprecated it in v5.2 in favour of
    warmup_steps. Computing in code preserves the auto-rescaling property
    when GPU count changes — `warmup_steps` baked into YAML would not.

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
import math
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


def resolve_warmup_steps(train_cfg: dict, num_train_examples: int) -> int:
    """
    Resolve warmup_steps from the recipe ratio and the actual training shape.

    Reads `warmup_ratio_recipe` (the recipe value, written by config_gen)
    and computes the equivalent step count. Honours an explicit
    `warmup_steps` override if present (back-compat for hand-edited
    configs). Refuses to silently accept the deprecated `warmup_ratio` key.

    Returns 0 if no warmup is configured.
    """
    if "warmup_steps" in train_cfg and train_cfg["warmup_steps"]:
        # Explicit override — trust it. Caveat: this won't auto-rescale
        # across GPU counts the way the ratio does. Logged for visibility.
        steps = int(train_cfg["warmup_steps"])
        log.info(
            f"Warmup: {steps} steps (explicit override; will not auto-rescale "
            f"across GPU counts)"
        )
        return steps

    if "warmup_ratio" in train_cfg:
        # Old-style key. TRL deprecated it; we refuse to pass it through to
        # avoid the deprecation warning, but we honour the value the user
        # clearly intended.
        log.warning(
            "Config uses deprecated `warmup_ratio` key. Rename to "
            "`warmup_ratio_recipe` (or regenerate the config with "
            "`make config-gen-sft`). Honouring the value for this run."
        )
        ratio = float(train_cfg["warmup_ratio"])
    elif "warmup_ratio_recipe" in train_cfg:
        ratio = float(train_cfg["warmup_ratio_recipe"])
    else:
        return 0

    if ratio <= 0.0:
        return 0

    # Resolve world size — accelerate/torchrun set this; fallback to 1.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batch = int(train_cfg["micro_batch_size"])
    grad_accum  = int(train_cfg.get("gradient_accumulation_steps", 1))
    epochs      = int(train_cfg.get("epochs", 1))

    global_batch = micro_batch * grad_accum * world_size
    steps_per_epoch = math.ceil(num_train_examples / global_batch)
    total_steps = steps_per_epoch * epochs
    steps = max(1, round(total_steps * ratio))

    log.info(
        f"Warmup: {steps} steps "
        f"({ratio:.1%} of {total_steps} total = "
        f"{steps_per_epoch} steps/epoch × {epochs} epochs; "
        f"global_batch={global_batch}, world_size={world_size})"
    )
    return steps


def build_sft_args(cfg: dict, output_dir: Path, num_train_examples: int):
    """
    Build SFTConfig for trl 0.17+.

    assistant_only_loss=True computes loss only on assistant response tokens.
    Requires {% generation %} / {% endgeneration %} tags in the chat template.
    SFTTrainer applies the chat template automatically for conversational datasets.

    Eval micro-batch:
        Defaults to half the training micro-batch. Eval forward materializes
        full logits (not chunked like train), so the spike at large
        micro_batch can OOM even when training fits.

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
    data_cfg  = cfg["data"]

    has_cuda = torch.cuda.is_available()
    precision = train_cfg.get("precision", "bf16")
    use_bf16  = has_cuda and precision == "bf16"
    use_fp16  = has_cuda and precision == "fp16"

    warmup_steps = resolve_warmup_steps(train_cfg, num_train_examples)

    save_steps = train_cfg.get("save_steps", 200)
    eval_steps = train_cfg.get("eval_steps", 200)
    if save_steps % eval_steps != 0:
        raise ValueError(
            f"save_steps ({save_steps}) must be a multiple of eval_steps "
            f"({eval_steps}) when load_best_model_at_end=True."
        )

    micro_batch = train_cfg["micro_batch_size"]
    eval_micro_batch = train_cfg.get(
        "eval_micro_batch_size",
        max(1, micro_batch // 2),
    )

    # Default packing on (the throughput win on conversational data is large).
    # Stays opt-out via YAML for any dataset where packing is wrong.
    packing = data_cfg.get("packing", True)

    # torch_compile is controlled by YAML. config_gen emits torch_compile:
    # true for production SFT/DPO configs (the SLM forward is compile-clean
    # under pretrain so it's safe here). Hand-written smoke configs may omit
    # the field entirely and the trainer will default to False, since the
    # one-time compile pass (~30-90s) only pays off on full runs.
    torch_compile = train_cfg.get("torch_compile", False)

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("epochs", 2),
        max_steps=train_cfg.get("max_steps", -1),
        warmup_steps=warmup_steps,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=eval_micro_batch,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=optim_cfg["lr"],
        weight_decay=optim_cfg.get("weight_decay", 0.01),
        adam_beta1=optim_cfg.get("beta1", 0.9),
        adam_beta2=optim_cfg.get("beta2", 0.98),
        max_grad_norm=train_cfg.get("gradient_clip_val", 1.0),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        bf16=use_bf16,
        fp16=use_fp16,
        torch_compile=torch_compile,
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
        packing=packing,
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
    # Pass num_train_examples so warmup_steps can be derived from the recipe
    # ratio without adding another round-trip after the trainer is built.
    sft_args = build_sft_args(cfg, output_dir, num_train_examples=len(train_dataset))
    log.info("Answer-only loss enabled (assistant_only_loss=True)")
    log.info(f"Packing: {sft_args.packing}")
    log.info(f"torch_compile: {sft_args.torch_compile}")
    log.info(
        f"Batch sizes: train={sft_args.per_device_train_batch_size}, "
        f"eval={sft_args.per_device_eval_batch_size}"
    )
    log.info("Best-checkpoint selection enabled (metric_for_best_model=eval_loss)")

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    from transformers import Trainer
    from trl import SFTTrainer


    class FastSFTTrainer(SFTTrainer):
        """
        Use TRL SFTTrainer for dataset formatting/collation, but bypass TRL's
        training-time token metrics.

        Reason:
            SLMForCausalLM uses chunked loss during training and returns only
            last-token logits to avoid materializing full [B, T, vocab] logits.
            TRL's SFTTrainer.compute_loss expects full-sequence logits for
            mean_token_accuracy/entropy and will shape-mismatch.

        Eval still runs with model.eval(), so model.py returns full logits and
        eval_loss remains comparable.
        """

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            return Trainer.compute_loss(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )


    trainer = FastSFTTrainer(
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