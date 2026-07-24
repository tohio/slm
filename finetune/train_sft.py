"""
finetune/train_sft.py
----------------------
Supervised Fine-Tuning using HuggingFace trl SFTTrainer.

Runs one SFT stage per invocation. Instruct and code data are prepared from
the pinned external-source contract in configs/sft_data_sources.yaml. Code SFT
is a sibling specialization branch that starts from instruct; it is not the
parent of the general chat/DPO model.

Answer-only loss:
    Uses trl's native assistant_only_loss=True in SFTConfig. This requires
    the chat template to include {% generation %} / {% endgeneration %} tags
    around assistant responses — these are baked into the tokenizer at
    train_tokenizer.py time.

    SFTTrainer automatically applies the chat template when given a
    conversational dataset (with a "messages" field containing
    role/content message dicts). No formatting_func needed.

Packing:
    Disabled. The current custom attention implementation does not enforce
    packed-example boundaries, so enabling packing could leak attention across
    unrelated conversations.

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
    warmup_ratio directly. Computing in code preserves the auto-rescaling property
    when GPU count changes — `warmup_steps` baked into YAML would not.

Usage:
    # Instruct SFT
    python finetune/train_sft.py --config finetune/configs/sft_instruct_125m.yaml

    # Code SFT
    python finetune/train_sft.py --config finetune/configs/sft_code_125m.yaml

    # Multi-GPU
    accelerate launch finetune/train_sft.py --config finetune/configs/sft_instruct_125m.yaml

    # Resume
    python finetune/train_sft.py --config finetune/configs/sft_instruct_125m.yaml --resume
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
from config.paths import tokenizer_dir, sft_instruct_dir, sft_code_dir, BASE_RESULTS_DIR
from config.runtime import configure_torch_runtime

RESULTS_DIR = BASE_RESULTS_DIR


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(path: Path):
    from datasets import Dataset
    records = []
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {path}")
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"SFT dataset is empty: {path}")
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
    if ratio > 1.0:
        raise ValueError(f"warmup ratio must be <= 1.0, got {ratio}")

    # Resolve world size — accelerate/torchrun set this; fallback to 1.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batch = int(train_cfg["micro_batch_size"])
    grad_accum  = int(train_cfg.get("gradient_accumulation_steps", 1))
    epochs = int(train_cfg.get("epochs", 1))

    global_batch = micro_batch * grad_accum * world_size
    steps_per_epoch = math.ceil(num_train_examples / global_batch)
    configured_max_steps = int(train_cfg.get("max_steps", -1))
    total_steps = (
        configured_max_steps
        if configured_max_steps > 0
        else steps_per_epoch * epochs
    )
    steps = max(1, round(total_steps * ratio))

    log.info(
        f"Warmup: {steps} steps "
        f"({ratio:.1%} of {total_steps} total = "
        f"{steps_per_epoch} steps/epoch × {epochs} epochs; "
        f"global_batch={global_batch}, world_size={world_size})"
    )
    return steps


def resolve_total_steps(train_cfg: dict, num_train_examples: int) -> int:
    max_steps = int(train_cfg.get("max_steps", -1))
    if max_steps > 0:
        return max_steps
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = (
        int(train_cfg["micro_batch_size"])
        * int(train_cfg.get("gradient_accumulation_steps", 1))
        * world_size
    )
    return math.ceil(num_train_examples / global_batch) * int(train_cfg.get("epochs", 1))


def build_sft_args(cfg: dict, output_dir: Path, num_train_examples: int):
    """
    Build SFTConfig for the pinned TRL stack.

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

    lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    beta1 = float(optim_cfg.get("beta1", 0.9))
    beta2 = float(optim_cfg.get("beta2", 0.98))

    has_cuda = torch.cuda.is_available()
    precision = train_cfg.get("precision", "bf16")
    use_bf16  = has_cuda and precision == "bf16"
    use_fp16  = has_cuda and precision == "fp16"

    warmup_steps = resolve_warmup_steps(train_cfg, num_train_examples)

    save_steps = int(train_cfg.get("save_steps", 200))
    eval_steps = int(train_cfg.get("eval_steps", 200))
    total_steps = resolve_total_steps(train_cfg, num_train_examples)
    if eval_steps <= 0 or save_steps <= 0:
        raise ValueError("eval_steps and save_steps must both be positive")
    if eval_steps > total_steps or save_steps > total_steps:
        raise ValueError(
            f"Training resolves to {total_steps} optimizer steps, but "
            f"eval_steps={eval_steps} and save_steps={save_steps}. This run "
            "would not produce a comparable evaluation/checkpoint."
        )
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

    # Packing is disabled by default because the current custom attention does
    # not enforce packed-example boundaries. Enable only after safe packed
    # attention / FA2 varlen support is implemented.
    packing = data_cfg.get("packing", False)

    # Length grouping is safe with normal causal attention. Transformers 5.14
    # exposes it through train_sampling_strategy rather than group_by_length.
    group_by_length = train_cfg.get("group_by_length", True)
    length_column_name = train_cfg.get("length_column_name", "length")

    # torch_compile is controlled by YAML. It is disabled by default because it
    # has not been proven faster for SFT in this repo. Enable only after profiling.
    torch_compile = train_cfg.get("torch_compile", False)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("epochs", 2),
        max_steps=train_cfg.get("max_steps", -1),
        warmup_steps=warmup_steps,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=eval_micro_batch,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=lr,
        weight_decay=weight_decay,
        adam_beta1=beta1,
        adam_beta2=beta2,
        max_grad_norm=train_cfg.get("gradient_clip_val", 1.0),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=has_cuda,
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
        train_sampling_strategy="group_by_length" if group_by_length else "random",
        length_column_name=length_column_name,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        # SFT-specific
        max_length=cfg["model"].get("max_seq_length", 2048),
        packing=packing,
        assistant_only_loss=True,
        loss_type=data_cfg.get("loss_type", "chunked_nll"),
        dataset_num_proc=data_cfg.get("dataset_num_proc"),
    )

    return sft_args


def _size_from_model_name(model_name: str) -> str:
    name = model_name.removeprefix("slm-")
    return name.split("-")[0]


def _sft_output_dir(model_name: str) -> Path:
    size = _size_from_model_name(model_name)
    return sft_code_dir(size) if model_name.endswith(("-code", "-chat-code")) else sft_instruct_dir(size)


def validate_model_tokenizer_contract(model, tokenizer, max_length: int) -> None:
    embedding_rows = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != embedding_rows:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer has {len(tokenizer):,} "
            f"tokens but embeddings have {embedding_rows:,} rows"
        )
    config_vocab = int(getattr(model.config, "vocab_size", embedding_rows))
    if config_vocab != embedding_rows:
        raise ValueError(
            f"Model config vocab_size={config_vocab:,} but embeddings have "
            f"{embedding_rows:,} rows"
        )
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        tokenizer_value = getattr(tokenizer, name)
        config_value = getattr(model.config, name, None)
        if tokenizer_value is None:
            raise ValueError(f"Tokenizer has no {name}")
        if config_value is not None and tokenizer_value != config_value:
            raise ValueError(
                f"{name} mismatch: tokenizer={tokenizer_value}, model={config_value}"
            )
    context = int(getattr(model.config, "max_position_embeddings", max_length))
    if max_length > context:
        raise ValueError(
            f"Configured max_seq_length={max_length} exceeds model context={context}"
        )


def validate_conversational_dataset(dataset, label: str) -> None:
    column = "messages" if "messages" in dataset.column_names else "conversations"
    if column not in dataset.column_names:
        raise ValueError(f"{label} data has neither messages nor conversations")
    for index, messages in enumerate(dataset[column]):
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"{label}[{index}] has an invalid conversation")
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if len(roles) != len(messages) or "user" not in roles or roles[-1] != "assistant":
            raise ValueError(
                f"{label}[{index}] must contain a user turn and end with assistant"
            )


def processed_dataset_audit(dataset, original_count: int, label: str) -> dict:
    retained = len(dataset)
    dropped = original_count - retained
    if retained <= 0:
        raise RuntimeError(f"{label} has no examples after TRL preprocessing")
    if "labels" not in dataset.column_names:
        raise RuntimeError(f"{label} preprocessing did not produce assistant-only labels")

    supervised_counts = [
        sum(token != -100 for token in labels)
        for labels in dataset["labels"]
    ]
    if not supervised_counts or min(supervised_counts) <= 0:
        raise RuntimeError(f"{label} contains examples with no supervised assistant tokens")
    return {
        "input_examples": original_count,
        "retained_examples": retained,
        "dropped_examples": dropped,
        "retention_ratio": retained / original_count,
        "supervised_tokens": sum(supervised_counts),
        "min_supervised_tokens": min(supervised_counts),
        "max_supervised_tokens": max(supervised_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="SLM Supervised Fine-Tuning")
    parser.add_argument("--config",     type=Path, required=True, help="Path to SFT config YAML")
    parser.add_argument("--base-model", type=Path, default=None,  help="Override base model path")
    parser.add_argument("--resume",     action="store_true",       help="Resume from latest checkpoint")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run model/tokenizer/data preprocessing audits without optimizing",
    )
    args = parser.parse_args()

    cfg             = load_config(args.config)
    model_name      = cfg["name"]
    output_dir      = _sft_output_dir(model_name)
    base_model_path = args.base_model or Path(
        os.path.expandvars(cfg["model"]["base_model_path"])
    )

    log.info(f"=== SLM Supervised Fine-Tuning ===")
    log.info(f"Config:     {args.config}")
    log.info(f"Name:       {model_name}")
    log.info(f"Base model: {base_model_path}")
    log.info(f"Output:     {output_dir}")
    log.info(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
    configure_torch_runtime(log)
    if cfg.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", str(cfg["wandb_project"]))

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
            f"falling back to target-scoped tokenizer"
        )
        size = _size_from_model_name(model_name)
        if size not in {"mini", "125m", "350m", "1b"}:
            raise ValueError(f"Cannot infer tokenizer size from model name {model_name!r}")
        tokenizer_path = tokenizer_dir(size)

    log.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)
    log.info(f"Vocab size: {tokenizer.vocab_size:,}")
    validate_model_tokenizer_contract(
        model,
        tokenizer,
        int(cfg["model"].get("max_seq_length", 2048)),
    )

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

    log.info(f"Loading dataset from {train_path}...")
    train_dataset = load_dataset_from_jsonl(train_path)
    val_dataset   = load_dataset_from_jsonl(val_path)
    validate_conversational_dataset(train_dataset, "train")
    validate_conversational_dataset(val_dataset, "validation")

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
    raw_train_count = len(train_dataset)
    raw_val_count = len(val_dataset)

    # ── SFT args ──────────────────────────────────────────────────────────────
    # Pass num_train_examples so warmup_steps can be derived from the recipe
    # ratio without adding another round-trip after the trainer is built.
    sft_args = build_sft_args(cfg, output_dir, num_train_examples=len(train_dataset))
    log.info("Answer-only loss enabled (assistant_only_loss=True)")
    log.info(f"Packing: {sft_args.packing}")
    log.info(f"torch_compile: {sft_args.torch_compile}")
    log.info(f"train_sampling_strategy: {sft_args.train_sampling_strategy}")
    log.info(f"length_column_name: {getattr(sft_args, 'length_column_name', None)}")
    log.info(
        f"Batch sizes: train={sft_args.per_device_train_batch_size}, "
        f"eval={sft_args.per_device_eval_batch_size}"
    )
    log.info("Best-checkpoint selection enabled (metric_for_best_model=eval_loss)")

    # ── SFTTrainer ────────────────────────────────────────────────────────────────
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    train_audit = processed_dataset_audit(
        trainer.train_dataset, raw_train_count, "train"
    )
    val_audit = processed_dataset_audit(
        trainer.eval_dataset, raw_val_count, "validation"
    )
    minimum_retention = float(data_cfg.get("min_retention_ratio", 0.90))
    for label, audit in (("train", train_audit), ("validation", val_audit)):
        log.info(
            "%s preprocessing: retained %,d/%,d (%.2f%%), supervised tokens=%,d",
            label,
            audit["retained_examples"],
            audit["input_examples"],
            100.0 * audit["retention_ratio"],
            audit["supervised_tokens"],
        )
        if audit["retention_ratio"] < minimum_retention:
            raise RuntimeError(
                f"{label} retained only {audit['retention_ratio']:.2%} after "
                f"tokenization/truncation; required >= {minimum_retention:.2%}. "
                "Revise the source selection or max sequence length before training."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = train_path.parent / "manifest.json"
    run_audit = {
        "config": str(args.config),
        "base_model": str(base_model_path),
        "tokenizer": str(tokenizer_path),
        "loss_type": sft_args.loss_type,
        "max_length": sft_args.max_length,
        "train": train_audit,
        "validation": val_audit,
        "data_manifest": str(data_manifest) if data_manifest.exists() else None,
    }
    (output_dir / "sft_run_audit.json").write_text(
        json.dumps(run_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.preflight_only:
        log.info("SFT preflight passed; no optimization was run.")
        return
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
    shutil.copy2(output_dir / "sft_run_audit.json", final_dir / "sft_run_audit.json")
    if data_manifest.exists():
        shutil.copy2(data_manifest, final_dir / "sft_data_manifest.json")

    log.info(f"Model saved to {final_dir}")
    log.info("SFT complete.")
    if model_name.endswith("-code") or "chat-code" in model_name:
        log.info("SFT code branch complete. Next optional step: make eval-code or prepare code-specific alignment.")
    else:
        log.info("SFT instruct branch complete. Next steps: make dpo for chat alignment or make sft-code for code specialization.")


if __name__ == "__main__":
    main()
