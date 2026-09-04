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

Base model: slm-{size}-instruct
Dataset:    pinned external preference contract in configs/dpo_data_sources.yaml

Eval batching:
    `training.eval_micro_batch_size` controls per-device eval batch size
    independently of the training micro-batch. Reference log probabilities
    are precomputed, so evaluation forwards only the policy over each
    chosen/rejected pair. Defaults to half the training micro-batch.

Best-checkpoint selection:
    load_best_model_at_end=True with metric_for_best_model="eval_loss".
    DPO reward margins typically peak early then degrade, so the best
    checkpoint is usually NOT the last. final/ contains the lowest-
    eval-loss checkpoint.

Warmup:
    The YAML stores `warmup_ratio_recipe` (e.g. 0.05 = 5% of total steps).
    We compute the equivalent `warmup_steps` at runtime from the resolved
    total step count and pass that to DPOConfig. We do NOT pass
    warmup_ratio directly. Computing in code preserves the auto-rescaling property
    when GPU count changes — `warmup_steps` baked into YAML would not.

Target library versions are pinned together in requirements.txt.
See requirements.txt for the full compatible stack.

Usage:
    python alignment/train_dpo.py --config alignment/configs/dpo_chat_125m.yaml

    # Multi-GPU
    accelerate launch alignment/train_dpo.py --config alignment/configs/dpo_chat_125m.yaml

    # Resume
    python alignment/train_dpo.py --config alignment/configs/dpo_chat_125m.yaml --resume
"""

import argparse
import hashlib
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

from config.paths import tokenizer_dir, dpo_chat_dir
from config.runtime import configure_torch_runtime


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(path: Path):
    from datasets import Dataset
    records = []
    if not path.exists():
        raise FileNotFoundError(f"DPO dataset not found: {path}")
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"DPO dataset is empty: {path}")
    return Dataset.from_list(records)


def load_tokenizer(tokenizer_path: Path):
    """
    Load the HuggingFace tokenizer saved by train_tokenizer.py.

    Uses PreTrainedTokenizerFast.from_pretrained() to load the full
    tokenizer config including the baked-in chat_template. Do not
    reconstruct from tokenizer.json directly — that bypasses
    tokenizer_config.json and loses the chat template, causing
    DPOTrainer's apply_chat_template() calls to use the wrong format.
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
        steps = int(train_cfg["warmup_steps"])
        log.info(
            f"Warmup: {steps} steps (explicit override; will not auto-rescale "
            f"across GPU counts)"
        )
        return steps

    if "warmup_ratio" in train_cfg:
        log.warning(
            "Config uses deprecated `warmup_ratio` key. Rename to "
            "`warmup_ratio_recipe` (or regenerate the config with "
            "`make config-gen-dpo`). Honouring the value for this run."
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


def build_dpo_args(cfg: dict, output_dir: Path, beta: float, num_train_examples: int):
    """
    Build DPOConfig for the pinned TRL stack.

    DPO-specific fields read from cfg["dpo"] include the beta, loss family,
    divergence, dropout behavior, and reference-log-probability strategy.

    DPO-specific fields read from cfg["model"]:
        max_seq_length      → DPOConfig.max_length (prompt + completion).

    Eval micro-batch:
        Defaults to half the training micro-batch. DPO eval forwards through
        policy + reference for both chosen + rejected, so the activation
        footprint can be spikier than training.

    load_best_model_at_end=True with metric_for_best_model="eval_loss".
    Constraints:
        - save_strategy must equal eval_strategy (both "steps")
        - save_steps must be a multiple of eval_steps
        - save_total_limit keeps N recent checkpoints PLUS always the best,
          so disk usage is up to save_total_limit + 1 checkpoints.
    """
    from trl import DPOConfig

    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]
    dpo_cfg   = cfg["dpo"]
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

    # torch_compile is controlled by YAML. The compile pass can be expensive
    # and is not assumed faster for DPO. Leave disabled unless profiled.
    torch_compile = train_cfg.get("torch_compile", False)

    return DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("epochs", 1),
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
        run_name=cfg.get("name", "slm-dpo"),
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=True,
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        # DPO-specific fields
        beta=beta,
        max_length=cfg["model"].get("max_seq_length", 2048),
        truncation_mode="keep_start",
        padding_free=False,
        dataset_num_proc=data_cfg.get("dataset_num_proc"),
        loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        label_smoothing=float(dpo_cfg.get("label_smoothing", 0.0)),
        f_divergence_type=dpo_cfg.get("f_divergence_type", "reverse_kl"),
        disable_dropout=bool(dpo_cfg.get("disable_dropout", True)),
        precompute_ref_log_probs=bool(
            dpo_cfg.get("precompute_ref_log_probs", True)
        ),
        precompute_ref_batch_size=dpo_cfg.get("precompute_ref_batch_size"),
    )


def _size_from_model_name(model_name: str) -> str:
    name = model_name.removeprefix("slm-")
    return name.split("-")[0]


def _canonical(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _canonical_prompt(messages: list[dict]) -> str:
    normalized = [
        {"role": message["role"], "content": _canonical(message["content"])}
        for message in messages
    ]
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(path: Path) -> str:
    files = [
        path / name
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if (path / name).exists()
    ]
    if not files:
        raise FileNotFoundError(f"No tokenizer files found at {path}")
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def validate_model_tokenizer_contract(model, tokenizer, max_length: int) -> None:
    embedding_rows = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != embedding_rows:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer has {len(tokenizer):,} "
            f"tokens but embeddings have {embedding_rows:,} rows"
        )
    if int(getattr(model.config, "vocab_size", embedding_rows)) != embedding_rows:
        raise ValueError("Model config vocabulary does not match embedding rows")
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
        raise ValueError(f"max_seq_length={max_length} exceeds model context={context}")


def validate_preference_dataset(dataset, label: str) -> set[str]:
    prompts: set[str] = set()
    for index, record in enumerate(dataset):
        prompt = record.get("prompt")
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not isinstance(prompt, list) or not prompt or prompt[-1].get("role") != "user":
            raise ValueError(f"{label}[{index}] has an invalid prompt")
        if (
            not isinstance(chosen, list)
            or len(chosen) != 1
            or chosen[0].get("role") != "assistant"
            or not chosen[0].get("content")
        ):
            raise ValueError(f"{label}[{index}] has an invalid chosen response")
        if (
            not isinstance(rejected, list)
            or len(rejected) != 1
            or rejected[0].get("role") != "assistant"
            or not rejected[0].get("content")
        ):
            raise ValueError(f"{label}[{index}] has an invalid rejected response")
        if _canonical(chosen[0]["content"]) == _canonical(rejected[0]["content"]):
            raise ValueError(f"{label}[{index}] has identical chosen/rejected responses")
        prompts.add(_canonical_prompt(prompt))
    return prompts


def validate_data_manifest(
    manifest_path: Path,
    train_path: Path,
    val_path: Path,
    tokenizer_path: Path,
) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"DPO manifest not found: {manifest_path}. Regenerate preference data."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (train_path, val_path):
        expected = manifest.get("files", {}).get(path.name, {}).get("sha256")
        if not expected or sha256_file(path) != expected:
            raise RuntimeError(f"{path} does not match its DPO manifest")
    expected_tokenizer = manifest.get("contract", {}).get("tokenizer_sha256")
    if expected_tokenizer != tokenizer_fingerprint(tokenizer_path):
        raise RuntimeError(
            "DPO data was filtered with a different tokenizer. Regenerate it "
            "before training."
        )
    if manifest.get("split", {}).get("prompt_overlap") != 0:
        raise RuntimeError("DPO manifest reports train/validation prompt leakage")
    return manifest


def processed_dataset_audit(dataset, original_count: int, label: str) -> dict:
    retained = len(dataset)
    if retained <= 0:
        raise RuntimeError(f"{label} has no DPO pairs after TRL preprocessing")
    required = {"prompt_ids", "chosen_ids", "rejected_ids"}
    if not required <= set(dataset.column_names):
        raise RuntimeError(
            f"{label} preprocessing is missing columns: "
            f"{sorted(required - set(dataset.column_names))}"
        )
    prompt_lengths = [len(ids) for ids in dataset["prompt_ids"]]
    chosen_lengths = [len(ids) for ids in dataset["chosen_ids"]]
    rejected_lengths = [len(ids) for ids in dataset["rejected_ids"]]
    if min(chosen_lengths, default=0) <= 0 or min(rejected_lengths, default=0) <= 0:
        raise RuntimeError(f"{label} contains an empty preference completion")
    return {
        "input_pairs": original_count,
        "retained_pairs": retained,
        "dropped_pairs": original_count - retained,
        "retention_ratio": retained / original_count,
        "prompt_tokens": sum(prompt_lengths),
        "chosen_tokens": sum(chosen_lengths),
        "rejected_tokens": sum(rejected_lengths),
        "max_prompt_tokens": max(prompt_lengths),
        "max_chosen_tokens": max(chosen_lengths),
        "max_rejected_tokens": max(rejected_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="SLM DPO Alignment")
    parser.add_argument("--config",     type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate model, tokenizer, manifest, and preference pairs without training",
    )
    args = parser.parse_args()

    cfg             = load_config(args.config)
    model_name      = cfg["name"]
    output_dir      = dpo_chat_dir(_size_from_model_name(model_name))
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
    configure_torch_runtime(log)
    if cfg.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", str(cfg["wandb_project"]))

    # ── Model ─────────────────────────────────────────────────────────────────
    from transformers import AutoConfig
    from model import SLMConfig, SLMForCausalLM

    AutoConfig.register("slm", SLMConfig)
    model = SLMForCausalLM.from_pretrained(
        str(base_model_path),
        weights_only=True,
    )
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    data_cfg   = cfg["data"]
    train_path = Path(os.path.expandvars(data_cfg["train_path"]))
    val_path   = Path(os.path.expandvars(data_cfg["val_path"]))

    train_dataset = load_dataset_from_jsonl(train_path)
    val_dataset   = load_dataset_from_jsonl(val_path)

    # Optionally truncate for mini validation runs
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset   = val_dataset.select(range(min(max_samples // 10, len(val_dataset))))
        log.info(f"Truncated to {max_samples} train / {len(val_dataset)} val (max_samples set)")

    log.info(f"Train: {len(train_dataset):,} pairs | Val: {len(val_dataset):,} pairs")
    train_prompts = validate_preference_dataset(train_dataset, "train")
    val_prompts = validate_preference_dataset(val_dataset, "validation")
    overlap = train_prompts & val_prompts
    if overlap:
        raise RuntimeError(
            f"DPO train/validation leakage: {len(overlap)} normalized prompts overlap"
        )
    manifest_path = train_path.parent / "manifest.json"
    data_manifest = validate_data_manifest(
        manifest_path,
        train_path,
        val_path,
        tokenizer_path,
    )
    expected_size = _size_from_model_name(model_name)
    if data_manifest.get("contract", {}).get("size") != expected_size:
        raise RuntimeError(
            f"DPO manifest size does not match model size {expected_size!r}"
        )

    # ── DPO args ──────────────────────────────────────────────────────────────
    # Pass num_train_examples so warmup_steps can be derived from the recipe
    # ratio without adding another round-trip after the trainer is built.
    dpo_args = build_dpo_args(cfg, output_dir, beta, num_train_examples=len(train_dataset))
    prepared_max_length = int(
        data_manifest["contract"].get("max_total_tokens", dpo_args.max_length)
    )
    if prepared_max_length > dpo_args.max_length:
        raise RuntimeError(
            f"DPO data permits {prepared_max_length} tokens but trainer max_length "
            f"is only {dpo_args.max_length}"
        )
    log.info(
        f"DPOConfig: max_length={dpo_args.max_length}, beta={beta}"
    )
    log.info(f"torch_compile: {dpo_args.torch_compile}")
    log.info(
        f"Batch sizes: train={dpo_args.per_device_train_batch_size}, "
        f"eval={dpo_args.per_device_eval_batch_size}"
    )
    log.info("Best-checkpoint selection enabled (metric_for_best_model=eval_loss)")

    preflight_audit = {
        "config": str(args.config),
        "base_model": str(base_model_path),
        "tokenizer": str(tokenizer_path),
        "data_manifest": str(manifest_path),
        "train_input_pairs": len(train_dataset),
        "validation_input_pairs": len(val_dataset),
        "prompt_overlap": 0,
        "max_length": dpo_args.max_length,
        "beta": beta,
        "loss_type": dpo_args.loss_type,
        "f_divergence_type": dpo_args.f_divergence_type,
        "precompute_ref_log_probs": dpo_args.precompute_ref_log_probs,
        "source_contract_sha256": data_manifest["contract_sha256"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "dpo_run_audit.json"
    audit_path.write_text(
        json.dumps(preflight_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.preflight_only:
        log.info("DPO preflight passed; no reference pass or optimization was run.")
        return

    # ── DPOTrainer ────────────────────────────────────────────────────────────
    from trl import DPOTrainer

    ref_model = None
    if not dpo_args.precompute_ref_log_probs:
        ref_model = SLMForCausalLM.from_pretrained(
            str(base_model_path),
            weights_only=True,
        )
        for parameter in ref_model.parameters():
            parameter.requires_grad = False

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    train_audit = processed_dataset_audit(
        trainer.train_dataset, len(train_dataset), "train"
    )
    validation_audit = processed_dataset_audit(
        trainer.eval_dataset, len(val_dataset), "validation"
    )
    minimum_retention = float(data_cfg.get("min_retention_ratio", 0.99))
    for label, audit in (("train", train_audit), ("validation", validation_audit)):
        log.info(
            "%s DPO preprocessing: retained %s/%s (%.2f%%)",
            label,
            f'{audit["retained_pairs"]:,}',
            f'{audit["input_pairs"]:,}',
            100.0 * audit["retention_ratio"],
        )
        if audit["retention_ratio"] < minimum_retention:
            raise RuntimeError(
                f"{label} retained only {audit['retention_ratio']:.2%} after "
                f"TRL preprocessing; required >= {minimum_retention:.2%}"
            )
    preflight_audit["train"] = train_audit
    preflight_audit["validation"] = validation_audit
    audit_path.write_text(
        json.dumps(preflight_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting DPO training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save ──────────────────────────────────────────────────────────────────
    # load_best_model_at_end=True means trainer.model is now the best
    # checkpoint by eval_loss, not the last.
    log.info("Saving best model (lowest eval_loss)...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))

    if tokenizer_path.exists() and any(tokenizer_path.iterdir()):
        shutil.copytree(tokenizer_path, final_dir / "tokenizer", dirs_exist_ok=True)
        log.info("Tokenizer copied alongside model")
    else:
        log.warning(f"Tokenizer empty or missing at {tokenizer_path} — skipping copy")
    preflight_audit["best_metric"] = trainer.state.best_metric
    preflight_audit["best_checkpoint"] = trainer.state.best_model_checkpoint
    audit_path.write_text(
        json.dumps(preflight_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(audit_path, final_dir / "dpo_run_audit.json")
    shutil.copy2(manifest_path, final_dir / "dpo_data_manifest.json")

    log.info(f"Model saved to {final_dir}")
    log.info("DPO complete. Next: make eval")


if __name__ == "__main__":
    main()
