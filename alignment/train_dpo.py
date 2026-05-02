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

Eval batching:
    `training.eval_micro_batch_size` controls per-device eval batch size
    independently of the training micro-batch. DPO eval forwards through
    both policy and reference for each chosen/rejected pair, so the
    activation footprint at eval can spike higher than train. Defaults
    to half the training micro-batch.

Best-checkpoint selection:
    load_best_model_at_end=True with metric_for_best_model="eval_loss".
    DPO reward margins typically peak early then degrade, so the best
    checkpoint is usually NOT the last. final/ contains the lowest-
    eval-loss checkpoint.

Warmup:
    The YAML stores `warmup_ratio_recipe` (e.g. 0.05 = 5% of total steps).
    We compute the equivalent `warmup_steps` at runtime from the resolved
    total step count and pass that to DPOConfig. We do NOT pass
    warmup_ratio because TRL deprecated it in v5.2 in favour of
    warmup_steps. Computing in code preserves the auto-rescaling property
    when GPU count changes — `warmup_steps` baked into YAML would not.

Target library versions: trl 0.28.x, transformers 5.5.x.
See requirements.txt for the full compatible stack.

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


def build_dpo_args(cfg: dict, output_dir: Path, beta: float, num_train_examples: int):
    """
    Build DPOConfig for trl 0.29.x.

    DPO-specific fields read from cfg["dpo"]:
        beta                — KL penalty temperature
        max_prompt_length   — per-prompt truncation applied by trl's data
                              collator. We also pre-filter overlong pairs
                              in prepare_dpo.py as defense in depth (the
                              pre-filter uses the real SLM tokenizer, so
                              counts are exact and the filtered dataset can
                              serve all model sizes).

    DPO-specific fields read from cfg["model"]:
        max_seq_length      → DPOConfig.max_length (prompt + completion).

    Eval micro-batch:
        Defaults to half the training micro-batch. DPO eval forwards through
        policy + reference for both chosen + rejected — the activation
        footprint can be spikier than training, where we get away with
        chunked loss in the SLM forward.

    NOTE: (max_prompt_length was removed in trl 0.29 — prompt-length capping
        is now entirely handled by the pre-filter in prepare_dpo.py, which
        uses the real SLM tokenizer so counts are exact.)

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

    # torch_compile is controlled by YAML. config_gen emits torch_compile:
    # true for production DPO configs. Hand-written smoke configs may omit
    # the field and the trainer defaults to False, since the one-time
    # compile pass (~30-90s) only pays off on full runs.
    torch_compile = train_cfg.get("torch_compile", False)

    return DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("epochs", 1),
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
        run_name=cfg.get("name", "slm-dpo"),
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        # DPO-specific fields
        beta=beta,
        max_length=cfg["model"].get("max_seq_length", 2048),
        # max_prompt_length was removed in trl <=0.29. Prompt-length capping
        # is now enforced upstream in prepare_dpo.py via max_total_tokens.
    )


def main():
    parser = argparse.ArgumentParser(description="SLM DPO Alignment")
    parser.add_argument("--config",     type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    cfg             = load_config(args.config)
    model_name      = cfg["name"]
    output_dir      = RESULTS_DIR / model_name
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
    # trl 0.29's DPOTrainer does `model.warnings_issued["estimate_tokens"] = True`
    # during __init__ to suppress a transformers FLOPs-estimation warning for
    # DPO batches (which contain prompt_input_ids, not input_ids).
    # `warnings_issued` used to be set by transformers' PreTrainedModel.__init__,
    # but transformers 5.x no longer initialises it — the dict doesn't exist
    # and the assignment raises AttributeError. Pre-seed it here.
    #
    # This is NOT related to SLMForCausalLM's __getattr__ (it does not override
    # one). It's a transformers-5 / trl-0.29 compat gap that would affect any
    # model loaded under this stack.
    #
    # trl removed this access in PR #4960 (post-0.29, pre-1.0). When we upgrade
    # past 0.29 verify the hack is no longer needed and remove it.
    model.warnings_issued = {}
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ref_model = SLMForCausalLM.from_pretrained(str(base_model_path))
    ref_model.warnings_issued = {}
    for p in ref_model.parameters():
        p.requires_grad = False

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
    data_cfg   = cfg["data"]
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
    # Pass num_train_examples so warmup_steps can be derived from the recipe
    # ratio without adding another round-trip after the trainer is built.
    dpo_args = build_dpo_args(cfg, output_dir, beta, num_train_examples=len(train_dataset))
    log.info(
        f"DPOConfig: max_length={dpo_args.max_length}, beta={beta}"
    )
    log.info(f"torch_compile: {dpo_args.torch_compile}")
    log.info(
        f"Batch sizes: train={dpo_args.per_device_train_batch_size}, "
        f"eval={dpo_args.per_device_eval_batch_size}"
    )
    log.info("Best-checkpoint selection enabled (metric_for_best_model=eval_loss)")

    # ── DPOTrainer ────────────────────────────────────────────────────────────
    from trl import DPOTrainer

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
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

    log.info(f"Model saved to {final_dir}")
    log.info("DPO complete. Next: make eval")


if __name__ == "__main__":
    main()