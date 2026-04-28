"""
pretrain/train.py
-----------------
Pretraining entry point using HuggingFace Trainer.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from transformers import Trainer, TrainerCallback

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


def _configure_cuda_for_performance() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(False)


_configure_cuda_for_performance()


_NUMERIC_CONFIG_KEYS = {
    "lr", "eps", "weight_decay", "beta1", "beta2",
    "gradient_clip_val", "warmup_ratio",
    "rms_norm_eps", "rope_theta", "initializer_range",
    "dpo_beta", "beta",
}


def _coerce_numeric(node):
    if isinstance(node, dict):
        return {
            k: (float(v) if k in _NUMERIC_CONFIG_KEYS and isinstance(v, str) else _coerce_numeric(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_coerce_numeric(x) for x in node]
    return node


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return _coerce_numeric(cfg)


def validate_tokenizer(tokenizer_dir: Path) -> None:
    if not tokenizer_dir.exists() or not any(tokenizer_dir.iterdir()):
        raise RuntimeError(
            f"Tokenizer directory missing or empty: {tokenizer_dir}\n"
            f"Run: make tokenizer-download\n"
            f"Or retrain: make tokenizer && make tokenizer-upload"
        )

    required_files = {
        "tokenizer_config.json": (
            "Contains the baked-in chat_template required by train_sft.py. "
            "Retrain the tokenizer: make tokenizer"
        ),
        "slm_tokenizer.json": (
            "Raw BPE tokenizer required by tokenize_data.py. "
            "Retrain the tokenizer: make tokenizer"
        ),
    }

    for filename, hint in required_files.items():
        path = tokenizer_dir / filename
        if not path.exists():
            raise RuntimeError(f"Missing tokenizer file: {path}\n{hint}")
        try:
            with open(path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Tokenizer file is not valid JSON: {path}\n"
                f"  Error: {e}\n"
                f"  Hint: {hint}"
            ) from e

    log.info(f"Tokenizer validated at {tokenizer_dir}")


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [
        p for p in output_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    if not candidates:
        return None
    numbered = []
    for p in candidates:
        try:
            step = int(p.name.split("-", 1)[1])
            numbered.append((step, p))
        except (IndexError, ValueError):
            continue
    if not numbered:
        return None
    numbered.sort()
    return numbered[-1][1]


class VRAMProbe(TrainerCallback):
    """Log peak VRAM at step 200 so the analytical profile can be calibrated."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 200 and torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            log.info(f"[VRAMProbe step 200] allocated peak: {alloc:.2f} GB, "
                     f"reserved peak: {reserved:.2f} GB")


class SLMTrainer(Trainer):
    """
    Trainer subclass with an explicit compute_loss override.

    SLMForCausalLM returns a plain dict when labels are present during
    training/eval. This avoids a torch.compile / Accelerate Dynamo issue
    where CausalLMOutputWithPast can be reconstructed incorrectly during
    scheduled eval, causing `.loss` to become a dict instead of a tensor.

    For generation or inference paths where labels are not present,
    SLMForCausalLM can still return CausalLMOutputWithPast for Hugging Face
    compatibility.
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        else:
            loss = outputs.loss

        if loss is None:
            raise ValueError(
                "Model returned no loss. Check that 'labels' is present in inputs. "
                f"Available keys: {list(inputs.keys())}. "
                f"Output type: {type(outputs)}"
            )

        if not torch.is_tensor(loss):
            raise TypeError(
                f"Expected loss to be a torch.Tensor, got {type(loss)}: {loss}"
            )

        return (loss, outputs) if return_outputs else loss


def build_training_args(cfg: dict, output_dir: Path, resume: bool):
    from transformers import TrainingArguments

    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]

    has_cuda = torch.cuda.is_available()
    precision = train_cfg.get("precision", "bf16")
    use_bf16  = has_cuda and precision == "bf16"
    use_fp16  = has_cuda and precision == "fp16"

    # torch.compile is ON by default — interacted badly with HF Trainer eval
    # (returned wrapped output that broke loss extraction). The SLMTrainer
    # subclass below also handles this, but keeping compile off until the
    # interaction is verified end-to-end on a full run.
    torch_compile = bool(train_cfg.get("torch_compile", True)) and has_cuda

    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=train_cfg["max_steps"],
        warmup_steps=train_cfg.get("warmup_steps", 2000),
        per_device_train_batch_size=train_cfg["micro_batch_size"],
        per_device_eval_batch_size=train_cfg["micro_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),

        learning_rate=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.1)),
        adam_beta1=float(optim_cfg.get("beta1", 0.9)),
        adam_beta2=float(optim_cfg.get("beta2", 0.95)),
        adam_epsilon=float(optim_cfg.get("eps", 1e-8)),
        max_grad_norm=float(train_cfg.get("gradient_clip_val", 1.0)),
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",

        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        bf16=use_bf16,
        fp16=use_fp16,

        torch_compile=torch_compile,
        torch_compile_backend=train_cfg.get("torch_compile_backend", "inductor"),
        torch_compile_mode=train_cfg.get("torch_compile_mode", "default"),

        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 1000),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 1000),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=False,

        logging_strategy="steps",
        logging_steps=train_cfg.get("log_steps", 10),
        report_to=train_cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("name", "slm-pretrain"),

        dataloader_num_workers=train_cfg.get("num_workers", 4),
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),

        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False,
    )


def main():
    parser = argparse.ArgumentParser(description="SLM Pretraining")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    model_name = cfg["name"]
    output_dir = args.results_dir / model_name

    log.info(f"=== SLM Pretraining ===")
    log.info(f"Config:     {args.config}")
    log.info(f"Model:      {model_name}")
    log.info(f"Output:     {output_dir}")
    log.info(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        log.info(f"GPU:        {torch.cuda.get_device_name(0)} "
                 f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = _find_latest_checkpoint(output_dir)
        if resume_checkpoint is None:
            log.warning(
                f"--resume passed but no checkpoint found in {output_dir}. "
                f"Training will start from scratch."
            )
        else:
            log.info(f"Resuming from checkpoint: {resume_checkpoint}")

    tokenizer_dir = args.data_dir / "tokenizer"
    validate_tokenizer(tokenizer_dir)

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
    model    = SLMForCausalLM(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    from pretrain.data.dataset import load_train_val

    tokenized_dir = args.data_dir / "tokenized"
    seq_len = model_cfg_dict["max_position_embeddings"]

    log.info(f"Loading datasets from {tokenized_dir}")
    train_ds, val_ds = load_train_val(tokenized_dir=tokenized_dir, seq_len=seq_len)

    log.info(f"Train examples: {len(train_ds):,}")
    log.info(f"Val examples:   {len(val_ds):,}")

    budget = train_ds.token_budget()
    log.info(f"Training tokens: {budget['total_training_tokens'] / 1e9:.2f}B")

    training_args = build_training_args(cfg, output_dir, resume=args.resume)

    log.info(
        f"Throughput knobs: "
        f"micro_batch={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, "
        f"bf16={training_args.bf16}, "
        f"optim={training_args.optim}, "
        f"compile={training_args.torch_compile}, "
        f"grad_ckpt={training_args.gradient_checkpointing}"
    )

    if "wandb" in training_args.report_to:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "slm"),
            name=model_name,
            config={
                "model":           model_cfg_dict,
                "training":        cfg["training"],
                "optimizer":       cfg["optimizer"],
                "n_params":        n_params,
                "n_train_tokens":  budget["total_training_tokens"],
                "n_val_examples":  len(val_ds),
            },
        )

    trainer = SLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[VRAMProbe()],
    )

    if not args.resume:
        log.info("Running baseline eval before training (step 0)...")
        baseline = trainer.evaluate()
        log.info(f"Baseline eval: {baseline}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint if args.resume else None)

    log.info("Saving final model...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    model_config.save_pretrained(str(final_dir))

    shutil.copytree(tokenizer_dir, final_dir / "tokenizer", dirs_exist_ok=True)
    log.info(f"Tokenizer copied to {final_dir / 'tokenizer'}")

    log.info(f"Model saved to {final_dir}")
    log.info("Pretraining complete.")
    log.info("Next step: make sft")


if __name__ == "__main__":
    main()