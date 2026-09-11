#!/usr/bin/env python3
"""Fully independent native-HF pretraining control.

This control intentionally shares only experiment values and the exact raw text
splits with SLM pretraining. It does not import or use SLM model, tokenizer,
tokenization, packing, dataset, Trainer, schedule, conversion, export, or
inference code.

The raw train/val/test JSONL files are the comparison data contract. Everything
from text -> token IDs -> packed windows -> model initialization -> training ->
evaluation is rebuilt with external Hugging Face components in this file.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import yaml

log = logging.getLogger("independent-pretrain")

DEFAULT_TOKENIZER = "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T"
SPLITS = ("train", "val", "test")
MANIFEST = "independent_data.json"
AUDIT = "independent_run_audit.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    p.add_argument("--config", type=Path, required=True,
                   help="SLM YAML used only as the shared experiment-value specification")
    p.add_argument("--reference-data-dir", type=Path, required=True,
                   help="Directory containing the exact raw train.jsonl/val.jsonl/test.jsonl comparison documents")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="External tokenizer repository; must match recipe vocab_size")
    p.add_argument("--tokenizer-revision", default="main")
    p.add_argument("--text-field", default="text")
    p.add_argument("--epochs", type=float, default=2.0,
                   help="Passes over independently tokenized/packed train data")
    p.add_argument("--learning-rate", type=float,
                   help="Explicit experiment override; otherwise recipe optimizer.lr")
    p.add_argument("--batch-size", type=int,
                   help="Explicit micro-batch override; otherwise recipe value")
    p.add_argument("--gradient-accumulation", type=int,
                   help="Explicit accumulation override; otherwise recipe value")
    p.add_argument("--warmup-ratio", type=float,
                   help="Explicit override; otherwise recipe warmup_steps/max_steps ratio")
    p.add_argument("--max-steps", type=int,
                   help="Optional bounded diagnostic override. Omit for epoch-based comparison")
    p.add_argument("--seed", type=int, help="Otherwise recipe training.seed")
    p.add_argument("--reuse-data", action="store_true",
                   help="Reuse only a complete hash-verified independently tokenized bundle")
    p.add_argument("--device", choices=("cuda",), default="cuda")
    args = p.parse_args()
    if args.epochs <= 0 or not math.isfinite(args.epochs):
        p.error("--epochs must be positive and finite")
    for name in ("batch_size", "gradient_accumulation", "max_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate is not None and (args.learning_rate <= 0 or not math.isfinite(args.learning_rate)):
        p.error("--learning-rate must be positive and finite")
    if args.warmup_ratio is not None and not 0 <= args.warmup_ratio < 1:
        p.error("--warmup-ratio must be in [0,1)")
    return args


def read_recipe(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or not all(k in cfg for k in ("model", "training", "optimizer")):
        raise ValueError(f"Invalid pretraining recipe: {path}")
    return cfg


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def resolve_hf_revision(repo: str, revision: str) -> str:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision)
    if not info.sha or len(info.sha) != 40:
        raise RuntimeError(f"Could not resolve immutable tokenizer revision for {repo}@{revision}")
    return info.sha


def load_external_tokenizer(args, *, local_dir: Path | None = None):
    from transformers import AutoTokenizer
    if local_dir is None:
        revision = resolve_hf_revision(args.tokenizer, args.tokenizer_revision)
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=revision, trust_remote_code=False, use_fast=True
        )
    else:
        revision = None
        tok = AutoTokenizer.from_pretrained(
            local_dir, local_files_only=True, trust_remote_code=False, use_fast=True
        )
    if not tok.is_fast:
        raise ValueError("Independent control requires a fast external tokenizer")
    if tok.bos_token_id is None or tok.eos_token_id is None:
        raise ValueError("Independent tokenizer must define BOS and EOS")
    if tok.pad_token_id is None:
        # Fixed-length pretraining has no padding. Reuse an existing special token
        # rather than mutating the vocabulary.
        tok.pad_token = tok.eos_token
    return tok, revision


def write_u16(handle, ids: list[int]) -> None:
    if any(i < 0 or i >= 65536 for i in ids):
        raise ValueError("Tokenizer emitted token id outside uint16 range")
    values = array("H", ids)
    if sys.byteorder != "little":
        values.byteswap()
    values.tofile(handle)


def source_identity(data_dir: Path, text_field: str) -> dict:
    result = {"text_field": text_field, "splits": {}}
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing exact raw comparison split: {path}")
        rows = 0
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if not isinstance(row.get(text_field), str):
                    raise ValueError(f"{path}: row {rows + 1} lacks string field {text_field!r}")
                rows += 1
        if rows == 0:
            raise ValueError(f"Empty comparison split: {path}")
        result["splits"][split] = {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": rows}
    return result


def prepare(args: argparse.Namespace, cfg: dict) -> tuple[Path, object, dict]:
    root = args.run_dir.resolve() / "independent-data"
    expected_source = source_identity(args.reference_data_dir.resolve(), args.text_field)
    if root.exists():
        if not args.reuse_data:
            raise RuntimeError(f"Independent data already exists: {root}; use --reuse-data or a fresh --run-dir")
        manifest_path = root / MANIFEST
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing independent data manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        digest = manifest.pop("manifest_sha256", None)
        if digest != stable_json_hash(manifest) or manifest.get("status") != "complete":
            raise RuntimeError("Independent data manifest is corrupt/incomplete")
        if manifest["source"] != expected_source:
            raise RuntimeError("Raw comparison data changed; use a fresh --run-dir")
        if manifest["tokenizer"]["repo"] != args.tokenizer:
            raise RuntimeError("External tokenizer changed; use a fresh --run-dir")
        resolved = resolve_hf_revision(args.tokenizer, args.tokenizer_revision)
        if resolved != manifest["tokenizer"]["revision"]:
            raise RuntimeError("External tokenizer revision changed; pin recorded commit or use fresh --run-dir")
        for split in SPLITS:
            path = root / f"{split}.bin"
            if sha256_file(path) != manifest["packed"][split]["sha256"]:
                raise RuntimeError(f"Independent packed split changed: {path}")
        tok, _ = load_external_tokenizer(args, local_dir=root / "tokenizer")
        manifest["manifest_sha256"] = digest
        return root, tok, manifest

    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise RuntimeError(f"Refusing nonempty unowned independent run directory: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    tok, revision = load_external_tokenizer(args)
    vocab_size = int(cfg["model"]["vocab_size"])
    if len(tok) != vocab_size:
        raise ValueError(
            f"External tokenizer vocab={len(tok)} but shared recipe vocab_size={vocab_size}; "
            "the independent control never resizes embeddings"
        )
    seq_len = int(cfg["model"]["max_position_embeddings"])
    if seq_len < 4:
        raise ValueError("Context length must be at least 4")

    partial = args.run_dir / f".independent-data-{os.getpid()}"
    partial.mkdir(parents=True)
    tok.save_pretrained(partial / "tokenizer")
    packed_meta = {}
    for split in SPLITS:
        source = args.reference_data_dir / f"{split}.jsonl"
        out = partial / f"{split}.bin"
        n_tokens = 0
        docs = 0
        with source.open() as src, out.open("wb") as dst:
            for line in src:
                text = json.loads(line)[args.text_field]
                ids = tok.encode(text, add_special_tokens=False)
                packed = [tok.bos_token_id, *ids, tok.eos_token_id]
                write_u16(dst, packed)
                n_tokens += len(packed)
                docs += 1
        windows = n_tokens // seq_len
        if windows < 1:
            raise RuntimeError(f"{split} produced no complete {seq_len}-token windows")
        packed_meta[split] = {
            "documents": docs,
            "tokens": n_tokens,
            "usable_tokens": windows * seq_len,
            "windows": windows,
            "sha256": sha256_file(out),
        }

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "control_kind": "independent_model_complete_pretrain",
        "source": expected_source,
        "tokenizer": {
            "repo": args.tokenizer,
            "requested_revision": args.tokenizer_revision,
            "revision": revision,
            "vocab_size": len(tok),
            "bos_token_id": tok.bos_token_id,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "files": {p.name: sha256_file(p) for p in sorted((partial / "tokenizer").iterdir()) if p.is_file()},
        },
        "packing": "external tokenizer; per-document BOS + text + EOS; concatenated fixed windows; local implementation",
        "seq_len": seq_len,
        "packed": packed_meta,
        "independence": {
            "uses_slm_tokenizer": False,
            "uses_slm_tokenized_data": False,
            "uses_slm_packing": False,
            "uses_slm_dataset": False,
        },
    }
    manifest["manifest_sha256"] = stable_json_hash(manifest)
    atomic_json(partial / MANIFEST, manifest)
    partial.rename(root)
    return prepare(args, cfg)


def train(args: argparse.Namespace, cfg: dict, data_root: Path, tokenizer, data_manifest: dict) -> None:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Independent control requires exactly one visible CUDA GPU, found {torch.cuda.device_count()}")
    if not os.environ.get("WANDB_API_KEY", "").strip():
        raise RuntimeError("WANDB_API_KEY is required for independent training")
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        raise RuntimeError("W&B must not be disabled for independent training")

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]
    seq_len = int(model_cfg["max_position_embeddings"])

    class PackedDataset(Dataset):
        def __init__(self, path: Path):
            self.data = np.memmap(path, dtype="<u2", mode="r")
            self.windows = len(self.data) // seq_len
        def __len__(self):
            return self.windows
        def __getitem__(self, index):
            start = index * seq_len
            ids = torch.from_numpy(np.array(self.data[start:start + seq_len], dtype=np.int64, copy=True))
            return {"input_ids": ids, "labels": ids.clone()}

    train_ds = PackedDataset(data_root / "train.bin")
    val_ds = PackedDataset(data_root / "val.bin")
    test_ds = PackedDataset(data_root / "test.bin")

    llama = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=int(model_cfg["hidden_size"]),
        intermediate_size=int(model_cfg["intermediate_size"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        num_attention_heads=int(model_cfg["num_attention_heads"]),
        num_key_value_heads=int(model_cfg["num_key_value_heads"]),
        max_position_embeddings=seq_len,
        rope_theta=float(model_cfg["rope_theta"]),
        rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        initializer_range=float(model_cfg.get("initializer_range", 0.02)),
        tie_word_embeddings=bool(model_cfg.get("tie_word_embeddings", True)),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attention_dropout=0.0,
    )

    micro = int(args.batch_size or train_cfg["micro_batch_size"])
    accum = int(args.gradient_accumulation or train_cfg.get("gradient_accumulation_steps", 1))
    lr = float(args.learning_rate or optim_cfg["lr"])
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    planned_steps = int(train_cfg["max_steps"])
    recipe_warmup_ratio = float(train_cfg.get("warmup_steps", 0)) / planned_steps if planned_steps else 0.0
    warmup_ratio = float(args.warmup_ratio if args.warmup_ratio is not None else recipe_warmup_ratio)

    output = args.run_dir / "independent-pretrain"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Independent training output already exists: {output}; choose a fresh --run-dir")
    output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        weight_decay=float(optim_cfg.get("weight_decay", 0.1)),
        adam_beta1=float(optim_cfg.get("beta1", 0.9)),
        adam_beta2=float(optim_cfg.get("beta2", 0.95)),
        adam_epsilon=float(optim_cfg.get("eps", 1e-8)),
        max_grad_norm=float(train_cfg.get("gradient_clip_val", 1.0)),
        optim="adamw_torch_fused",
        lr_scheduler_type=str(train_cfg.get("lr_scheduler", "cosine")),
        bf16=True,
        fp16=False,
        torch_compile=False,
        eval_strategy="steps",
        eval_steps=int(train_cfg.get("eval_steps", 500)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 500)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        logging_strategy="steps",
        logging_steps=int(train_cfg.get("log_steps", 10)),
        report_to=["wandb"],
        run_name=f"independent-native-llama-{args.config.stem}",
        dataloader_num_workers=int(train_cfg.get("num_workers", 4)),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        prediction_loss_only=True,
        gradient_checkpointing=False,
        seed=seed,
    )

    shared = {
        "architecture": {
            k: model_cfg[k] for k in (
                "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
                "rope_theta", "rms_norm_eps", "initializer_range", "tie_word_embeddings",
            )
        },
        "optimizer": {
            "lr": lr,
            "weight_decay": float(optim_cfg.get("weight_decay", 0.1)),
            "beta1": float(optim_cfg.get("beta1", 0.9)),
            "beta2": float(optim_cfg.get("beta2", 0.95)),
            "eps": float(optim_cfg.get("eps", 1e-8)),
        },
        "training": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "micro_batch_size": micro,
            "gradient_accumulation_steps": accum,
            "gradient_clip_val": float(train_cfg.get("gradient_clip_val", 1.0)),
            "lr_scheduler": str(train_cfg.get("lr_scheduler", "cosine")),
            "warmup_ratio": warmup_ratio,
            "precision": "bf16",
            "seed": seed,
        },
    }
    audit = {
        "schema_version": 1,
        "control_kind": "independent_model_complete_pretrain",
        "config_source": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
        "shared_experiment_values": shared,
        "raw_data": data_manifest["source"],
        "external_tokenizer": data_manifest["tokenizer"],
        "direct_hf_model_config": llama.to_dict(),
        "independence": {
            "model": "transformers.LlamaForCausalLM constructed directly",
            "trainer": "transformers.Trainer",
            "tokenization": "external AutoTokenizer + local standalone packing",
            "initialization": "direct HF Llama initialization after transformers.set_seed",
            "uses_slm_model": False,
            "uses_slm_initial_weights": False,
            "uses_slm_conversion": False,
            "uses_slm_tokenizer": False,
            "uses_slm_tokenized_data": False,
            "uses_slm_dataset": False,
            "uses_slm_trainer": False,
            "uses_slm_schedule_code": False,
            "uses_slm_inference": False,
        },
    }
    audit["audit_sha256"] = stable_json_hash(audit)
    atomic_json(output / AUDIT, audit)

    set_seed(seed)
    model = LlamaForCausalLM(llama)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )

    baseline = trainer.evaluate(metric_key_prefix="baseline_validation")
    atomic_json(output / "baseline.json", baseline)
    result = trainer.train()
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    final_dir = output / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir / "tokenizer")
    final_val = trainer.evaluate(metric_key_prefix="final_validation")
    final_test = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="final_test")

    report = {
        "audit_sha256": audit["audit_sha256"],
        "baseline": baseline,
        "train": result.metrics,
        "final_validation": final_val,
        "final_test": final_test,
        "global_step": trainer.state.global_step,
        "raw_train_documents": data_manifest["source"]["splits"]["train"]["rows"],
        "independent_train_tokens": data_manifest["packed"]["train"]["usable_tokens"],
        "independent_train_windows": data_manifest["packed"]["train"]["windows"],
    }
    atomic_json(output / "learning_report.json", report)
    log.info("Independent learning report: %s", output / "learning_report.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.reference_data_dir = args.reference_data_dir.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    cfg = read_recipe(args.config)

    if args.stage == "prepare":
        prepare(args, cfg)
        return
    if args.stage == "train":
        data_root, tokenizer, manifest = prepare(args, cfg)
        train(args, cfg, data_root, tokenizer, manifest)
        return
    data_root, tokenizer, manifest = prepare(args, cfg)
    train(args, cfg, data_root, tokenizer, manifest)


if __name__ == "__main__":
    main()
