#!/usr/bin/env python3
"""
Train raw code-completion SFT.

Input records are JSONL with:
    {"prompt": str, "completion": str}

Training concatenates prompt + completion + EOS and masks labels over prompt
tokens, so loss applies only to body-completion tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

from config.runtime import configure_torch_runtime
from config.chat import tokenizer_fingerprint
from config.checkpoints import (
    bundled_tokenizer_dir, checkpoint_identity, find_latest_checkpoint,
    resolve_training_checkpoint, validate_or_write_run_audit, architecture_identity,
)
from config.holdout import sha256_file
from curator.state import stable_digest

RAW_AUDIT = "code_completion_run_audit.json"

from model.config import SLMConfig
from model.model import SLMForCausalLM

AutoConfig.register("slm", SLMConfig)


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class RawCompletionDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows: list[dict[str, Any]] = []
        self.stats = {
            "input_records": len(records),
            "retained_records": 0,
            "invalid_records": 0,
            "truncated_records": 0,
            "supervised_tokens": 0,
        }

        for rec in records:
            prompt = rec.get("prompt", "")
            completion = rec.get("completion", "")

            if not prompt or not completion:
                self.stats["invalid_records"] += 1
                continue

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

            original_length = len(prompt_ids) + len(completion_ids) + 1
            if original_length > max_length:
                self.stats["truncated_records"] += 1
                # Keep the end of the prompt (normally the function signature
                # and docstring) and prioritize supervised completion tokens.
                if len(completion_ids) + 1 < max_length:
                    prompt_budget = max_length - len(completion_ids) - 1
                    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
                else:
                    prompt_budget = min(len(prompt_ids), max(1, max_length // 4))
                    prompt_ids = prompt_ids[-prompt_budget:]
                    completion_ids = completion_ids[: max_length - prompt_budget - 1]

            input_ids = prompt_ids + completion_ids + [tokenizer.eos_token_id]
            labels = [-100] * len(prompt_ids) + completion_ids + [tokenizer.eos_token_id]

            if all(x == -100 for x in labels):
                self.stats["invalid_records"] += 1
                continue

            self.rows.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )
            self.stats["retained_records"] += 1
            self.stats["supervised_tokens"] += sum(x != -100 for x in labels)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.rows[idx]


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in batch)

        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for row in batch:
            n = len(row["input_ids"])
            pad = max_len - n

            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad)
            labels.append(row["labels"] + [-100] * pad)
            attention_mask.append([1] * n + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def evaluate_loss(model, loader, device: torch.device) -> float:
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            token_count = int((batch["labels"][:, 1:] != -100).sum().item())
            weighted_loss += float(out.loss.detach().cpu()) * token_count
            supervised_tokens += token_count

    model.train()

    if not supervised_tokens:
        return float("inf")

    return weighted_loss / supervised_tokens


def save_checkpoint(model, tokenizer, out_dir: Path, metadata: dict[str, Any], *,
                    tokenizer_source: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(out_dir))
    # Raw completion does not alter the tokenizer/template. Preserve its exact
    # source files instead of reserializing incidental tokenizer-config fields.
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "chat_template.jinja", "slm_tokenizer.json")
    for destination in (out_dir / "tokenizer", out_dir):
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = tokenizer_source / name
            if source.is_file():
                shutil.copy2(source, destination / name)
        if (tokenizer_source / "chat_templates").is_dir():
            shutil.copytree(tokenizer_source / "chat_templates", destination / "chat_templates", dirs_exist_ok=True)

    (out_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "code_completion_training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_model_tokenizer(model, tokenizer, max_length: int) -> None:
    embedding_rows = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != embedding_rows:
        raise ValueError(
            f"Tokenizer has {len(tokenizer):,} tokens but model embeddings have "
            f"{embedding_rows:,} rows"
        )
    context = int(getattr(model.config, "max_position_embeddings", max_length))
    if max_length > context:
        raise ValueError(f"max_length={max_length} exceeds model context={context}")
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, name)
        config_id = getattr(model.config, name, None)
        if token_id is None:
            raise ValueError(f"Tokenizer has no {name}")
        if config_id is not None and token_id != config_id:
            raise ValueError(f"{name} mismatch: tokenizer={token_id}, model={config_id}")


def latest_recovery_checkpoint(output_dir: Path) -> Path | None:
    return find_latest_checkpoint(output_dir)


def training_batches(num_examples: int, batch_size: int, *, seed: int,
                     epoch: int, next_batch: int = 0) -> list[list[int]]:
    """Replay an epoch's permutation from its own generator, then apply the cursor.

    The sampler never consumes the model/dropout RNG. Checkpoints are saved only
    after complete optimizer updates, so no partially accumulated gradients need
    to be reconstructed.
    """
    if min(num_examples, batch_size) <= 0 or epoch < 0:
        raise ValueError("Invalid sampler dimensions or epoch")
    count = math.ceil(num_examples / batch_size)
    if not 0 <= next_batch <= count:
        raise RuntimeError("Recovery data cursor is outside the current dataset")
    order = torch.randperm(num_examples, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    return [order[start:start + batch_size]
            for start in range(next_batch * batch_size, num_examples, batch_size)]


def validate_prepared_data(train_path: Path, val_path: Path, base_model: Path) -> dict:
    """Require records derived only from the parent stage's original split membership."""
    stats_path = train_path.parent / "stats.json"
    if val_path.parent != train_path.parent or not stats_path.is_file():
        raise RuntimeError("Restore/reprepare raw completion train, val and stats.json together")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("validation_origin") != "parent_val_only":
        raise RuntimeError("Reprepare raw completion data: validation must come only from the parent holdout")
    for path in (train_path, val_path):
        if stats.get("files", {}).get(path.name) != sha256_file(path):
            raise RuntimeError(f"Raw completion preparation identity mismatch: {path}")
    parent_path = base_model / "sft_data_manifest.json"
    if not parent_path.is_file():
        raise RuntimeError("Raw completion requires the parent code-SFT data manifest")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    audit = json.loads((base_model / "sft_run_audit.json").read_text(encoding="utf-8"))
    contract = audit.get("contract", {})
    if (audit.get("contract_sha256") != stable_digest(contract)
            or contract.get("stage") != "sft_code"
            or contract.get("data_manifest_sha256") != stable_digest(parent)):
        raise RuntimeError("Parent code-SFT manifest/audit identity mismatch")
    for filename in ("train.jsonl", "val.jsonl"):
        if stats.get("parent_files", {}).get(filename) != parent.get("files", {}).get(filename, {}).get("sha256"):
            raise RuntimeError("Raw completion inputs do not match the parent SFT training/holdout files")
    return stats


def save_recovery_state(
    model,
    tokenizer,
    output_dir: Path,
    optimizer,
    scheduler,
    update: int,
    best_val: float,
    metadata: dict[str, Any],
    *, epoch: int, next_batch: int, sampler_shape: dict,
    run_contract_sha256: str, tokenizer_source: Path,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{update}"
    if checkpoint_dir.exists():
        raise RuntimeError(f"Recovery checkpoint already exists: {checkpoint_dir}. "
                           "Move later checkpoints aside before explicitly rewinding this run.")
    staging = output_dir / f".checkpoint-{update}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    save_checkpoint(model, tokenizer, staging, metadata, tokenizer_source=tokenizer_source)
    shutil.copy2(output_dir / RAW_AUDIT, staging / RAW_AUDIT)
    best_dir = output_dir / "best"
    # 'best/' can advance after this recovery step and before a crash. Snapshot
    # it here so a valid recovery does not depend on that mutable future state.
    saved_best = staging / "best"
    if best_dir.is_dir() and best_val < float("inf"):
        shutil.copytree(best_dir, saved_best)
    state = {
        "schema_version": 2,
        "contract_sha256": run_contract_sha256,
        "data_cursor": {"epoch": epoch, "next_batch": next_batch, **sampler_shape},
        "best_checkpoint_identity": checkpoint_identity(saved_best) if saved_best.is_dir() else None,
        "update": update,
        "best_val": best_val,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, staging / "training_state.pt")
    staging.replace(checkpoint_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train raw code-completion SFT")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from the latest checkpoint, or from the supplied checkpoint path",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    base_model = expand_path(model_cfg["base_model_path"])
    output_dir = expand_path(model_cfg["output_dir"])
    final_dir = output_dir / "final"

    train_path = expand_path(data_cfg["train_path"])
    val_path = expand_path(data_cfg["val_path"])

    max_length = int(data_cfg.get("max_length", 768))
    micro_batch_size = int(train_cfg.get("micro_batch_size", 8))
    eval_micro_batch_size = int(train_cfg.get("eval_micro_batch_size", micro_batch_size))
    gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 4))
    max_updates = int(train_cfg.get("max_updates", 500))
    learning_rate = float(train_cfg.get("learning_rate", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
    eval_steps = int(train_cfg.get("eval_steps", 100))
    save_steps = int(train_cfg.get("save_steps", eval_steps))
    save_best = bool(train_cfg.get("save_best", True))
    if min(micro_batch_size, eval_micro_batch_size, gradient_accumulation_steps) <= 0:
        raise SystemExit("Batch sizes and gradient accumulation must be positive")
    if max_updates <= 0 or eval_steps <= 0 or save_steps <= 0:
        raise SystemExit("max_updates, eval_steps, and save_steps must be positive")
    if eval_steps > max_updates or save_steps > max_updates:
        raise SystemExit(
            "eval_steps/save_steps exceed max_updates; the run would have no "
            "comparable evaluation or recoverable checkpoint"
        )

    if not base_model.exists():
        raise SystemExit(f"Missing base model: {base_model}")
    if not train_path.exists():
        raise SystemExit(f"Missing train data: {train_path}")
    if not val_path.exists():
        raise SystemExit(f"Missing val data: {val_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    configure_torch_runtime()

    print("=== Raw code-completion SFT ===")
    print(f"Config:      {cfg_path}")
    print(f"Base model:  {base_model}")
    print(f"Train data:  {train_path}")
    print(f"Val data:    {val_path}")
    print(f"Output:      {final_dir}")
    print(f"Device:      {device}")
    print(f"Max updates: {max_updates}")
    print(f"LR:          {learning_rate}")

    resume_dir: Path | None = None
    if args.resume:
        resume_dir = (
            latest_recovery_checkpoint(output_dir)
            if args.resume == "latest"
            else expand_path(args.resume)
        )
        if resume_dir is None or not (resume_dir / "training_state.pt").is_file():
            raise SystemExit(f"Incomplete raw recovery checkpoint in {output_dir}; "
                             "restore all state or explicitly select an earlier checkpoint")
        if resume_dir.resolve().parent != output_dir.resolve():
            raise SystemExit("Restore the recovery checkpoint and run audit together under the configured output_dir")
    else:
        resolve_training_checkpoint(output_dir, resume=False, audit_filename=RAW_AUDIT)

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("The optional raw-completion loop is single-process; use one GPU for this branch")
    preparation = validate_prepared_data(train_path, val_path, base_model)
    base_identity = checkpoint_identity(base_model)
    tokenizer_path = bundled_tokenizer_dir(resume_dir or base_model)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tokenizer_fingerprint(tokenizer_path) != base_identity["tokenizer_sha256"]:
        raise RuntimeError("Recovery tokenizer differs from the original parent tokenizer")

    model = SLMForCausalLM.from_pretrained(
        str(resume_dir or base_model),
        dtype=dtype,
        weights_only=True,
    ).to(device)
    validate_model_tokenizer(model, tokenizer, max_length)

    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path)

    train_dataset = RawCompletionDataset(train_records, tokenizer, max_length=max_length)
    val_dataset = RawCompletionDataset(val_records, tokenizer, max_length=max_length)

    if len(train_dataset) == 0:
        raise SystemExit("No train records after tokenization")
    if len(val_dataset) == 0:
        raise SystemExit("No val records after tokenization")
    print(f"Train tokenization audit: {json.dumps(train_dataset.stats, sort_keys=True)}")
    print(f"Val tokenization audit:   {json.dumps(val_dataset.stats, sort_keys=True)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_micro_batch_size,
        shuffle=False,
        collate_fn=Collator(tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    if not 0.0 <= warmup_ratio <= 1.0:
        raise SystemExit(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")
    warmup_steps = round(max_updates * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_updates,
    )

    contract = {
        "contract_version": 1, "stage": "sft_code_completion",
        "resolved_config": cfg, "base_checkpoint": base_identity,
        "architecture": architecture_identity(model.config),
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer_path),
        "preparation_sha256": stable_digest(preparation),
        "sampler": "seed_plus_epoch_v1", "precision": str(dtype),
    }
    validate_or_write_run_audit(
        output_dir, contract, audit_filename=RAW_AUDIT, schema_version=1,
        resume=bool(args.resume), write=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    update = 0
    micro = 0
    running_loss = 0.0
    best_val = float("inf")
    epoch = 0
    next_batch = 0
    sampler_shape = {"examples": len(train_dataset), "batch_size": micro_batch_size,
                     "gradient_accumulation_steps": gradient_accumulation_steps}
    if resume_dir is not None:
        state = torch.load(
            resume_dir / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        required = {"optimizer", "scheduler", "update", "best_val", "python_random_state",
                    "torch_random_state", "data_cursor", "contract_sha256", "best_checkpoint_identity"}
        if state.get("schema_version") != 2 or not required.issubset(state):
            raise RuntimeError("Incomplete/legacy raw recovery state: sampler cursor and complete optimizer state are required")
        if state["contract_sha256"] != stable_digest(contract):
            raise RuntimeError("Raw recovery state belongs to a different run contract")
        cursor = state["data_cursor"]
        if any(cursor.get(k) != v for k, v in sampler_shape.items()):
            raise RuntimeError("Raw recovery sampler shape changed")
        epoch, next_batch = cursor["epoch"], cursor["next_batch"]
        training_batches(len(train_dataset), micro_batch_size, seed=seed, epoch=epoch, next_batch=next_batch)
        step_suffix = resume_dir.name.removeprefix("checkpoint-")
        if not step_suffix.isdigit() or state["update"] != int(step_suffix) or not 0 < state["update"] <= max_updates:
            raise RuntimeError("Raw recovery update does not match its checkpoint")
        expected_batches = state["update"] * gradient_accumulation_steps
        actual_batches = epoch * math.ceil(len(train_dataset) / micro_batch_size) + next_batch
        if actual_batches != expected_batches:
            raise RuntimeError("Raw recovery cursor disagrees with completed optimizer updates")
        if device.type == "cuda" and "cuda_random_state" not in state:
            raise RuntimeError("Missing CUDA RNG recovery state")
        if (not state["optimizer"].get("state")
                or state["scheduler"].get("last_epoch") != state["update"]):
            raise RuntimeError("Raw recovery is missing optimizer progress or its matching scheduler step")
        if save_best and state["best_val"] < float("inf"):
            saved_best = resume_dir / "best"
            if not saved_best.is_dir() or checkpoint_identity(saved_best) != state["best_checkpoint_identity"]:
                raise RuntimeError("Restore the best-model snapshot contained in this raw recovery checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        update = int(state["update"])
        best_val = float(state["best_val"])
        random.setstate(state["python_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        if device.type == "cuda" and "cuda_random_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_random_state"])
        if save_best:
            # Restore the best model belonging to this step, not an uncommitted
            # best from later work. This is run-state recovery, not retention.
            best_dir = output_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            if best_val < float("inf"):
                shutil.copytree(resume_dir / "best", best_dir)
        print(f"Resumed from {resume_dir} at update {update}, epoch {epoch}, next batch {next_batch}")

    validate_or_write_run_audit(
        output_dir, contract, audit_filename=RAW_AUDIT, schema_version=1,
        resume=bool(args.resume), write=True,
    )
    while update < max_updates:
        batches = training_batches(len(train_dataset), micro_batch_size, seed=seed,
                                   epoch=epoch, next_batch=next_batch)
        train_loader = DataLoader(
            train_dataset, batch_sampler=batches, collate_fn=Collator(tokenizer.pad_token_id),
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        for batch in train_loader:
            next_batch += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss / gradient_accumulation_steps
            loss.backward()

            running_loss += float(out.loss.detach().cpu())
            micro += 1

            if micro % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                update += 1

                if update == 1 or update % 25 == 0:
                    avg = running_loss / max(1, micro)
                    print(
                        f"update={update}/{max_updates} "
                        f"train_loss={avg:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0
                    micro = 0

                if eval_steps > 0 and update % eval_steps == 0:
                    val_loss = evaluate_loss(model, val_loader, device)
                    print(f"update={update}/{max_updates} val_loss={val_loss:.4f}")

                    if save_best and val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(
                            model,
                            tokenizer,
                            output_dir / "best",
                            {
                                "config": str(cfg_path),
                                "base_model": str(base_model),
                                "train_path": str(train_path),
                                "val_path": str(val_path),
                                "update": update,
                                "val_loss": val_loss,
                            },
                            tokenizer_source=tokenizer_path,
                        )
                        print(f"Saved best checkpoint: {output_dir / 'best'}")

                if update % save_steps == 0:
                    save_recovery_state(
                        model,
                        tokenizer,
                        output_dir,
                        optimizer,
                        scheduler,
                        update,
                        best_val,
                        {
                            "config": str(cfg_path),
                            "base_model": str(base_model),
                            "train_path": str(train_path),
                            "val_path": str(val_path),
                            "update": update,
                            "best_val_loss": (
                                best_val if best_val < float("inf") else None
                            ),
                            "train_tokenization": train_dataset.stats,
                            "val_tokenization": val_dataset.stats,
                        },
                        epoch=epoch, next_batch=next_batch, sampler_shape=sampler_shape,
                        run_contract_sha256=stable_digest(contract), tokenizer_source=tokenizer_path,
                    )
                    print(f"Saved recovery checkpoint: {output_dir / f'checkpoint-{update}'}")

                if update >= max_updates:
                    break
        if next_batch == math.ceil(len(train_dataset) / micro_batch_size):
            epoch += 1
            next_batch = 0

    # Evaluate the actual final update if it did not coincide with eval_steps.
    if update % eval_steps:
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"update={update}/{max_updates} val_loss={val_loss:.4f}")
        if save_best and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model,
                tokenizer,
                output_dir / "best",
                {"config": str(cfg_path), "update": update, "val_loss": val_loss},
                tokenizer_source=tokenizer_path,
            )

    final_metadata = {
        "config": str(cfg_path),
        "base_model": str(base_model),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "updates": update,
        "max_length": max_length,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "train_tokenization": train_dataset.stats,
        "val_tokenization": val_dataset.stats,
    }

    if save_best:
        best_dir = output_dir / "best"
        if not best_dir.exists():
            raise RuntimeError("save_best=true but no evaluated best checkpoint exists")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(best_dir, final_dir)
        (final_dir / "code_completion_training_metadata.json").write_text(
            json.dumps(final_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        save_checkpoint(model, tokenizer, final_dir, final_metadata, tokenizer_source=tokenizer_path)
    shutil.copy2(output_dir / RAW_AUDIT, final_dir / RAW_AUDIT)
    print(f"Saved final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
