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

        for rec in records:
            prompt = rec.get("prompt", "")
            completion = rec.get("completion", "")

            if not prompt or not completion:
                continue

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

            input_ids = prompt_ids + completion_ids + [tokenizer.eos_token_id]
            labels = [-100] * len(prompt_ids) + completion_ids + [tokenizer.eos_token_id]

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            if all(x == -100 for x in labels):
                continue

            self.rows.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

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
    losses: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.detach().cpu()))

    model.train()

    if not losses:
        return float("inf")

    return sum(losses) / len(losses)


def copy_runtime_files(base_model: Path, out_dir: Path) -> None:
    for name in [
        "attention.py",
        "block.py",
        "config.py",
        "mlp.py",
        "model.py",
        "norm.py",
        "README.md",
        "chat_template.jinja",
    ]:
        src = base_model / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    src_remote = base_model / "slm_remote"
    if src_remote.exists():
        dst_remote = out_dir / "slm_remote"
        if dst_remote.exists():
            shutil.rmtree(dst_remote)
        shutil.copytree(src_remote, dst_remote)


def save_checkpoint(model, tokenizer, base_model: Path, out_dir: Path, metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    tokenizer.save_pretrained(str(out_dir))

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

    copy_runtime_files(base_model, out_dir)

    (out_dir / "code_completion_training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train raw code-completion SFT")
    parser.add_argument("--config", required=True)
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
    save_best = bool(train_cfg.get("save_best", True))

    if not base_model.exists():
        raise SystemExit(f"Missing base model: {base_model}")
    if not train_path.exists():
        raise SystemExit(f"Missing train data: {train_path}")
    if not val_path.exists():
        raise SystemExit(f"Missing val data: {val_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=== Raw code-completion SFT ===")
    print(f"Config:      {cfg_path}")
    print(f"Base model:  {base_model}")
    print(f"Train data:  {train_path}")
    print(f"Val data:    {val_path}")
    print(f"Output:      {final_dir}")
    print(f"Device:      {device}")
    print(f"Max updates: {max_updates}")
    print(f"LR:          {learning_rate}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model / "tokenizer"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SLMForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=dtype,
    ).to(device)

    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path)

    train_dataset = RawCompletionDataset(train_records, tokenizer, max_length=max_length)
    val_dataset = RawCompletionDataset(val_records, tokenizer, max_length=max_length)

    if len(train_dataset) == 0:
        raise SystemExit("No train records after tokenization")
    if len(val_dataset) == 0:
        raise SystemExit("No val records after tokenization")

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=Collator(tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_micro_batch_size,
        shuffle=False,
        collate_fn=Collator(tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_steps = max(5, int(max_updates * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_updates,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    update = 0
    micro = 0
    running_loss = 0.0
    best_val = float("inf")

    while update < max_updates:
        for batch in train_loader:
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
                            base_model,
                            output_dir / "best",
                            {
                                "config": str(cfg_path),
                                "base_model": str(base_model),
                                "train_path": str(train_path),
                                "val_path": str(val_path),
                                "update": update,
                                "val_loss": val_loss,
                            },
                        )
                        print(f"Saved best checkpoint: {output_dir / 'best'}")

                if update >= max_updates:
                    break

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
    }

    save_checkpoint(model, tokenizer, base_model, final_dir, final_metadata)
    print(f"Saved final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
