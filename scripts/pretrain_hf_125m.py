#!/usr/bin/env python3
"""
scripts/pretrain_hf_125m.py
---------------------------

125M-only clean Hugging Face pretraining-data substitution benchmark.

Purpose
-------
This script validates the 125M model/training stack with an already-cleaned
Hugging Face pretraining dataset. It intentionally writes the SAME normal
pipeline paths used by the regular curated-data workflow so downstream commands
need no modification:

    data/curated/train.jsonl
    data/curated/val.jsonl
    data/curated/blend_stats.json
    data/tokenized/...
    results/slm-125m/final

After this script prepares/pretrains the 125M base model, the existing flow can
continue unchanged:

    make sft-chat SIZE=125m
    make sft-code SIZE=125m
    make dpo SIZE=125m
    make eval-chat SIZE=125m

Destructive behavior
--------------------
This is a data-substitution tool, not a separate training lane. It can overwrite
normal 125M artifacts. Before writing new data, it checks for existing:

    data/curated
    data/tokenized
    results/slm-125m

If any exist, the script refuses to proceed unless one of these is supplied:

    --backup-existing
        Move or copy existing artifacts into timestamped backup directories
        before writing new data.

    --force --no-backup
        Delete existing artifacts without backup. Use only when you are sure.

Backups are written under:

    data/backups/hf_125m_baseline/<timestamp>/
    results/backups/hf_125m_baseline/<timestamp>/

Default dataset
---------------
    HuggingFaceTB/dclm-edu

This is intentionally 125M-only. It is not a new general-purpose pipeline and
should not be expanded into a second full workflow unless explicitly decided.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

SIZE = "125m"
DEFAULT_DATASET = "HuggingFaceTB/dclm-edu"
DEFAULT_SOURCE_NAME = "dclm_edu"
DEFAULT_TARGET_TOKENS = 6_500_000_000


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    results_dir: Path

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"

    @property
    def tokenized_dir(self) -> Path:
        return self.data_dir / "tokenized"

    @property
    def result_dir(self) -> Path:
        return self.results_dir / f"slm-{SIZE}"

    @property
    def data_backup_root(self) -> Path:
        return self.data_dir / "backups" / "hf_125m_baseline"

    @property
    def results_backup_root(self) -> Path:
        return self.results_dir / "backups" / "hf_125m_baseline"


def parse_int(value: str) -> int:
    try:
        return int(value.replace("_", ""))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}") from exc


def default_chars_per_token() -> float:
    try:
        from config.data_mix import CHARS_PER_TOKEN  # type: ignore
        return float(CHARS_PER_TOKEN)
    except Exception:
        return 4.3


def default_val_fraction() -> float:
    try:
        from config.data_mix import PRETRAIN_VAL_FRACTION  # type: ignore
        return float(PRETRAIN_VAL_FRACTION)
    except Exception:
        return 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="125M-only clean HF pretraining data substitution benchmark."
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "tokenizer", "tokenize", "pretrain", "all"],
        default="prepare",
        help="Stage to run. 'prepare' writes data/curated. 'all' runs prepare + tokenizer + tokenize + pretrain.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--target-tokens", type=parse_int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--chars-per-token", type=float, default=default_chars_per_token())
    parser.add_argument("--val-fraction", type=float, default=default_val_fraction())
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "data")))
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("RESULTS_DIR", "results")))
    parser.add_argument("--max-docs", type=parse_int, default=None, help="Optional doc cap for smoke testing.")
    parser.add_argument("--max-chars", type=parse_int, default=None, help="Optional char cap for smoke testing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backup-existing", action="store_true")
    parser.add_argument(
        "--backup-mode",
        choices=["move", "copy"],
        default="move",
        help="Default move preserves artifacts while avoiding duplicate disk usage. Use copy for literal copy.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--tokenizer-cmd", default="make tokenizer SIZE=125m")
    parser.add_argument("--tokenize-cmd", default="make tokenize SIZE=125m")
    parser.add_argument("--pretrain-cmd", default="make pretrain SIZE=125m")
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[pretrain_hf_125m] {msg}", flush=True)


def existing_artifacts(paths: Paths) -> list[Path]:
    return [p for p in [paths.curated_dir, paths.tokenized_dir, paths.result_dir] if p.exists()]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def backup_or_delete_existing(paths: Paths, args: argparse.Namespace) -> None:
    existing = existing_artifacts(paths)
    if not existing:
        return

    log("Existing artifacts detected:")
    for path in existing:
        log(f"  - {path}")

    if args.backup_existing and args.no_backup:
        raise SystemExit("Use either --backup-existing or --no-backup, not both.")

    if args.no_backup:
        if not args.force:
            raise SystemExit("--no-backup requires --force.")
        for path in existing:
            log(f"Deleting without backup: {path}")
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        return

    if not args.backup_existing:
        raise SystemExit(
            "Refusing to overwrite existing artifacts. Re-run with --backup-existing, "
            "or use --force --no-backup if you intentionally want to delete them."
        )

    ts = timestamp()
    data_backup_dir = paths.data_backup_root / ts
    results_backup_dir = paths.results_backup_root / ts
    data_backup_dir.mkdir(parents=True, exist_ok=True)
    results_backup_dir.mkdir(parents=True, exist_ok=True)

    for path in existing:
        dest = (results_backup_dir / path.name) if path == paths.result_dir else (data_backup_dir / path.name)
        log(f"Backing up ({args.backup_mode}): {path} -> {dest}")
        if args.backup_mode == "copy":
            shutil.copytree(path, dest) if path.is_dir() else shutil.copy2(path, dest)
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        else:
            shutil.move(str(path), str(dest))


def clean_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or None


def write_jsonl_record(fh, record: dict) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_curated_data(args: argparse.Namespace, paths: Paths) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install requirements before running this script.") from exc

    backup_or_delete_existing(paths, args)

    paths.curated_dir.mkdir(parents=True, exist_ok=True)
    train_path = paths.curated_dir / "train.jsonl"
    val_path = paths.curated_dir / "val.jsonl"
    stats_path = paths.curated_dir / "blend_stats.json"

    target_chars = int(args.target_tokens * args.chars_per_token)
    rng = random.Random(args.seed)

    log("Preparing clean HF curated data")
    log(f"dataset:       {args.dataset}")
    log(f"split:         {args.split}")
    log(f"text_field:    {args.text_field}")
    log(f"source_name:   {args.source_name}")
    log(f"target_tokens: {args.target_tokens:,}")
    log(f"target_chars:  {target_chars:,}")
    log(f"val_fraction:  {args.val_fraction}")
    log(f"output:        {paths.curated_dir}")

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    total_docs = total_chars = train_docs = train_chars = val_docs = val_chars = skipped = 0

    with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
        for example in ds:
            if args.max_docs is not None and total_docs >= args.max_docs:
                break
            if args.max_chars is not None and total_chars >= args.max_chars:
                break
            if total_chars >= target_chars:
                break

            text = clean_text(example.get(args.text_field))
            if text is None:
                skipped += 1
                continue

            record = {
                "text": text,
                "source": args.source_name,
                "language": "en",
                "dataset": args.dataset,
                "split": args.split,
            }

            text_chars = len(text)
            total_docs += 1
            total_chars += text_chars

            if rng.random() < args.val_fraction:
                write_jsonl_record(val_f, record)
                val_docs += 1
                val_chars += text_chars
            else:
                write_jsonl_record(train_f, record)
                train_docs += 1
                train_chars += text_chars

            if total_docs % 100_000 == 0:
                est_tokens = total_chars / args.chars_per_token
                log(f"docs={total_docs:,} chars={total_chars:,} est_tokens={est_tokens/1e9:.2f}B")

    estimated_tokens = int(total_chars / args.chars_per_token)
    deficit_chars = max(0, target_chars - total_chars)

    stats = {
        "target": SIZE,
        "generated_by": "scripts/pretrain_hf_125m.py",
        "dataset": args.dataset,
        "split": args.split,
        "source_name": args.source_name,
        "target_tokens": args.target_tokens,
        "target_chars": target_chars,
        "chars_per_token": args.chars_per_token,
        "estimated_tokens": estimated_tokens,
        "train_documents": train_docs,
        "val_documents": val_docs,
        "total_documents": total_docs,
        "total_chars": total_chars,
        "train_chars": train_chars,
        "val_chars": val_chars,
        "skipped_documents": skipped,
        "val_fraction": args.val_fraction,
        "source_mix": {
            args.source_name: {
                "docs": total_docs,
                "chars": total_chars,
                "target_chars": target_chars,
                "deficit": deficit_chars,
                "val_docs": val_docs,
                "val_chars": val_chars,
            }
        },
        "notes": (
            "Clean HF 125M pretraining baseline. This intentionally bypasses "
            "the custom curator mix and writes normal curated train/val files."
        ),
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    log("Prepared curated HF baseline data")
    log(f"train:            {train_path} ({train_docs:,} docs, {train_chars:,} chars)")
    log(f"val:              {val_path} ({val_docs:,} docs, {val_chars:,} chars)")
    log(f"blend_stats:      {stats_path}")
    log(f"estimated_tokens: {estimated_tokens:,}")
    if deficit_chars:
        log(
            f"WARNING: stopped before target_chars by {deficit_chars:,} chars "
            f"({deficit_chars / args.chars_per_token / 1e9:.2f}B tokens)"
        )


def run_command(command: str) -> None:
    log(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {command}")


def main() -> None:
    args = parse_args()
    paths = Paths(data_dir=args.data_dir, results_dir=args.results_dir)

    if args.stage == "prepare":
        prepare_curated_data(args, paths)
    elif args.stage == "tokenizer":
        run_command(args.tokenizer_cmd)
    elif args.stage == "tokenize":
        run_command(args.tokenize_cmd)
    elif args.stage == "pretrain":
        run_command(args.pretrain_cmd)
    elif args.stage == "all":
        prepare_curated_data(args, paths)
        run_command(args.tokenizer_cmd)
        run_command(args.tokenize_cmd)
        run_command(args.pretrain_cmd)
    else:
        raise SystemExit(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
