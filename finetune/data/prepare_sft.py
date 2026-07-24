#!/usr/bin/env python3
"""Prepare pinned external datasets for instruct and code SFT.

This repository consumes SFT data; it does not generate synthetic SFT data.
Sources, immutable revisions, formats, and row caps live in
``finetune/configs/sft_data_sources.yaml`` so a regenerated synthetic dataset
can later replace UltraChat or Magicoder without changing the trainer.

The output remains conversational JSONL:

    {"conversations": [{"role": "system", ...}, ...], "source": "..."}

An accompanying ``manifest.json`` records provenance and integrity checks.
Train/validation splitting is grouped by normalized user prompt so duplicate
prompts can never leak across the evaluation boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import sft_code_data_dir, sft_instruct_data_dir

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DEFAULT_SYSTEM = "You are a helpful, harmless, and honest assistant."
CODE_SYSTEM = (
    "You are an expert programming assistant. When code is requested, write "
    "code directly and avoid unnecessary explanation. When explanation is "
    "requested, explain clearly in prose."
)
VALID_SIZES = ("mini", "125m", "350m", "1b")
VALID_STAGES = ("instruct", "code")
SPACE_RE = re.compile(r"\s+")


def canonical_text(value: str) -> str:
    return SPACE_RE.sub(" ", value.strip().lower())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_user_prompt(messages: list[dict[str, str]]) -> str:
    return next(
        (canonical_text(message["content"]) for message in messages if message["role"] == "user"),
        "",
    )


def normalize_messages(raw: Any, system_prompt: str) -> list[dict[str, str]] | None:
    if not isinstance(raw, list):
        return None

    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        role = item.get("role") or item.get("from")
        content = item.get("content") or item.get("value")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None
        content = content.strip()
        if not content:
            return None
        messages.append({"role": role, "content": content})

    if not messages:
        return None
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})

    conversational = [m for m in messages if m["role"] != "system"]
    if len(conversational) < 2 or conversational[0]["role"] != "user":
        return None
    if conversational[-1]["role"] != "assistant":
        return None
    if any(
        left["role"] == right["role"]
        for left, right in zip(conversational, conversational[1:])
    ):
        return None
    return messages


def normalize_record(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any] | None:
    source_format = source["format"]
    if source_format == "conversational":
        raw_messages = row.get("messages") or row.get("conversations")
        messages = normalize_messages(raw_messages, DEFAULT_SYSTEM)
        sft_type = "general_assistant"
    elif source_format == "magicoder":
        prompt = row.get("problem") or row.get("instruction")
        response = row.get("solution") or row.get("response") or row.get("output")
        if not isinstance(prompt, str) or not isinstance(response, str):
            return None
        prompt, response = prompt.strip(), response.strip()
        if not prompt or not response:
            return None
        messages = normalize_messages(
            [
                {"role": "system", "content": CODE_SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            CODE_SYSTEM,
        )
        sft_type = "code_generation"
    else:
        raise ValueError(f"Unsupported SFT source format: {source_format!r}")

    if messages is None:
        return None
    return {
        "conversations": messages,
        "source": source["source"],
        "sft_type": sft_type,
    }


def load_source(source: dict[str, Any]):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": source["split"],
        "revision": source["revision"],
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return load_dataset(source["dataset"], **kwargs)


def select_source_rows(dataset, cap: int, seed: int):
    if cap <= 0:
        raise ValueError(f"max_records must be positive, got {cap}")
    if len(dataset) <= cap:
        return dataset
    return dataset.shuffle(seed=seed).select(range(cap))


def prepare_records(dataset, source: dict[str, Any], quality: dict[str, float]) -> tuple[list[dict], dict]:
    valid: list[dict[str, Any]] = []
    invalid = 0
    for row in dataset:
        record = normalize_record(row, source)
        if record is None:
            invalid += 1
        else:
            valid.append(record)

    total = len(dataset)
    invalid_fraction = invalid / total if total else 1.0
    if invalid_fraction > float(quality["max_invalid_fraction"]):
        raise RuntimeError(
            f"{source['source']} rejected {invalid}/{total} rows "
            f"({invalid_fraction:.2%}), above max_invalid_fraction="
            f"{float(quality['max_invalid_fraction']):.2%}"
        )

    unique: dict[str, dict[str, Any]] = {}
    prompt_counts: Counter[str] = Counter()
    for record in valid:
        key = sha256_json(record["conversations"])
        unique.setdefault(key, record)
        prompt_counts[first_user_prompt(record["conversations"])] += 1

    duplicate_count = len(valid) - len(unique)
    duplicate_fraction = duplicate_count / len(valid) if valid else 1.0
    if duplicate_fraction > float(quality["max_duplicate_fraction"]):
        raise RuntimeError(
            f"{source['source']} contains {duplicate_count}/{len(valid)} exact "
            f"duplicates ({duplicate_fraction:.2%}), above max_duplicate_fraction="
            f"{float(quality['max_duplicate_fraction']):.2%}. Fix the source dataset."
        )

    records = list(unique.values())
    unique_prompt_ratio = len(prompt_counts) / len(valid) if valid else 0.0
    if unique_prompt_ratio < float(quality["min_unique_prompt_ratio"]):
        raise RuntimeError(
            f"{source['source']} unique-prompt ratio is {unique_prompt_ratio:.2%}, "
            f"below min_unique_prompt_ratio="
            f"{float(quality['min_unique_prompt_ratio']):.2%}. Fix the source dataset."
        )

    stats = {
        "selected_rows": total,
        "valid_rows": len(valid),
        "invalid_rows": invalid,
        "invalid_fraction": invalid_fraction,
        "exact_duplicates_removed": duplicate_count,
        "duplicate_fraction": duplicate_fraction,
        "unique_conversations": len(records),
        "unique_prompts": len(prompt_counts),
        "unique_prompt_ratio": unique_prompt_ratio,
        "maximum_prompt_multiplicity": max(prompt_counts.values(), default=0),
    }
    return records, stats


def grouped_split(
    records: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if len(records) < 2:
        raise RuntimeError("At least two valid, unique SFT records are required")

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        prompt = first_user_prompt(record["conversations"])
        groups.setdefault(prompt, []).append(record)
    if len(groups) < 2:
        raise RuntimeError("At least two unique user prompts are required")

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(records) * validation_fraction))
    val_keys: set[str] = set()
    val_count = 0
    for key in keys:
        if val_count >= target and val_keys:
            break
        val_keys.add(key)
        val_count += len(groups[key])

    train = [record for key in keys if key not in val_keys for record in groups[key]]
    val = [record for key in keys if key in val_keys for record in groups[key]]
    random.Random(seed + 1).shuffle(train)
    random.Random(seed + 2).shuffle(val)
    return train, val


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_dir(stage: str, size: str) -> Path:
    return sft_instruct_data_dir(size) if stage == "instruct" else sft_code_data_dir(size)


def source_contract(config: dict[str, Any], stage: str, size: str) -> dict[str, Any]:
    source = dict(config["stages"][stage])
    caps = source.pop("max_records_by_size")
    if size not in caps:
        raise KeyError(f"No {stage} row cap configured for size {size!r}")
    return {
        "config_version": config["version"],
        "stage": stage,
        "size": size,
        "seed": int(config["seed"]),
        "validation_fraction": float(config["validation_fraction"]),
        "quality": config["quality"],
        "source": source,
        "max_records": int(caps[size]),
    }


def prepare_stage(config: dict[str, Any], stage: str, size: str, force: bool) -> None:
    contract = source_contract(config, stage, size)
    contract_hash = sha256_json(contract)
    destination = output_dir(stage, size)
    manifest_path = destination / "manifest.json"

    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") == contract_hash:
            for filename in ("train.jsonl", "val.jsonl"):
                path = destination / filename
                expected_hash = existing.get("files", {}).get(filename, {}).get("sha256")
                if not path.exists() or not expected_hash or sha256_file(path) != expected_hash:
                    raise RuntimeError(
                        f"{path} is missing or differs from its manifest. "
                        "Restore it or rerun intentionally with --force."
                    )
            log.info("%s/%s already matches source contract; reusing it", size, stage)
            return
        raise RuntimeError(
            f"{destination} was prepared from a different source contract. "
            "Inspect or archive it, then rerun with --force."
        )
    if not manifest_path.exists() and destination.exists() and any(destination.iterdir()) and not force:
        raise RuntimeError(
            f"{destination} contains unmanifested data. Inspect or archive it, "
            "then rerun with --force."
        )

    source = contract["source"]
    log.info(
        "Loading %s@%s (%s)",
        source["dataset"],
        source["revision"],
        source["split"],
    )
    dataset = select_source_rows(
        load_source(source),
        cap=contract["max_records"],
        seed=contract["seed"],
    )
    records, quality_stats = prepare_records(dataset, source, contract["quality"])
    train, val = grouped_split(
        records,
        validation_fraction=contract["validation_fraction"],
        seed=contract["seed"],
    )

    train_prompts = {first_user_prompt(row["conversations"]) for row in train}
    val_prompts = {first_user_prompt(row["conversations"]) for row in val}
    overlap = train_prompts & val_prompts
    if overlap:
        raise AssertionError(f"Internal error: {len(overlap)} prompts cross train/val")

    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "train.jsonl", train)
    write_jsonl(destination / "val.jsonl", val)
    files = {
        filename: {
            "records": len(rows),
            "sha256": sha256_file(destination / filename),
        }
        for filename, rows in (("train.jsonl", train), ("val.jsonl", val))
    }
    manifest = {
        "schema_version": 1,
        "contract_sha256": contract_hash,
        "contract": contract,
        "quality": quality_stats,
        "files": files,
        "split": {
            "train_records": len(train),
            "validation_records": len(val),
            "train_unique_prompts": len(train_prompts),
            "validation_unique_prompts": len(val_prompts),
            "prompt_overlap": 0,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log.info(
        "Prepared %s/%s: %,d train, %,d validation",
        size,
        stage,
        len(train),
        len(val),
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = {"version", "seed", "validation_fraction", "quality", "stages"} - set(config)
    if missing:
        raise ValueError(f"SFT source config missing keys: {sorted(missing)}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*VALID_STAGES, "both"), default="both")
    parser.add_argument("--size", choices=VALID_SIZES, default="125m")
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("finetune/configs/sft_data_sources.yaml"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing prepared data after its source contract changes",
    )
    args = parser.parse_args()

    config = load_config(args.source_config)
    stages = VALID_STAGES if args.stage == "both" else (args.stage,)
    for stage in stages:
        prepare_stage(config, stage, args.size, args.force)


if __name__ == "__main__":
    main()
