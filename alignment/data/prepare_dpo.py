#!/usr/bin/env python3
"""Prepare a pinned external preference dataset for chat DPO.

The SLM repository consumes DPO data; it does not generate or repeat synthetic
preferences. Source identity, revision, row cap, token budget, and quality
thresholds live in ``alignment/configs/dpo_data_sources.yaml``.

Outputs are conversational JSONL plus a provenance/integrity manifest. Splits
are grouped by normalized prompt so variants of one prompt cannot cross the
train/validation boundary.
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

from config.paths import dpo_chat_data_dir, tokenizer_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_SYSTEM = "You are a helpful, harmless, and honest assistant."
VALID_SIZES = ("mini", "125m", "350m", "1b")
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


def normalize_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    for item in value:
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
    return messages or None


def normalize_prompt(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, str) and value.strip():
        messages = [{"role": "user", "content": value.strip()}]
    else:
        messages = normalize_messages(value)
    if not messages:
        return None
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})
    if any(message["role"] == "system" for message in messages[1:]):
        return None
    conversation = messages[1:]
    if not conversation or conversation[0]["role"] != "user":
        return None
    if conversation[-1]["role"] != "user":
        return None
    if any(
        left["role"] == right["role"]
        for left, right in zip(conversation, conversation[1:])
    ):
        return None
    return messages


def canonical_prompt(messages: list[dict[str, str]]) -> str:
    return canonical_json(
        [
            {"role": message["role"], "content": canonical_text(message["content"])}
            for message in messages
        ]
    )


def response_text(messages: list[dict[str, str]]) -> str:
    return messages[0]["content"]


def normalize_response_side(value: Any) -> tuple[list[dict[str, str]], str] | None:
    if isinstance(value, str) and value.strip():
        return [], value.strip()
    messages = normalize_messages(value)
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages[:-1], messages[-1]["content"].strip()


def normalize_preference(row: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    chosen_side = normalize_response_side(row.get("chosen"))
    rejected_side = normalize_response_side(row.get("rejected"))
    if not chosen_side or not rejected_side:
        return None
    chosen_prefix, chosen = chosen_side
    rejected_prefix, rejected = rejected_side
    if bool(chosen_prefix) != bool(rejected_prefix):
        return None
    if chosen_prefix and rejected_prefix:
        chosen_prompt = normalize_prompt(chosen_prefix)
        rejected_prompt = normalize_prompt(rejected_prefix)
        if (
            chosen_prompt is None
            or rejected_prompt is None
            or canonical_prompt(chosen_prompt) != canonical_prompt(rejected_prompt)
        ):
            return None
        prompt = chosen_prompt
    else:
        prompt = normalize_prompt(row.get("prompt"))
    if prompt is None:
        return None

    if not chosen or not rejected or canonical_text(chosen) == canonical_text(rejected):
        return None

    record = {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "source": source_name,
        "dpo_type": "general_preference",
    }
    source_id = row.get("id", row.get("prompt_id"))
    if source_id is not None:
        record["source_id"] = str(source_id)
    return record


def load_source(source: dict[str, Any]):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": source["split"],
        "revision": source["revision"],
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    if os.environ.get("HF_TOKEN"):
        kwargs["token"] = os.environ["HF_TOKEN"]
    return load_dataset(source["dataset"], **kwargs)


def select_source_rows(dataset, cap: int, seed: int):
    if cap <= 0:
        raise ValueError(f"max_records must be positive, got {cap}")
    if len(dataset) <= cap:
        return dataset
    return dataset.shuffle(seed=seed).select(range(cap))


def prepare_records(
    dataset, source: dict[str, Any], quality: dict[str, float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source["format"] != "conversational_preference":
        raise ValueError(f"Unsupported DPO source format: {source['format']!r}")

    valid: list[dict[str, Any]] = []
    invalid = 0
    for row in dataset:
        record = normalize_preference(row, source["source"])
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
    reverse_conflicts = 0
    seen_directional: set[str] = set()
    for record in valid:
        prompt_key = canonical_prompt(record["prompt"])
        chosen_key = canonical_text(response_text(record["chosen"]))
        rejected_key = canonical_text(response_text(record["rejected"]))
        pair_key = sha256_json((prompt_key, chosen_key, rejected_key))
        reverse_key = sha256_json((prompt_key, rejected_key, chosen_key))
        if reverse_key in seen_directional:
            reverse_conflicts += 1
        seen_directional.add(pair_key)
        unique.setdefault(pair_key, record)
        prompt_counts[prompt_key] += 1

    duplicate_count = len(valid) - len(unique)
    duplicate_fraction = duplicate_count / len(valid) if valid else 1.0
    if duplicate_fraction > float(quality["max_duplicate_fraction"]):
        raise RuntimeError(
            f"{source['source']} contains {duplicate_count}/{len(valid)} exact "
            f"preference duplicates ({duplicate_fraction:.2%}), above "
            f"max_duplicate_fraction={float(quality['max_duplicate_fraction']):.2%}"
        )
    if reverse_conflicts > int(quality["max_reverse_conflicts"]):
        raise RuntimeError(
            f"{source['source']} contains {reverse_conflicts} reversed preference "
            "conflicts. Fix the source dataset."
        )

    unique_prompt_ratio = len(prompt_counts) / len(valid) if valid else 0.0
    if unique_prompt_ratio < float(quality["min_unique_prompt_ratio"]):
        raise RuntimeError(
            f"{source['source']} unique-prompt ratio is {unique_prompt_ratio:.2%}, "
            f"below min_unique_prompt_ratio="
            f"{float(quality['min_unique_prompt_ratio']):.2%}"
        )

    records = list(unique.values())
    return records, {
        "selected_rows": total,
        "valid_rows": len(valid),
        "invalid_rows": invalid,
        "invalid_fraction": invalid_fraction,
        "exact_duplicates_removed": duplicate_count,
        "duplicate_fraction": duplicate_fraction,
        "reverse_conflicts": reverse_conflicts,
        "unique_pairs": len(records),
        "unique_prompts": len(prompt_counts),
        "unique_prompt_ratio": unique_prompt_ratio,
        "maximum_prompt_multiplicity": max(prompt_counts.values(), default=0),
    }


def chat_template_ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    return list(encoded)


def apply_token_budget(
    records: list[dict[str, Any]],
    tokenizer,
    max_prompt_tokens: int,
    max_total_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < max_prompt_tokens < max_total_tokens:
        raise ValueError("Token budgets require 0 < max_prompt_tokens < max_total_tokens")

    kept: list[dict[str, Any]] = []
    prompt_too_long = 0
    pair_too_long = 0
    prefix_mismatch = 0
    for record in records:
        prompt_ids = chat_template_ids(tokenizer, record["prompt"], True)
        chosen_ids = chat_template_ids(tokenizer, record["prompt"] + record["chosen"], False)
        rejected_ids = chat_template_ids(tokenizer, record["prompt"] + record["rejected"], False)
        if chosen_ids[: len(prompt_ids)] != prompt_ids or rejected_ids[: len(prompt_ids)] != prompt_ids:
            prefix_mismatch += 1
            continue
        if len(prompt_ids) > max_prompt_tokens:
            prompt_too_long += 1
            continue
        if max(len(chosen_ids), len(rejected_ids)) > max_total_tokens:
            pair_too_long += 1
            continue
        kept.append(record)

    if prefix_mismatch:
        raise RuntimeError(
            f"{prefix_mismatch} DPO rows have prompt/completion tokenization prefix "
            "mismatches. Fix the tokenizer template or source formatting."
        )
    return kept, {
        "input_pairs": len(records),
        "retained_pairs": len(kept),
        "retention_ratio": len(kept) / len(records) if records else 0.0,
        "prompt_too_long": prompt_too_long,
        "pair_too_long": pair_too_long,
        "prefix_mismatch": prefix_mismatch,
    }


def grouped_split(
    records: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(canonical_prompt(record["prompt"]), []).append(record)
    if len(groups) < 2:
        raise RuntimeError("At least two unique DPO prompts are required")

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(records) * validation_fraction))
    validation_keys: set[str] = set()
    validation_count = 0
    for key in keys:
        if validation_count >= target and validation_keys:
            break
        validation_keys.add(key)
        validation_count += len(groups[key])

    train = [record for key in keys if key not in validation_keys for record in groups[key]]
    validation = [record for key in keys if key in validation_keys for record in groups[key]]
    random.Random(seed + 1).shuffle(train)
    random.Random(seed + 2).shuffle(validation)
    return train, validation


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"version", "seed", "validation_fraction", "quality", "source"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"DPO source config missing keys: {sorted(missing)}")
    return config


def source_contract(
    config: dict[str, Any],
    size: str,
    tokenizer_path: Path,
) -> dict[str, Any]:
    source = dict(config["source"])
    record_caps = source.pop("max_records_by_size")
    prompt_budgets = source.pop("max_prompt_tokens_by_size")
    total_budgets = source.pop("max_total_tokens_by_size")
    return {
        "config_version": config["version"],
        "size": size,
        "seed": int(config["seed"]),
        "validation_fraction": float(config["validation_fraction"]),
        "quality": config["quality"],
        "source": source,
        "max_records": int(record_caps[size]),
        "max_prompt_tokens": int(prompt_budgets[size]),
        "max_total_tokens": int(total_budgets[size]),
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer_path),
    }


def prepare(config: dict[str, Any], size: str, force: bool) -> None:
    from transformers import PreTrainedTokenizerFast

    destination = dpo_chat_data_dir(size)
    manifest_path = destination / "manifest.json"
    tokenizer_path = tokenizer_dir(size)
    if not (tokenizer_path / "tokenizer_config.json").exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run tokenizer preparation first."
        )
    contract = source_contract(config, size, tokenizer_path)
    contract_hash = sha256_json(contract)

    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") == contract_hash:
            for filename in ("train.jsonl", "val.jsonl"):
                path = destination / filename
                expected = existing.get("files", {}).get(filename, {}).get("sha256")
                if not path.exists() or not expected or sha256_file(path) != expected:
                    raise RuntimeError(
                        f"{path} is missing or differs from its manifest. "
                        "Restore it or rerun intentionally with --force."
                    )
            log.info("%s DPO data matches its source contract; reusing it", size)
            return
        raise RuntimeError(
            f"{destination} was prepared from a different DPO contract. "
            "Inspect or archive it, then rerun with --force."
        )
    if not manifest_path.exists() and destination.exists() and any(destination.iterdir()) and not force:
        raise RuntimeError(
            f"{destination} contains unmanifested data. Inspect or archive it, "
            "then rerun with --force."
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    source = contract["source"]
    log.info(
        "Loading %s@%s (%s)",
        source["dataset"],
        source["revision"],
        source["split"],
    )
    dataset = select_source_rows(
        load_source(source),
        contract["max_records"],
        contract["seed"],
    )
    records, quality_stats = prepare_records(dataset, source, contract["quality"])
    records, token_stats = apply_token_budget(
        records,
        tokenizer,
        contract["max_prompt_tokens"],
        contract["max_total_tokens"],
    )
    minimum_token_retention = float(
        contract["quality"].get("min_token_retention_ratio", 0.0)
    )
    if token_stats["retention_ratio"] < minimum_token_retention:
        raise RuntimeError(
            f"DPO token filtering retained only {token_stats['retention_ratio']:.2%}; "
            f"required >= {minimum_token_retention:.2%}. Revise the source or "
            "token budgets before training."
        )
    if len(records) < 2:
        raise RuntimeError("Too few DPO pairs remain after validation and token filtering")
    train, validation = grouped_split(
        records,
        contract["validation_fraction"],
        contract["seed"],
    )

    train_prompts = {canonical_prompt(record["prompt"]) for record in train}
    validation_prompts = {canonical_prompt(record["prompt"]) for record in validation}
    if train_prompts & validation_prompts:
        raise AssertionError("Internal error: DPO prompt leakage across splits")

    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "train.jsonl", train)
    write_jsonl(destination / "val.jsonl", validation)
    files = {
        filename: {
            "records": len(rows),
            "sha256": sha256_file(destination / filename),
        }
        for filename, rows in (("train.jsonl", train), ("val.jsonl", validation))
    }
    manifest = {
        "schema_version": 1,
        "contract_sha256": contract_hash,
        "contract": contract,
        "quality": quality_stats,
        "token_filter": token_stats,
        "files": files,
        "split": {
            "train_pairs": len(train),
            "validation_pairs": len(validation),
            "train_unique_prompts": len(train_prompts),
            "validation_unique_prompts": len(validation_prompts),
            "prompt_overlap": 0,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log.info(
        "Prepared %s DPO: %,d train, %,d validation",
        size,
        len(train),
        len(validation),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        choices=VALID_SIZES,
        default=os.environ.get("SIZE", "125m"),
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("alignment/configs/dpo_data_sources.yaml"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace prepared data after intentionally changing its contract",
    )
    args = parser.parse_args()
    prepare(load_config(args.source_config), args.size, args.force)


if __name__ == "__main__":
    main()
