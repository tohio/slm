#!/usr/bin/env python3
"""Controlled, fail-fast SFT response comparison.

This is a diagnostic harness, not a production training recipe. It compares
two base checkpoints on the same semantic records, optimizer updates, batch
order, completion-only objective, and deterministic prompts. Each model keeps
its own tokenizer and chat template, so token counts are measured and reported
rather than incorrectly described as equal.

The harness owns tokenization and labels. This makes dataset retention explicit
and prevents a TRL preprocessing change from silently giving the models
different examples. Production TRL compatibility is covered separately by
tests/test_trl_smoke.py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
from statistics import mean, median
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
DEFAULT_EXPORTS_DIR = Path(
    os.environ.get("EXPORTS_DIR", str(DEFAULT_RESULTS_DIR / "exports"))
)
SYSTEM_PROMPT = "You are a helpful assistant."
HARNESS_VERSION = "3.0"

SMOL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% generation %}"
    "{{ message['content'] + '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)

EVALUATION_CASES = [
    {
        "id": "simple_explanation",
        "prompt": "Explain what a neural network is in one simple sentence.",
    },
    {
        "id": "exact_count",
        "prompt": "Give me exactly two quick tips for writing clean Python code.",
    },
    {
        "id": "capital",
        "prompt": "What is the capital of France?",
        "expected_contains": "paris",
    },
    {
        "id": "arithmetic",
        "prompt": "What is 7 + 5? Answer with only the number.",
        "expected_exact": "12",
    },
    {
        "id": "exact_repetition",
        "prompt": "Repeat the word blue exactly three times, separated by spaces.",
        "expected_exact": "blue blue blue",
    },
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_ref: str
    tokenizer_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare checkpoint integrity and response to controlled SFT."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("smol", "tohio"),
        default=["smol", "tohio"],
    )
    parser.add_argument(
        "--smol-model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Native Transformers Smol checkpoint or Hub ID.",
    )
    parser.add_argument(
        "--tohio-model",
        default=str(DEFAULT_EXPORTS_DIR / "125m" / "base"),
        help=(
            "Native Transformers Tohio export or republished Hub ID. Run "
            "'make export-base-local SIZE=125m' to create the default."
        ),
    )
    parser.add_argument("--train-examples", type=int, default=32)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--candidate-pool", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run checkpoint/cache diagnostics without downloading data or training.",
    )
    parser.add_argument(
        "--allow-cache-mismatch",
        action="store_true",
        help=(
            "Continue after cached/uncached divergence. Intended only to "
            "isolate a known cache defect; canonical comparison output remains "
            "uncached."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics/sft-comparison"),
    )
    return parser.parse_args()


def choose_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    os.environ["ACCELERATE_USE_CPU"] = "true"
    if torch.backends.mps.is_available():
        print("WARNING: bypassing MPS for deterministic comparison; using CPU.")
    return "cpu", torch.float32


def reset_random_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_specs(args: argparse.Namespace) -> dict[str, ModelSpec]:
    return {
        "smol": ModelSpec("smol", args.smol_model, "smol"),
        "tohio": ModelSpec("tohio", args.tohio_model, "native"),
    }


def load_comparison_tokenizer(spec: ModelSpec):
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_ref,
        trust_remote_code=False,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"{spec.model_ref} must provide a fast tokenizer.")

    if spec.tokenizer_kind == "smol":
        required = ("<|im_start|>", "<|im_end|>")
        missing = [
            token
            for token in required
            if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id
        ]
        if missing:
            raise RuntimeError(
                f"{spec.model_ref} is missing chat control tokens: {missing}"
            )
        tokenizer.chat_template = SMOL_CHAT_TEMPLATE
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|im_end|>"
    else:
        if not tokenizer.chat_template:
            raise RuntimeError(f"{spec.model_ref} has no chat template.")
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise RuntimeError(
                f"{spec.model_ref} must define pad_token_id and eos_token_id."
            )

    return tokenizer


def chat_messages(
    user_text: str,
    assistant_text: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def first_user_assistant_pair(messages: list[dict]) -> tuple[str, str] | None:
    for index in range(len(messages) - 1):
        current = messages[index]
        following = messages[index + 1]
        if current.get("role") == "user" and following.get("role") == "assistant":
            user = str(current.get("content", "")).strip()
            assistant = str(following.get("content", "")).strip()
            if user and assistant:
                return user, assistant
    return None


def _tokenized_training_row(
    tokenizer,
    user_text: str,
    assistant_text: str,
    max_length: int,
) -> dict[str, list[int]] | None:
    """Build one exact completion-only example without trainer preprocessing."""
    prompt_text = tokenizer.apply_chat_template(
        chat_messages(user_text),
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        chat_messages(user_text, assistant_text),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise RuntimeError(
            "Full conversation does not begin with its generation prompt."
        )

    supervised_text = assistant_text + tokenizer.eos_token
    if not full_text[len(prompt_text):].startswith(supervised_text):
        raise RuntimeError(
            "Assistant rendering does not begin with content followed by EOS."
        )

    # Exclude masked trailing turn separators. They cannot affect a final-token
    # objective and should not be counted as supervised assistant output.
    training_text = prompt_text + supervised_text
    encoded = tokenizer(
        training_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(input_ids) > max_length:
        return None

    boundary = len(prompt_text)
    labels = []
    crossed_boundary = False
    for token_id, (start, end) in zip(input_ids, offsets, strict=True):
        if end <= boundary:
            labels.append(-100)
        elif start >= boundary:
            labels.append(token_id)
        else:
            crossed_boundary = True
            break
    if crossed_boundary:
        return None

    supervised = [token for token in labels if token != -100]
    if not supervised or tokenizer.eos_token_id not in supervised:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"minimum": 0, "median": 0, "mean": 0, "p95": 0, "maximum": 0}
    return {
        "minimum": min(values),
        "median": median(values),
        "mean": mean(values),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _sft_source_contract() -> dict[str, Any]:
    path = ROOT / "finetune" / "configs" / "sft_data_sources.yaml"
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    return contract["stages"]["instruct"]


def build_common_dataset(
    tokenizers: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, list[dict]], dict]:
    required = args.train_examples + args.eval_examples
    if args.candidate_pool < required:
        raise ValueError("--candidate-pool must cover train + eval examples.")

    source_contract = _sft_source_contract()
    source = load_dataset(
        source_contract["dataset"],
        split=f"{source_contract['split']}[:{args.candidate_pool}]",
        revision=source_contract["revision"],
    ).shuffle(seed=args.seed)

    accepted: list[dict] = []
    tokenized = {name: [] for name in tokenizers}
    rejected = {"missing_pair": 0, **{name: 0 for name in tokenizers}}

    for shuffled_index, raw_row in enumerate(source):
        pair = first_user_assistant_pair(raw_row["messages"])
        if pair is None:
            rejected["missing_pair"] += 1
            continue

        user_text, assistant_text = pair
        model_rows = {
            name: _tokenized_training_row(
                tokenizer,
                user_text,
                assistant_text,
                args.max_length,
            )
            for name, tokenizer in tokenizers.items()
        }
        invalid_for = [name for name, row in model_rows.items() if row is None]
        if invalid_for:
            for name in invalid_for:
                rejected[name] += 1
            continue

        canonical = json.dumps(
            {"user": user_text, "assistant": assistant_text},
            ensure_ascii=False,
            sort_keys=True,
        )
        accepted.append(
            {
                "id": raw_row.get("prompt_id")
                or hashlib.sha256(canonical.encode()).hexdigest(),
                "source_index_after_shuffle": shuffled_index,
                "messages": chat_messages(user_text, assistant_text),
            }
        )
        for name, row in model_rows.items():
            tokenized[name].append(row)

        if len(accepted) == required:
            break

    if len(accepted) != required:
        raise RuntimeError(
            f"Found {len(accepted)} jointly eligible examples; need {required}. "
            "Increase --candidate-pool or --max-length."
        )

    report = {
        "source": source_contract,
        "candidate_pool": args.candidate_pool,
        "candidates_examined": accepted[-1]["source_index_after_shuffle"] + 1,
        "selected_train_examples": args.train_examples,
        "selected_eval_examples": args.eval_examples,
        "rejected": rejected,
        "model_token_statistics": {},
    }
    for name, rows in tokenized.items():
        total = [len(row["input_ids"]) for row in rows]
        supervised = [
            sum(token != -100 for token in row["labels"])
            for row in rows
        ]
        report["model_token_statistics"][name] = {
            "all_selected_sequence_tokens": summarize(total),
            "all_selected_supervised_tokens": summarize(supervised),
            "train_supervised_token_total": sum(
                supervised[: args.train_examples]
            ),
            "eval_supervised_token_total": sum(
                supervised[args.train_examples :]
            ),
        }

    return accepted, tokenized, report


@dataclass
class CompletionOnlyCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * padding
            )
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {
            name: torch.tensor(values, dtype=torch.long)
            for name, values in batch.items()
        }


def _chat_inputs(tokenizer, prompt: str, device: torch.device):
    encoded = tokenizer.apply_chat_template(
        chat_messages(prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return encoded.to(device)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def generate_cases(
    model,
    tokenizer,
    max_new_tokens: int,
) -> list[dict]:
    """Canonical capability output intentionally does not use the KV cache."""
    device = next(model.parameters()).device
    model.eval()
    records = []
    with torch.no_grad():
        for case in EVALUATION_CASES:
            inputs = _chat_inputs(tokenizer, case["prompt"], device)
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            response_ids = output[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            ).strip()
            normalized = _normalize(response)
            record = {
                "id": case["id"],
                "prompt": case["prompt"],
                "response": response,
                "generated_tokens": int(response_ids.numel()),
                "ended_with_eos": bool(
                    response_ids.numel()
                    and response_ids[-1].item() == tokenizer.eos_token_id
                ),
            }
            if "expected_contains" in case:
                record["passed_expected_contains"] = (
                    case["expected_contains"] in normalized
                )
            if "expected_exact" in case:
                record["passed_expected_exact"] = (
                    normalized == case["expected_exact"]
                )
            records.append(record)
    return records


def cache_parity(model, tokenizer, max_new_tokens: int = 16) -> dict:
    device = next(model.parameters()).device
    inputs = _chat_inputs(
        tokenizer,
        "What is the capital of France?",
        device,
    )
    outputs = {}
    model.eval()
    with torch.no_grad():
        for use_cache in (False, True):
            outputs[use_cache] = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )[0].detach().cpu()

    cached = outputs[True]
    uncached = outputs[False]
    mismatch = None
    for index, (left, right) in enumerate(
        zip(cached.tolist(), uncached.tolist(), strict=False)
    ):
        if left != right:
            mismatch = index
            break
    if mismatch is None and cached.numel() != uncached.numel():
        mismatch = min(cached.numel(), uncached.numel())

    prompt_length = inputs["input_ids"].shape[1]
    return {
        "matches": torch.equal(cached, uncached),
        "first_mismatch_token_position": mismatch,
        "cached_response": tokenizer.decode(
            cached[prompt_length:],
            skip_special_tokens=True,
        ).strip(),
        "uncached_response": tokenizer.decode(
            uncached[prompt_length:],
            skip_special_tokens=True,
        ).strip(),
    }


def prompt_sensitivity(model, tokenizer) -> dict:
    device = next(model.parameters()).device
    model.eval()
    last_logits = []
    with torch.no_grad():
        for prompt in ("What is the capital of France?", "Write a Python loop."):
            inputs = _chat_inputs(tokenizer, prompt, device)
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
            last_logits.append(logits)
    maximum_difference = float((last_logits[0] - last_logits[1]).abs().max())
    return {
        "last_token_max_abs_difference": maximum_difference,
        "passes": maximum_difference > 1e-6,
    }


def checkpoint_integrity(model, tokenizer) -> dict:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    nonfinite_tensors = []
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all():
            nonfinite_tensors.append(name)
    return {
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "tokenizer_length": len(tokenizer),
        "config_vocab_size": model.config.vocab_size,
        "input_embedding_rows": input_embeddings.num_embeddings,
        "output_embedding_rows": getattr(
            output_embeddings,
            "out_features",
            None,
        ),
        "nonfinite_parameter_tensors": nonfinite_tensors,
    }


def align_special_tokens(model, tokenizer) -> None:
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, field)
        setattr(model.config, field, value)
        setattr(model.generation_config, field, value)


def load_model(
    spec: ModelSpec,
    device: str,
    dtype: torch.dtype,
):
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_ref,
        trust_remote_code=False,
        dtype=dtype,
        device_map=None,
        weights_only=True,
    ).to(device)
    return model


def run_preflight(
    spec: ModelSpec,
    tokenizer,
    device: str,
    dtype: torch.dtype,
) -> dict:
    model = load_model(spec, device, dtype)
    align_special_tokens(model, tokenizer)
    integrity = checkpoint_integrity(model, tokenizer)
    vocab_values = {
        integrity["tokenizer_length"],
        integrity["config_vocab_size"],
        integrity["input_embedding_rows"],
    }
    if len(vocab_values) != 1:
        raise RuntimeError(f"{spec.name} vocabulary mismatch: {integrity}")
    if integrity["nonfinite_parameter_tensors"]:
        raise RuntimeError(
            f"{spec.name} contains non-finite parameters: "
            f"{integrity['nonfinite_parameter_tensors'][:5]}"
        )

    report = {
        "checkpoint_integrity": integrity,
        "prompt_sensitivity": prompt_sensitivity(model, tokenizer),
        "cache_parity": cache_parity(model, tokenizer),
        "uncached_generation": generate_cases(model, tokenizer, 24),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _training_arguments(
    args: argparse.Namespace,
    output_dir: Path,
    device: str,
    dtype: torch.dtype,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        logging_steps=max(1, min(10, args.max_steps)),
        logging_strategy="steps",
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        use_cpu=(device == "cpu"),
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )


def run_training_comparison(
    spec: ModelSpec,
    tokenizer,
    rows: list[dict],
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
) -> dict:
    reset_random_state(args.seed)
    model = load_model(spec, device, dtype)
    align_special_tokens(model, tokenizer)
    model.config.use_cache = False

    train_rows = rows[: args.train_examples]
    eval_rows = rows[args.train_examples :]
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)
    trainer = Trainer(
        model=model,
        args=_training_arguments(
            args,
            args.output_dir / spec.name,
            device,
            dtype,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    if len(trainer.train_dataset) != args.train_examples:
        raise RuntimeError("Trainer changed the selected training records.")
    if len(trainer.eval_dataset) != args.eval_examples:
        raise RuntimeError("Trainer changed the selected evaluation records.")

    print(f"\nRunning {spec.name} pre-SFT evaluation...")
    pre_loss = trainer.evaluate(metric_key_prefix="pre_sft")
    pre_generation = generate_cases(model, tokenizer, args.max_new_tokens)

    print(f"Training {spec.name} for {args.max_steps} optimizer steps...")
    training = trainer.train()

    print(f"Running {spec.name} post-SFT evaluation...")
    post_loss = trainer.evaluate(metric_key_prefix="post_sft")
    post_generation = generate_cases(
        trainer.model,
        tokenizer,
        args.max_new_tokens,
    )

    result = {
        "model": asdict(spec),
        "selected_examples": {
            "train": len(train_dataset),
            "eval": len(eval_dataset),
        },
        "pre_sft": {
            "evaluation_metrics": pre_loss,
            "uncached_generation": pre_generation,
        },
        "training_metrics": training.metrics,
        "post_sft": {
            "evaluation_metrics": post_loss,
            "uncached_generation": post_generation,
        },
    }
    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _save_selected(
    path: Path,
    accepted: list[dict],
    train_examples: int,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(accepted):
            output = {
                **record,
                "comparison_split": (
                    "train" if index < train_examples else "eval"
                ),
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def print_summary(report: dict) -> None:
    print("\n" + "=" * 72)
    print("CONTROLLED SFT COMPARISON")
    print("=" * 72)
    for name, result in report.get("results", {}).items():
        pre = result["pre_sft"]["evaluation_metrics"]["pre_sft_loss"]
        post = result["post_sft"]["evaluation_metrics"]["post_sft_loss"]
        print(f"{name}: held-out loss {pre:.6f} -> {post:.6f}")
        for stage in ("pre_sft", "post_sft"):
            print(f"  {stage} uncached output:")
            for case in result[stage]["uncached_generation"]:
                print(f"    [{case['id']}] {case['response']}")


def main() -> None:
    args = parse_args()
    if min(args.train_examples, args.eval_examples, args.max_steps) <= 0:
        raise ValueError("Example counts and max steps must be positive.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device, dtype = choose_device()
    reset_random_state(args.seed)
    specs = model_specs(args)

    tokenizers = {}
    preflight = {}
    for name in args.models:
        spec = specs[name]
        print(f"\nPreflight: {name} ({spec.model_ref})")
        tokenizers[name] = load_comparison_tokenizer(spec)
        preflight[name] = run_preflight(
            spec,
            tokenizers[name],
            device,
            dtype,
        )

    preflight_report = {
        "harness_version": HARNESS_VERSION,
        "device": device,
        "dtype": str(dtype),
        "models": {
            name: asdict(specs[name])
            for name in args.models
        },
        "results": preflight,
    }
    _write_json(args.output_dir / "preflight_report.json", preflight_report)

    cache_failures = [
        name
        for name, result in preflight.items()
        if not result["cache_parity"]["matches"]
    ]
    sensitivity_failures = [
        name
        for name, result in preflight.items()
        if not result["prompt_sensitivity"]["passes"]
    ]
    if sensitivity_failures:
        raise RuntimeError(
            "Prompt-sensitivity preflight failed for "
            f"{sensitivity_failures}; refusing to train."
        )
    if cache_failures and not args.allow_cache_mismatch:
        raise RuntimeError(
            "Cached/uncached generation diverged for "
            f"{cache_failures}. Fix or export the checkpoint before spending "
            "time on SFT, or pass --allow-cache-mismatch for deliberate "
            "uncached-only diagnosis."
        )
    if args.preflight_only:
        print(f"Preflight report: {args.output_dir / 'preflight_report.json'}")
        return

    print("\nSelecting one pinned, jointly eligible dataset...")
    accepted, tokenized, selection = build_common_dataset(tokenizers, args)
    _save_selected(
        args.output_dir / "selected_examples.jsonl",
        accepted,
        args.train_examples,
    )

    configuration = {
        "harness_version": HARNESS_VERSION,
        "models": {
            name: asdict(specs[name])
            for name in args.models
        },
        "train_examples": args.train_examples,
        "eval_examples": args.eval_examples,
        "candidate_pool": args.candidate_pool,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "device": device,
        "dtype": str(dtype),
        "comparison_contract": {
            "same_semantic_records": True,
            "same_record_order": True,
            "same_optimizer_updates": True,
            "same_optimizer_hyperparameters": True,
            "completion_only_labels": True,
            "canonical_generation_uses_cache": False,
            "token_counts_equal": False,
            "token_count_note": (
                "Tokenizer-specific sequence and supervised-token totals are "
                "reported; different tokenizers cannot have identical tokens."
            ),
        },
        "dataset_selection": selection,
    }
    _write_json(args.output_dir / "run_configuration.json", configuration)

    report = {
        "run_configuration": configuration,
        "preflight": preflight,
        "results": {},
    }
    for name in args.models:
        report["results"][name] = run_training_comparison(
            specs[name],
            tokenizers[name],
            tokenized[name],
            args,
            device,
            dtype,
        )
        _write_json(args.output_dir / "comparison_report.json", report)

    print_summary(report)
    print(f"\nReport: {args.output_dir / 'comparison_report.json'}")


if __name__ == "__main__":
    main()
