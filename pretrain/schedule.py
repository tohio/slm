"""Resolve pretraining duration from verified tokenized artifacts."""

from __future__ import annotations

import copy
import math

from config.data_mix import epochs as configured_epochs


def resolve_realized_token_schedule(
    cfg: dict,
    *,
    run_size: str,
    realized_train_tokens: int,
    seq_len: int,
    world_size: int,
) -> tuple[dict, dict | None]:
    """Resolve production steps from usable tokenized training tokens.

    Static YAML steps remain a planning fallback and preserve bounded mini and
    smoke recipes. Production configs opt in explicitly.
    """
    resolved = copy.deepcopy(cfg)
    training = resolved["training"]
    if not training.get("schedule_from_realized_tokens", False):
        return resolved, None

    for label, value in (
        ("realized_train_tokens", realized_train_tokens),
        ("seq_len", seq_len),
        ("world_size", world_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label} must be a positive integer, got {value!r}")

    try:
        epoch_count = configured_epochs(run_size)
    except KeyError as exc:
        raise ValueError(
            f"No configured epoch contract for run size {run_size!r}"
        ) from exc

    micro_batch = training["micro_batch_size"]
    grad_accum = training.get("gradient_accumulation_steps", 1)
    for label, value in (
        ("micro_batch_size", micro_batch),
        ("gradient_accumulation_steps", grad_accum),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label} must be a positive integer, got {value!r}")

    usable_train_tokens = (realized_train_tokens // seq_len) * seq_len
    if usable_train_tokens <= 0:
        raise ValueError(
            f"realized_train_tokens={realized_train_tokens:,} cannot form one "
            f"seq_len={seq_len:,} training example"
        )
    tokens_per_step = micro_batch * grad_accum * world_size * seq_len
    target_consumed_tokens = usable_train_tokens * epoch_count
    max_steps = math.ceil(target_consumed_tokens / tokens_per_step)

    planned_max_steps = training.get("max_steps")
    planned_warmup_steps = training.get("warmup_steps", 0)
    if (
        not isinstance(planned_max_steps, int)
        or isinstance(planned_max_steps, bool)
        or planned_max_steps <= 0
    ):
        raise ValueError("Configured max_steps must be a positive integer")
    if (
        not isinstance(planned_warmup_steps, int)
        or isinstance(planned_warmup_steps, bool)
        or planned_warmup_steps < 0
    ):
        raise ValueError("Configured warmup_steps must be a non-negative integer")

    warmup_ratio = planned_warmup_steps / planned_max_steps
    warmup_steps = min(max_steps, round(max_steps * warmup_ratio))
    scheduled_tokens = max_steps * tokens_per_step
    schedule = {
        "basis": "verified_tokenized_train_tokens",
        "run_size": run_size,
        "epochs": epoch_count,
        "realized_train_tokens": realized_train_tokens,
        "usable_train_tokens_per_epoch": usable_train_tokens,
        "tokens_discarded_by_sequence_packing": (
            realized_train_tokens - usable_train_tokens
        ),
        "world_size": world_size,
        "global_batch_sequences": micro_batch * grad_accum * world_size,
        "tokens_per_step": tokens_per_step,
        "target_consumed_tokens": target_consumed_tokens,
        "max_steps": max_steps,
        "scheduled_tokens": scheduled_tokens,
        "rounding_excess_tokens": scheduled_tokens - target_consumed_tokens,
        "warmup_ratio_from_planning_config": warmup_ratio,
        "warmup_steps": warmup_steps,
        "planning_max_steps_replaced": planned_max_steps,
        "planning_warmup_steps_replaced": planned_warmup_steps,
    }
    training["max_steps"] = max_steps
    training["warmup_steps"] = warmup_steps
    resolved["realized_token_schedule"] = schedule
    return resolved, schedule
