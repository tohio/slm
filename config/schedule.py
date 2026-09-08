"""SFT/DPO update and warmup planning for the pinned Trainer/DDP semantics."""
from __future__ import annotations

import logging
import math
import os

log = logging.getLogger(__name__)


def resolve_total_steps(train_cfg: dict, num_train_examples: int, *,
                        default_epochs: float = 1.0) -> int:
    max_steps = int(train_cfg.get("max_steps", -1))
    if max_steps > 0:
        return max_steps
    epochs = float(train_cfg.get("epochs", default_epochs))
    if not math.isfinite(epochs) or epochs <= 0:
        raise ValueError("epochs must be a finite positive number")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro = int(train_cfg["micro_batch_size"])
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    if num_train_examples <= 0 or min(world_size, micro, accumulation) <= 0:
        raise ValueError("Training examples, batch size, accumulation and world size must be positive")
    # Accelerate's default even-batch DDP loader pads the last batch group.
    # Trainer 5.14.1 uses ceil(loader_batches / accumulation), then ceil(epochs * updates).
    batches_per_rank = math.ceil(num_train_examples / (micro * world_size))
    updates_per_epoch = math.ceil(batches_per_rank / accumulation)
    return math.ceil(epochs * updates_per_epoch)


def resolve_warmup_steps(train_cfg: dict, num_train_examples: int, *,
                         default_epochs: float = 1.0) -> int:
    if "warmup_steps" in train_cfg:
        steps = int(train_cfg["warmup_steps"])
        if steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        return steps
    if "warmup_ratio" in train_cfg:
        log.warning("Rename deprecated warmup_ratio to warmup_ratio_recipe; honoring it for this run")
    ratio = float(train_cfg.get("warmup_ratio", train_cfg.get("warmup_ratio_recipe", 0.0)))
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("warmup ratio must be finite and in [0, 1]")
    if ratio == 0:
        return 0
    total = resolve_total_steps(train_cfg, num_train_examples, default_epochs=default_epochs)
    steps = max(1, round(total * ratio))
    log.info("Warmup: %d steps (%.1f%% of %d planned updates)", steps, 100 * ratio, total)
    return steps
