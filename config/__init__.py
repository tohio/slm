"""
config/
-------
Shared configuration for the SLM training pipeline.

This package is the single source of truth for values referenced across
multiple stages (data curation, pretraining, SFT, DPO, export, notebooks).
If a value exists here, no other file should hardcode it.

Organising principle: if changing a value would break a locked contract
(data mix, token budgets, tokenizer special IDs, shuffle RAM budget, etc.)
it belongs here. If it's a stage-specific tuning knob (SFT learning rate,
DPO beta, eval few-shot counts) it belongs in the stage's own config.
"""

from config.data_mix import (
    # Data mix
    DATA_MIX,
    CODE_SUBMIX,
    OVERFLOW_SINK,
    NON_CODE_SOURCES,
    CODE_SOURCES,
    ALL_SOURCES,
    # Token targets
    TARGET_CONFIGS,
    # Curator constants
    CHARS_PER_TOKEN,
    CC_CHARS_PER_SEGMENT,
    SHUFFLE_RAM_BUDGET_GB,
    PRETRAIN_VAL_FRACTION,
    MINI_OVERRIDES,
    # Helpers
    dataset_link,
    total_tokens,
    token_target_display,
    epochs,
    validate,
)

__all__ = [
    "DATA_MIX",
    "CODE_SUBMIX",
    "OVERFLOW_SINK",
    "NON_CODE_SOURCES",
    "CODE_SOURCES",
    "ALL_SOURCES",
    "TARGET_CONFIGS",
    "CHARS_PER_TOKEN",
    "CC_CHARS_PER_SEGMENT",
    "SHUFFLE_RAM_BUDGET_GB",
    "PRETRAIN_VAL_FRACTION",
    "MINI_OVERRIDES",
    "dataset_link",
    "total_tokens",
    "token_target_display",
    "epochs",
    "validate",
]