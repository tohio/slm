"""
config/data_mix.py
-------------------
Single source of truth for the SLM pretraining data mix, token budgets,
and the locked curator-side constants.

Referenced by:
    - curator/scripts/curate.py   (data preparation)
    - curator/constants.py        (re-exports CHARS_PER_TOKEN etc.)
    - pretrain/train.py           (indirectly via configs and dataset.py)
    - export/export.py            (model card generation)
    - tests/conftest.py           (source list if a test asserts it)
    - notebooks/*.ipynb           (data analysis / plots)

If you change a value here you are changing the contract for the entire
training pipeline. Do not duplicate these values anywhere else.

Organising principle:
    - Locked values that multiple stages read → here.
    - Stage-specific tuning knobs (SFT LR, DPO beta, eval few-shot counts)
      stay in their stage's own config file.

Section layout:
    1. DATA_MIX                 top-level source percentages + metadata
    2. CODE_SUBMIX              sub-mix of the 10% code share
    3. OVERFLOW_SINK            which source absorbs supply deficits
    4. Source name lists        NON_CODE_SOURCES, CODE_SOURCES, ALL_SOURCES
    5. TARGET_CONFIGS           per-size token budgets, CC crawls, epochs
    6. Curator constants        CHARS_PER_TOKEN, CC_CHARS_PER_SEGMENT,
                                SHUFFLE_RAM_BUDGET_GB, PRETRAIN_VAL_FRACTION,
                                MINI_OVERRIDES
    7. Helpers                  dataset_link, total_tokens, epochs, validate
"""

from __future__ import annotations

import os


# ── 1. Top-level data mix ──────────────────────────────────────────────────────
#
# Each entry carries:
#   pct       — percentage of total pretraining tokens (scale-invariant).
#   display   — human-readable name used in docs and model cards.
#   hub       — optional HuggingFace Hub dataset id (enables rich links).
#   url       — fallback URL when the source isn't on the Hub.
#
# Percentages across DATA_MIX sum to 100. The curator reads `pct` as a float
# share (divided by 100) via the module-level CURATOR_SOURCE_MIX view below.

DATA_MIX: dict[str, dict] = {
    "common_crawl": {
        "pct":     10.0,
        "display": "Common Crawl",
        "url":     "https://commoncrawl.org",
    },
    "fineweb": {
        "pct":     47.5,
        "display": "FineWeb",
        "hub":     "HuggingFaceFW/fineweb",
    },
    "wikipedia": {
        "pct":     10.0,
        "display": "Wikipedia (EN)",
        "hub":     "wikimedia/wikipedia",
    },
    "pg19": {
        "pct":     2.5,
        "display": "PG-19 (Project Gutenberg)",
        "hub":     "pg19",
    },
    "pes2o": {
        "pct":     5.0,
        "display": "peS2o (academic papers)",
        "hub":     "allenai/peS2o",
    },
    "open_web_math": {
        "pct":     10.0,
        "display": "OpenWebMath",
        "hub":     "open-web-math/open-web-math",
    },
    "stackexchange": {
        "pct":     5.0,
        "display": "StackExchange",
        "hub":     "HuggingFaceH4/stack-exchange-preferences",
    },
    "code": {
        "pct":     10.0,
        "display": "Code (multi-source)",
        # Dispatched across CODE_SUBMIX. The "code" entry itself is a logical
        # bucket — the actual per-source char targets are computed using
        # CODE_SUBMIX percentages × the 10% code share.
    },
}


# ── 2. Code sub-mix ────────────────────────────────────────────────────────────
#
# Percentages of the 10% code share (not of total tokens). stack_v2 is capped
# at 50% so bulk raw code doesn't drown out the curated code sources.

CODE_SUBMIX: dict[str, dict] = {
    "stack_v2": {
        "pct":     50.0,
        "display": "The Stack v2 (capped)",
        "hub":     "bigcode/the-stack-v2-dedup",
    },
    "codesearchnet": {
        "pct":     35.0,
        "display": "CodeSearchNet",
        "hub":     "code_search_net",
    },
    "stack_smol": {
        "pct":     10.0,
        "display": "The Stack (smol)",
        "hub":     "bigcode/the-stack-smol",
    },
    "jupyter": {
        "pct":     4.0,
        "display": "Jupyter notebooks",
        "hub":     "bigcode/jupyter-parsed",
    },
    "conala": {
        "pct":     1.0,
        "display": "CoNaLa",
        "hub":     "neulab/conala",
    },
}


# ── 3. Overflow sink ───────────────────────────────────────────────────────────
#
# When supply-constrained sources (Wikipedia, pg19, etc.) fall short of their
# character budget, the deficit is routed to this source. FineWeb has ~15T
# tokens available, so it can always close the gap.

OVERFLOW_SINK: str = "fineweb"


# ── 4. Source name lists ───────────────────────────────────────────────────────
#
# Derived lists for iteration. The "code" key in DATA_MIX is a logical bucket;
# the concrete source names used by the curator come from CODE_SUBMIX.

NON_CODE_SOURCES: list[str] = [name for name in DATA_MIX if name != "code"]
CODE_SOURCES: list[str] = list(CODE_SUBMIX.keys())
ALL_SOURCES: list[str] = NON_CODE_SOURCES + CODE_SOURCES


# ── 5. Target configurations ───────────────────────────────────────────────────
#
# Per-size training targets. Carries everything a size-specific run needs:
#   total_tokens — the pretraining token budget
#   epochs       — number of training epochs (documentation; pretrain/train.py
#                  uses max_steps which is computed elsewhere from this)
#   cc_crawls    — Common Crawl snapshots to draw from at this scale
#   display_tokens — human-readable shorthand (5B / 15B / 30B)
#
# cc_segments is computed at runtime from total_tokens × cc_share ×
# CHARS_PER_TOKEN ÷ CC_CHARS_PER_SEGMENT — see curator/scripts/curate.py.

TARGET_CONFIGS: dict[str, dict] = {
    "mini": {
        "total_tokens":   1_000_000,
        "epochs":         1,
        "cc_crawls":      ["CC-MAIN-2024-10"],
        "display_tokens": "1M",
    },
    "125m": {
        "total_tokens":   5_000_000_000,
        "epochs":         2,
        "cc_crawls":      ["CC-MAIN-2024-10"],
        "display_tokens": "5B",
    },
    "350m": {
        "total_tokens":   15_000_000_000,
        "epochs":         2,
        "cc_crawls":      ["CC-MAIN-2024-10", "CC-MAIN-2023-50"],
        "display_tokens": "15B",
    },
    "1b": {
        "total_tokens":   30_000_000_000,
        "epochs":         1,
        "cc_crawls":      ["CC-MAIN-2024-10", "CC-MAIN-2023-50", "CC-MAIN-2023-40"],
        "display_tokens": "30B",
    },
}


# ── 6. Curator constants ───────────────────────────────────────────────────────
#
# These were previously scattered across curator/constants.py and curator/
# scripts/curate.py. Centralising means a retokenizer run (which could shift
# CHARS_PER_TOKEN) is a single-file change instead of chasing every reference.

# Average characters per BPE token from the trained tokenizer. Locked at 5.0
# based on the 32k-vocab tokenizer trained on the pretraining corpus. If the
# tokenizer is retrained on a different corpus, rerun the chars-per-token
# measurement (see tokenizer/README.md) before updating.
CHARS_PER_TOKEN: int = 5

# Empirical characters of English prose produced per Common Crawl WARC segment
# after trafilatura extraction + language filtering. Derived from the 125m
# curation run; the earlier value of 24M caused a consistent undershoot.
CC_CHARS_PER_SEGMENT: int = 17_000_000

# RAM budget (in GB) for the blend stage's in-memory shuffle fast path. When
# the estimated effective RAM (staging size × 5 for Python-object overhead)
# exceeds this budget, the curator falls back to a chunked disk shuffle.
# Env-overridable for instance-size variance.
SHUFFLE_RAM_BUDGET_GB: float = float(os.environ.get("SHUFFLE_RAM_BUDGET_GB", "12"))

# Pretraining val fraction — portion of the blended corpus routed to val.jsonl
# at the end of the blend stage. Deliberately small (0.5%) because pretraining
# val is only used for perplexity; more val tokens would cost training tokens.
# SFT and DPO val fractions are stage-specific and live in their own modules
# (0.02 chat SFT, 0.05 code SFT, 0.05 DPO) — do not conflate with this.
PRETRAIN_VAL_FRACTION: float = 0.005

# Per-source doc caps used when `--mini` is passed to the curator. Exercises
# every source at small scale to validate the pipeline end-to-end before
# committing to a full run. common_crawl's cap is in WARC segments (not docs)
# because that's the unit of CC streaming.
MINI_OVERRIDES: dict[str, int] = {
    "common_crawl":  2,         # WARC segments
    "fineweb":       10_000,
    "wikipedia":     5_000,
    "pg19":          50,
    "pes2o":         2_000,
    "open_web_math": 3_000,
    "stackexchange": 2_000,
    "codesearchnet": 5_000,
    "stack_smol":    2_000,
    "stack_v2":      3_000,
    "jupyter":       500,
    "conala":        500,
}


# ── 7. Helpers ─────────────────────────────────────────────────────────────────

def dataset_link(entry: dict) -> str:
    """
    Return a markdown link for a data-mix entry. Prefers the Hub id if present,
    falls back to the raw URL, falls back to the plain display name.
    """
    name = entry["display"]
    if "hub" in entry:
        return f"[{name}](https://huggingface.co/datasets/{entry['hub']})"
    if "url" in entry:
        return f"[{name}]({entry['url']})"
    return name


def total_tokens(size: str) -> int:
    """Return the integer token budget for a given model size."""
    return TARGET_CONFIGS[size]["total_tokens"]


def token_target_display(size: str) -> str:
    """Return the human-readable token budget (e.g. "5B") for a given size."""
    return TARGET_CONFIGS[size]["display_tokens"]


def epochs(size: str) -> int:
    """Return the training epoch count for a given model size."""
    return TARGET_CONFIGS[size]["epochs"]


def validate() -> None:
    """
    Runtime sanity check. Called at import time so typos fail fast.

    Verifies:
      - DATA_MIX percentages sum to 100
      - CODE_SUBMIX percentages sum to 100
      - OVERFLOW_SINK exists in DATA_MIX
      - All CODE_SUBMIX source names are distinct from DATA_MIX source names
      - Curator constants are positive numbers
      - Every TARGET_CONFIGS entry has the required keys
      - Every MINI_OVERRIDES key is a real source in ALL_SOURCES
    """
    top_total = sum(entry["pct"] for entry in DATA_MIX.values())
    assert abs(top_total - 100.0) < 1e-6, (
        f"DATA_MIX percentages sum to {top_total}, expected 100"
    )

    code_total = sum(entry["pct"] for entry in CODE_SUBMIX.values())
    assert abs(code_total - 100.0) < 1e-6, (
        f"CODE_SUBMIX percentages sum to {code_total}, expected 100"
    )

    assert OVERFLOW_SINK in DATA_MIX, (
        f"OVERFLOW_SINK={OVERFLOW_SINK!r} not present in DATA_MIX"
    )

    overlap = set(CODE_SUBMIX) & set(DATA_MIX)
    assert not overlap, (
        f"CODE_SUBMIX names collide with DATA_MIX top-level names: {overlap}"
    )

    assert CHARS_PER_TOKEN > 0,        f"CHARS_PER_TOKEN must be > 0, got {CHARS_PER_TOKEN}"
    assert CC_CHARS_PER_SEGMENT > 0,   f"CC_CHARS_PER_SEGMENT must be > 0, got {CC_CHARS_PER_SEGMENT}"
    assert SHUFFLE_RAM_BUDGET_GB > 0,  f"SHUFFLE_RAM_BUDGET_GB must be > 0, got {SHUFFLE_RAM_BUDGET_GB}"
    assert 0.0 < PRETRAIN_VAL_FRACTION < 1.0, (
        f"PRETRAIN_VAL_FRACTION must be in (0, 1), got {PRETRAIN_VAL_FRACTION}"
    )

    required_keys = {"total_tokens", "epochs", "cc_crawls", "display_tokens"}
    for size, cfg in TARGET_CONFIGS.items():
        missing = required_keys - set(cfg)
        assert not missing, (
            f"TARGET_CONFIGS[{size!r}] missing required keys: {missing}"
        )
        assert cfg["total_tokens"] > 0
        assert cfg["epochs"] >= 1
        assert len(cfg["cc_crawls"]) >= 1

    unknown_mini = set(MINI_OVERRIDES) - set(ALL_SOURCES)
    assert not unknown_mini, (
        f"MINI_OVERRIDES references sources not in ALL_SOURCES: {unknown_mini}"
    )


# Run the sanity check at import time — cheap, and catches typos the moment
# this module is loaded rather than at training time.
validate()