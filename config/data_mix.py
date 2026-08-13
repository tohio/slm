"""
config/data_mix.py
-------------------
Single source of truth for the SLM pretraining data mix, token budgets,
and the locked curator-side constants.

Referenced by:
    - curator/scripts/curate.py   (data preparation)
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

Token vocabulary:
    corpus_tokens     curation-side corpus target for a given model size.
                      The curator attempts to build this token budget before
                      final validation/filtering/tokenization losses. It is
                      the public planning figure shown on model cards and in
                      README token-target tables, but it is not a guarantee
                      that the final retained/tokenized corpus will contain
                      exactly this many usable tokens.

                      Example: a curation target can produce fewer retained
                      tokens after validation and exact tokenizer measurement.

                      Future targets include a retention buffer so final
                      retained/tokenized artifacts land closer to the intended
                      usable-token goal.

    retained_tokens   final usable tokenized tokens after validation,
                      filtering, blending, train/val split, and tokenization.
                      This is measured after the pipeline runs. It is not
                      stored in TARGET_CONFIGS because it depends on source
                      availability, filtering loss, tokenizer behavior, and
                      validation outcomes.

    consumed_tokens   corpus_tokens × epochs. The planning quantity used by
                      config_gen/config_gen.py to propose max_steps before a
                      corpus exists. Production training replaces that plan
                      with tokenizer-measured train tokens × epochs. Computed
                      by the consumed_tokens() helper below; not a stored field.
                      Do not confuse this with retained_tokens.

Section layout:
    1. DATA_MIX                 top-level source percentages + metadata
    2. CODE_SUBMIX              sub-mix of the 15% code share
    3. Supplemental caps        fixed/supply-limited source controls
    4. OVERFLOW_SINK            which source absorbs supply deficits
    5. Source names + routing   NON_CODE_SOURCES, CODE_SOURCES, ALL_SOURCES,
                                SYNTHETIC_SOURCES, FILTER_SOURCE_FAMILIES,
                                PROSE_HEURISTIC_SKIP_SOURCES, DEDUP_PRIORITY
    6. TARGET_CONFIGS           per-size corpus + epochs + CC crawls
    7. Curator constants        CHARS_PER_TOKEN, CC_CHARS_PER_SEGMENT,
                                SHUFFLE_RAM_BUDGET_GB, PRETRAIN_VAL_FRACTION,
                                MINI_OVERRIDES
    8. Helpers                  dataset_link, corpus_tokens, consumed_tokens,
                                epochs, validate
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
        "pct":     5.0,
        "display": "Common Crawl",
        "url":     "https://commoncrawl.org",
    },
    "fineweb": {
        "pct":     10.0,
        "display": "FineWeb",
        "hub":     "HuggingFaceFW/fineweb",
    },
    "fineweb_edu": {
        "pct":     31.50,
        "display": "FineWeb-Edu",
        "hub":     "HuggingFaceFW/fineweb-edu",
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
        "display": "Common Pile peS2o (filtered academic papers)",
        "hub":     "common-pile/peS2o_filtered",
    },
    "nemotron_cc_math": {
        "pct":     7.0,
        "display": "Nemotron CC Math",
        "hub":     "nvidia/Nemotron-CC-Math-v1",
        "config":  "4plus",
    },
    "stackexchange": {
        "pct":     1.00,
        "display": "StackExchange",
        "hub":     "HuggingFaceH4/stack-exchange-preferences",
    },
    "synthetic_arithmetic": {
        "pct":     0.1475,
        "display": "Synthetic arithmetic",
        "hub":     "tohio/slm-synthetic-arithmetic",
    },
    "synthetic_task_code": {
        "pct":     0.3934,
        "display": "Synthetic task code",
        "hub":     "tohio/slm-synthetic-task-code",
    },
    "educational_qa_mcq_math": {
        "pct":     0.1475,
        "display": "Educational QA/MCQ (math)",
        "hub":     "tohio/slm-synthetic-educational-qa-mcq-math",
    },
    "educational_qa_mcq_general": {
        "pct":     0.2459,
        "display": "Educational QA/MCQ (general)",
        "hub":     "tohio/slm-synthetic-educational-qa-mcq-general",
    },
    "factual_restraint": {
        "pct":     0.0657,
        "display": "Factual restraint",
        "hub":     "tohio/slm-synthetic-factual-restraint",
    },
    "nemotron_specialized": {
        "pct":     12.0,
        "display": "Nemotron Specialized",
        "hub":     "nvidia/Nemotron-Pretraining-Specialized-v1.1",
    },
    "code": {
        "pct":     15.0,
        "display": "Code (multi-source)",
        # Dispatched across CODE_SUBMIX. The "code" entry itself is a logical
        # bucket — the actual per-source char targets are computed using
        # CODE_SUBMIX percentages × the 15% code share.
    },
}


# ── 2. Code sub-mix ────────────────────────────────────────────────────────────
#
# Percentages of the 15% code share (not of total tokens). The Stack v1 is the
# primary scalable code source; the smaller sources add language, notebook,
# and natural-language-to-code diversity.

CODE_SUBMIX: dict[str, dict] = {
    "stack_v1": {
        "pct": 83.0,
        "display": 'The Stack v1 dedup',
        "hub": 'bigcode/the-stack-dedup',
    },
    "codesearchnet": {
        "pct": 15.0,
        "display": 'CodeSearchNet',
        "hub": 'code-search-net/code_search_net',
    },
    "stack_smol": {
        "pct": 1.0,
        "display": 'The Stack (smol)',
        "hub": 'bigcode/the-stack-smol',
    },
    "jupyter": {
        "pct": 0.5,
        "display": 'Jupyter notebooks',
        "hub": 'bigcode/jupyter-parsed',
    },
    "conala": {
        "pct": 0.5,
        "display": 'CoNaLa',
        "hub": 'neulab/conala',
    },
}


# ── 3. Supplemental/fixed-supply caps ─────────────────────────────────────────
#
# Some sources are high-signal supplements rather than scalable corpus pillars.
# Their nominal percentage targets are capped by target size so they remain
# useful without silently forcing large overflow when supply or uniqueness runs
# out. The curator applies min(percentage_target_chars, cap).

SUPPLEMENTAL_CHAR_CAPS: dict[str, dict[str, int]] = {
    # Caps are character budgets. At CHARS_PER_TOKEN=4.3, the combined
    # synthetic caps are approximately:
    #   125m: 100M tokens
    #   350m: 200M tokens
    #   1b:   300M tokens
    # The split follows the validated 300M-token HF inventory profile.
    "synthetic_task_code": {
        "125m": 169_000_000,
        "350m": 338_000_000,
        "1b":   508_000_000,
    },
    "educational_qa_mcq_general": {
        "125m": 106_000_000,
        "350m": 211_000_000,
        "1b":   317_000_000,
    },
    "synthetic_arithmetic": {
        "125m":  63_000_000,
        "350m": 127_000_000,
        "1b":   190_000_000,
    },
    "educational_qa_mcq_math": {
        "125m":  63_000_000,
        "350m": 127_000_000,
        "1b":   190_000_000,
    },
    "factual_restraint": {
        "125m":  28_000_000,
        "350m":  56_000_000,
        "1b":    85_000_000,
    },
}


# Source-specific average record sizes for HF-backed synthetic datasets.
#
# Synthetic records are generated and validated in the separate
# tohio/slm-synthetic-data repo, then consumed here from Hugging Face. These
# estimates convert character budgets into max_docs caps for streaming reads.
# max_chars remains the primary stop condition, so the values only need to be
# conservative enough to avoid under-reading the published datasets.
SYNTHETIC_AVG_CHARS_PER_DOC: dict[str, int] = {
    "synthetic_arithmetic": 65,
    "synthetic_task_code": 190,
    "educational_qa_mcq_math": 215,
    "educational_qa_mcq_general": 215,
    "factual_restraint": 105,
}


# Extra document request headroom for source-level validation, quality
# filtering, and exact dedup. max_chars remains the primary stop condition,
# so this does not force extra retained data beyond the target.
SYNTHETIC_DOC_INFLATION: float = 1.5


# ── 4. Overflow sink ───────────────────────────────────────────────────────────
#
# When supply-constrained sources (Wikipedia, pg19, etc.) fall short of their
# character budget, the deficit is routed to this source. FineWeb has ~15T
# tokens available, so it can always close the gap.

OVERFLOW_SINK: str = "fineweb"


# ── 5. Source name lists ───────────────────────────────────────────────────────
#
# Derived lists for iteration. The "code" key in DATA_MIX is a logical bucket;
# the concrete source names used by the curator come from CODE_SUBMIX.

NON_CODE_SOURCES: list[str] = [name for name in DATA_MIX if name != "code"]
CODE_SOURCES: list[str] = list(CODE_SUBMIX.keys())
ALL_SOURCES: list[str] = NON_CODE_SOURCES + CODE_SOURCES
SYNTHETIC_SOURCES: frozenset[str] = frozenset({
    "synthetic_arithmetic",
    "synthetic_task_code",
    "educational_qa_mcq_math",
    "educational_qa_mcq_general",
    "factual_restraint",
})

# Exhaustive source-family contract for quality-filter routing. These families
# describe corpus type; they do not make all families share web-data filters.
# Every concrete source must appear exactly once so new sources cannot silently
# inherit the generic prose path.
FILTER_SOURCE_FAMILIES: dict[str, frozenset[str]] = {
    "raw_web": frozenset({"common_crawl"}),
    "curated_web": frozenset({"fineweb", "fineweb_edu"}),
    "reference_prose": frozenset({
        "wikipedia",
        "pg19",
        "pes2o",
        "stackexchange",
    }),
    "specialized": frozenset({"nemotron_cc_math", "nemotron_specialized"}),
    "code": frozenset(CODE_SOURCES),
    "synthetic": SYNTHETIC_SOURCES,
}


def source_filter_family(source: str) -> str:
    """Return the one configured filter family for a concrete source."""
    matches = [
        family
        for family, family_sources in FILTER_SOURCE_FAMILIES.items()
        if source in family_sources
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Source {source!r} must belong to exactly one filter family; "
            f"matches={matches}"
        )
    return matches[0]


PROSE_HEURISTIC_SKIP_SOURCES: frozenset[str] = frozenset(
    set(CODE_SOURCES)
    | set(SYNTHETIC_SOURCES)
    | {"nemotron_cc_math", "nemotron_specialized"}
)

# Exact cross-source duplicates are retained from the first source processed.
# Keep curated/reference/specialized corpora ahead of broad web crawls so the
# preferred copy wins deterministically.
DEDUP_PRIORITY: list[str] = [
    "wikipedia",
    "pg19",
    "pes2o",
    "stackexchange",
    "codesearchnet",
    "stack_smol",
    "jupyter",
    "conala",
    "synthetic_arithmetic",
    "synthetic_task_code",
    "educational_qa_mcq_math",
    "educational_qa_mcq_general",
    "factual_restraint",
    "nemotron_cc_math",
    "nemotron_specialized",
    "stack_v1",
    "fineweb_edu",
    "fineweb",
    "common_crawl",
]


# ── 6. Target configurations ───────────────────────────────────────────────────
#
# Per-size training targets. Carries everything a size-specific run needs:
#   corpus_tokens    — curation-side corpus target for this model size
#                      (PUBLIC planning figure). The curator attempts to build
#                      this many tokens before final validation/filtering and
#                      tokenization losses. The final retained/tokenized token
#                      count may be lower and should be measured from the
#                      generated train/val artifacts after the run.
#
#                      Multiplying corpus_tokens by `epochs` gives the planning
#                      consumed_tokens used by config_gen. Production max_steps
#                      is resolved from measured tokenized train tokens.
#
#   epochs           — number of training epochs over the corpus target.
#   cc_crawls        — Common Crawl snapshots to draw from at this scale.
#   display_corpus   — human-readable shorthand of corpus_tokens. Used by
#                      export.py when rendering model cards.
#
# cc_segments is computed at runtime from corpus_tokens × cc_share ×
# CHARS_PER_TOKEN ÷ CC_CHARS_PER_SEGMENT — see curator/scripts/curate.py.
#
# Important:
#   Finite/supply-constrained sources may not hit their nominal percentage at
#   larger scales after filtering. Deficits are routed to OVERFLOW_SINK, which
#   is currently FineWeb. Therefore DATA_MIX describes the intended target mix;
#   the realized post-filter mix should be audited after curation.

TARGET_CONFIGS: dict[str, dict] = {
    "mini": {
        "corpus_tokens":  1_000_000,
        "epochs":         1,
        "cc_crawls":      ["CC-MAIN-2024-10"],
        "display_corpus": "1M",
    },
    "125m": {
        "corpus_tokens":  10_000_000_000,
        "epochs":         2,
        "cc_crawls":      ["CC-MAIN-2024-10"],
        "display_corpus": "10B",
    },
    "350m": {
        "corpus_tokens":  25_000_000_000,
        "epochs":         2,
        "cc_crawls":      ["CC-MAIN-2024-10", "CC-MAIN-2023-50"],
        "display_corpus": "25B",
    },
    "1b": {
        "corpus_tokens":  75_000_000_000,
        "epochs":         1,
        "cc_crawls":      ["CC-MAIN-2024-10", "CC-MAIN-2023-50", "CC-MAIN-2023-40"],
        "display_corpus": "75B",
    },
}


# ── 7. Curator constants ───────────────────────────────────────────────────────
#
# These were previously scattered across curator/constants.py and curator/
# scripts/curate.py. Centralising means a retokenizer run (which could shift
# CHARS_PER_TOKEN) is a single-file change instead of chasing every reference.

# Average characters per BPE token from the trained tokenizer. Measured at
# 4.284 chars/token on the 32k-vocab tokenizer trained on the 125m
# pretraining corpus (10k docs sampled from data/runs/<size>/validated/train.jsonl,
# excluding code sources). Rounded to 4.3 for budget math.
#
# Previous value of 5 was a planning estimate that overshot retained tokens.
# The completed 125M run showed that validation/filtering/tokenization can
# reduce the final retained/tokenized corpus relative to the curation target.
# Consumers use this constant in arithmetic only — int → float is safe.
#
# If the tokenizer is retrained on a substantially different mix, rerun
# the chars-per-token measurement (see tokenizer/README.md) and update.
CHARS_PER_TOKEN: float = 4.3

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
    "fineweb_edu":   5_000,
    "wikipedia":     5_000,
    "pg19":          50,
    "pes2o":         2_000,
    "nemotron_cc_math": 3_000,
    "stackexchange": 2_000,
    "synthetic_arithmetic": 2_000,
    "synthetic_task_code":   2_000,
    "educational_qa_mcq_math":    2_000,
    "educational_qa_mcq_general": 2_000,
    "factual_restraint":     1_000,
    "nemotron_specialized":  2_000,
    "stack_v1":      3_000,
    # Keep every code sub-source bounded in mini validation runs. Without
    # these overrides, supply-bound production sources stream their complete
    # upstream datasets even though mini only needs enough records to exercise
    # the loader/filter/dedup/blend path.
    "codesearchnet": 10_000,
    "stack_smol":     1_000,
    "jupyter":          250,
    "conala":         2_000,
}


# ── 8. Helpers ─────────────────────────────────────────────────────────────────

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


def corpus_tokens(size: str) -> int:
    """
    Return the curation-side corpus target for a given model size.

    This is the public planning figure used by the curator and shown in
    model cards / README token-target tables. It is not a guarantee that
    the final retained/tokenized corpus will contain exactly this many
    usable tokens; filtering, validation, source availability, blending,
    and tokenization can reduce the final artifact size.

    Multiply by epochs(size) to get the consumed_tokens planning quantity used
    by config_gen before tokenized artifacts exist.
    """
    return TARGET_CONFIGS[size]["corpus_tokens"]


def consumed_tokens(size: str) -> int:
    """
    Return corpus_tokens × epochs for a given size.

    This is the number of tokens the optimiser sees across the whole
    pretraining run. Used by config_gen/config_gen.py to propose max_steps;
    production training resolves the final schedule from tokenized train data.

    Do NOT report this number on model cards or in public docs — it
    conflates corpus size with epoch count, which is the exact ambiguity
    that motivated the corpus_tokens / consumed_tokens vocabulary split.
    Public docs should always report corpus_tokens and epochs separately.
    """
    cfg = TARGET_CONFIGS[size]
    return cfg["corpus_tokens"] * cfg["epochs"]


def corpus_tokens_display(size: str) -> str:
    """Return the human-readable corpus size (e.g. "10B") for a given size."""
    return TARGET_CONFIGS[size]["display_corpus"]


def epochs(size: str) -> int:
    """Return the training epoch count for a given model size."""
    return TARGET_CONFIGS[size]["epochs"]


# ── Validation ────────────────────────────────────────────────────────────────

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
      - DEDUP_PRIORITY contains every concrete source exactly once
      - FILTER_SOURCE_FAMILIES contains every concrete source exactly once
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

    assert SYNTHETIC_DOC_INFLATION > 0, (
        f"SYNTHETIC_DOC_INFLATION must be > 0, got {SYNTHETIC_DOC_INFLATION}"
    )
    for source_name, avg_chars in SYNTHETIC_AVG_CHARS_PER_DOC.items():
        assert source_name in DATA_MIX, (
            f"SYNTHETIC_AVG_CHARS_PER_DOC references unknown source: {source_name}"
        )
        assert avg_chars > 0, (
            f"SYNTHETIC_AVG_CHARS_PER_DOC[{source_name!r}] must be > 0, got {avg_chars}"
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

    required_keys = {"corpus_tokens", "epochs", "cc_crawls", "display_corpus"}
    for size, cfg in TARGET_CONFIGS.items():
        missing = required_keys - set(cfg)
        assert not missing, (
            f"TARGET_CONFIGS[{size!r}] missing required keys: {missing}"
        )
        assert cfg["corpus_tokens"] > 0
        assert cfg["epochs"] >= 1
        assert len(cfg["cc_crawls"]) >= 1

    unknown_mini = set(MINI_OVERRIDES) - set(ALL_SOURCES)
    assert not unknown_mini, (
        f"MINI_OVERRIDES references sources not in ALL_SOURCES: {unknown_mini}"
    )

    assert len(DEDUP_PRIORITY) == len(set(DEDUP_PRIORITY)), (
        "DEDUP_PRIORITY contains duplicate source names"
    )
    assert set(DEDUP_PRIORITY) == set(ALL_SOURCES), (
        "DEDUP_PRIORITY must contain every ALL_SOURCES entry exactly once; "
        f"missing={sorted(set(ALL_SOURCES) - set(DEDUP_PRIORITY))}, "
        f"unknown={sorted(set(DEDUP_PRIORITY) - set(ALL_SOURCES))}"
    )
    assert SYNTHETIC_SOURCES <= set(ALL_SOURCES)
    assert PROSE_HEURISTIC_SKIP_SOURCES <= set(ALL_SOURCES)
    routed_sources = [
        source
        for family_sources in FILTER_SOURCE_FAMILIES.values()
        for source in family_sources
    ]
    assert len(routed_sources) == len(set(routed_sources)), (
        "FILTER_SOURCE_FAMILIES contains a source in multiple families"
    )
    assert set(routed_sources) == set(ALL_SOURCES), (
        "FILTER_SOURCE_FAMILIES must contain every ALL_SOURCES entry exactly "
        f"once; missing={sorted(set(ALL_SOURCES) - set(routed_sources))}, "
        f"unknown={sorted(set(routed_sources) - set(ALL_SOURCES))}"
    )


# Run the sanity check at import time — cheap, and catches typos the moment
# this module is loaded rather than at training time.
validate()
