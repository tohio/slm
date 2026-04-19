"""
curator/constants.py
---------------------
Shared constants for the curator pipeline.

Centralized here so recalibration is a single-file change rather than a
repo-wide find-and-replace.

Constants:
    CHARS_PER_TOKEN: Empirical characters-per-token ratio for our trained
        BPE tokenizer. Used by sources for estimated_tokens stats and by
        the blend stage to size per-source character budgets.

        This is a rough average — true chars/token varies by domain
        (English prose ~4.5, code ~3.5, math/symbols ~3). Calibrate
        against tokenizer.encode output from an actual 125m run and
        update if the empirical average shifts by more than ~10%.

        Default 5 is a conservative estimate slightly above the prose
        average — biases us toward over-provisioning character budgets,
        which means we hit token targets rather than falling short.

    CC_CHARS_PER_SEGMENT: Empirical extracted-text output per Common Crawl
        WARC segment after trafilatura + language filtering. Used by
        curate.py to compute cc_segments at runtime from target token
        count rather than hardcoding per-scale values.

        Observed at ~17M chars/segment in the 125m run. Previously
        assumed 24M, which is why the 125m run produced 3.28B tokens vs
        the 5B target.
"""

CHARS_PER_TOKEN = 5

CC_CHARS_PER_SEGMENT = 17_000_000