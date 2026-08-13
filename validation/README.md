# Validation

## Purpose

`validation/` applies post-curation document checks to the blended pretraining
train and validation splits. It removes structurally broken prose and
excessive line repetition while preserving code, math, and other non-prose
sources for which English-prose heuristics are inappropriate. KenLM measures
eligible prose by default and filters only when an explicit threshold is set.

It does not load sources, deduplicate documents, train the tokenizer, or
tokenize the corpus.

## Contents

```text
validation/
└── scripts/
    └── validate.py   split validation, reporting, and completion manifest
```

## How It Fits In

Input:

```text
$DATA_DIR/runs/<size>/curated/train.jsonl
$DATA_DIR/runs/<size>/curated/val.jsonl
```

Output:

```text
$DATA_DIR/runs/<size>/validated/train.jsonl
$DATA_DIR/runs/<size>/validated/val.jsonl
$DATA_DIR/runs/<size>/validated/validation_stats.json
$DATA_DIR/runs/<size>/validated/_SUCCESS.json
```

Tokenizer training consumes the validated training split. Binary tokenization
consumes both validated splits.

## Validation Rules

| Rule | Applies to | Rejection condition |
|---|---|---|
| Terminal punctuation | prose-like sources | no non-empty line ends in `.`, `!`, `?`, `'`, or `"` |
| Repeated-line ratio | every source | duplicate non-empty lines exceed 30% when the record has at least four lines |
| KenLM perplexity | prose-like sources | report-only unless an explicit threshold is exceeded |

Code, configured synthetic sources, Nemotron math, and Nemotron specialized
records bypass terminal-punctuation and KenLM checks. They still receive the
repeated-line check.

When no explicit perplexity threshold is supplied, the validator:

1. scores eligible prose without changing corpus membership;
2. records counts, mean, minimum, maximum, and bounded deterministic
   per-source percentile samples for both splits;
3. writes the distributions under `perplexity_audit` in
   `validation_stats.json`.

The validator does not derive a rejection threshold from the corpus. A
self-derived percentile guarantees attrition without establishing that the
removed tail is harmful, and a Wikipedia-trained KenLM can penalize valuable
technical or specialized prose. Use `--perplexity-threshold` only after an
explicit threshold has been calibrated and approved for this corpus.

`--no-perplexity` records an explicit no-KenLM run. It is not an automatic
fallback for a missing model.

## Prerequisites

Install the KenLM bindings and download the English model:

```bash
make install-kenlm
make download-kenlm-model DATA_DIR=/data/slm/data
```

The default model path is:

```text
$DATA_DIR/models/en.arpa.bin
```

## Usage

Validate a size-scoped curated corpus:

```bash
make validate SIZE=125m
```

Use an explicitly approved fixed threshold:

```bash
python validation/scripts/validate.py \
  --size 125m \
  --perplexity-threshold 800
```

Disable perplexity filtering explicitly:

```bash
python validation/scripts/validate.py \
  --size 125m \
  --no-perplexity
```

Override every path:

```bash
python validation/scripts/validate.py \
  --size 125m \
  --train /data/slm/data/runs/125m/curated/train.jsonl \
  --val /data/slm/data/runs/125m/curated/val.jsonl \
  --train-output /data/slm/data/runs/125m/validated/train.jsonl \
  --val-output /data/slm/data/runs/125m/validated/val.jsonl
```

The validator reuses an existing output only when its completion manifest
matches the input files, implementation, KenLM selection, threshold policy,
report sample size, and current outputs.

## Artifact Transfer

Upload validated data and metadata:

```bash
make validate-upload SIZE=125m
```

Equivalent explicit stage selection:

```bash
make artifacts-upload \
  SIZE=125m \
  ARTIFACT_STAGES="validated,metadata"
```

## Tests

Validate mini artifacts:

```bash
make test-validate SIZE=mini
```

Validate a completed full-size artifact:

```bash
make test-validate SIZE=125m
```

The artifact test requires the expected files and fails rather than skipping
when they are absent.

## Gotchas

- Omitting `--perplexity-threshold` is report-only; it does not remove the
  highest-scoring percentile.
- Changing the KenLM model or threshold changes the stage contract.
- `validation_stats.json` contains the KenLM policy, per-source distributions,
  measured rejection counts, and any explicit threshold; inspect it before
  tokenizer training.
