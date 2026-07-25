# Tests

The test suite separates inexpensive contract checks from tests that inspect
artifacts produced by real data and training runs. Full pretraining, SFT, and
DPO are not launched merely to test them.

## Test layers

| Layer | Command | Requires | Purpose |
|---|---|---|---|
| CPU contracts | `make test-unit` | CPU environment | Architecture, configuration, data, export, and one-step trainer contracts |
| GPU acceptance | `make test-gpu-gate` | Supported NVIDIA host | CUDA stack, BF16, compile, optimizer, and generation checks |
| Data artifacts | `make test-data-pipeline SIZE=<size>` | Completed curator, validation, and tokenizer outputs | Inspect real stage manifests and artifacts |
| Training artifacts | Stage-specific `make test-* SIZE=<size>` | Existing final checkpoint | Validate the result of that actual run |
| Model comparison preflight | `make compare-sft-preflight` | SmolLM2 plus local 125M base export | Reject invalid comparison candidates without training |
| Controlled SFT comparison | `make compare-sft` | Preflight success and dataset access | Compare response to identical selected records and update counts |

## CPU contract gate

Run before submitting changes:

```bash
make test-unit
```

This aggregates:

- model shape, masking, cache, and generation invariants;
- native export conversion and parity contracts;
- curation, SFT, and DPO data contracts;
- training and dependency configuration;
- config and accelerator generation;
- one-step synthetic SFT/DPO integration;
- controlled-comparison logic; and
- miscellaneous repository consistency checks.

Run a focused group while developing:

```bash
make test-model
make test-export
make test-data-unit
make test-training-args
make test-config-gen
make test-accel-gen
make test-comparison
make test-misc
```

## GPU acceptance gate

Run once for each new GPU image or dependency/CUDA upgrade:

```bash
make test-gpu-gate
```

The gate verifies the pinned environment and driver, native compute capability,
BF16, eager backward and optimizer behavior, `torch.compile`, and
cached/uncached generation. It uses synthetic inputs, downloads no dataset,
and loads no trained checkpoint.

## Data artifact tests

After completing the mini data pipeline:

```bash
make test-curator SIZE=mini
make test-validate SIZE=mini
make test-tokenizer SIZE=mini
```

Or run the aggregate:

```bash
make test-data-pipeline SIZE=mini
```

These tests inspect existing run-scoped outputs and manifests. They do not
rebuild the corpus.

## Training artifact tests

Run the matching test immediately after each paid stage:

```bash
make test-training SIZE=125m
make test-sft-instruct SIZE=125m
make test-sft-code SIZE=125m
make test-dpo-chat SIZE=125m
```

Make targets pass the strict artifact requirement, so a missing requested
checkpoint fails. Direct exploratory `pytest` calls may skip tests whose
artifacts are absent.

A bounded end-to-end artifact rehearsal uses the mini profile:

```bash
make curate-mini
make validate SIZE=mini
make tokenizer SIZE=mini
make tokenizer-test SIZE=mini
make tokenize SIZE=mini
make restore-size-tokenizer SIZE=mini
make pretrain-mini SIZE=mini GPUS=1
make prepare-sft SIZE=mini
make sft-instruct-mini SIZE=mini GPUS=1
make sft-code-mini SIZE=mini GPUS=1
make prepare-dpo SIZE=mini
make dpo-chat-mini SIZE=mini GPUS=1
make test-artifacts SIZE=mini
```

This is a pipeline rehearsal, not a quality benchmark.

## Controlled model comparison

Build the local 125M base export and run the fail-fast checks:

```bash
make export-base-local SIZE=125m
make compare-sft-preflight
```

Preflight validates architecture and vocabulary integrity, parameter count,
prompt sensitivity, and cached/uncached greedy parity before dataset selection
or fine-tuning.

Only after it succeeds:

```bash
make compare-sft
```

Override the bounded defaults explicitly:

```bash
make compare-sft \
  COMPARE_TRAIN_EXAMPLES=32 \
  COMPARE_EVAL_EXAMPLES=32 \
  COMPARE_MAX_STEPS=200
```

The harness selects the same pinned UltraChat record identities and ordering
for both models and applies the same optimizer-update schedule and
completion-only objective. Each tokenizer necessarily produces different
token sequences, so the report includes sequence and supervised-token totals
instead of claiming identical token exposure.

## Layout

```text
tests/
├── data_pipeline/       # Real curator, validation, and tokenizer artifacts
├── gpu_pipeline/        # Real pretraining, SFT, and DPO artifacts
├── model/               # Architecture, masking, and KV-cache invariants
├── test_export.py       # Native Transformers conversion
├── test_trl_smoke.py    # One-step synthetic SFT and DPO
└── test_sft_comparison.py
```

Keep new tests at the cheapest layer that can prove the contract. Expensive
quality measurements belong in `eval/`, while tests should establish
correctness and artifact integrity.
