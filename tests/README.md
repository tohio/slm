# Tests

The suite is layered so expensive training is not repeated merely to test it.
Tests either exercise cheap synthetic inputs or validate artifacts produced by
the real pipeline run.

## Test policy

| Gate | Command | Cost | Run when |
|---|---|---:|---|
| CPU/unit | `make test-unit` | Minutes, no pipeline artifacts | Every change |
| GPU stack | `make test-gpu-gate` | Bounded synthetic compile/train/generate | Once per GPU image or dependency upgrade |
| Data artifacts | `make test-data-pipeline SIZE=mini` | Reads existing artifacts | After the mini data pipeline |
| Training artifacts | Stage-specific `make test-* SIZE=<size>` | Loads existing checkpoints | Immediately after that real stage |
| Model comparison preflight | `make compare-sft-preflight` | Two checkpoint loads, no dataset | Before any comparison training |
| SFT response comparison | `make compare-sft` | Two opt-in fine-tunes | Only after preflight passes |

Full pretraining, SFT, and DPO are not test-suite prerequisites. A completed
real run is the artifact under test; do not launch a second full run just to
validate it.

## Mandatory cheap gates

```bash
make test-unit
```

This covers architecture and cache behavior, native export conversion, data
contracts, config generation, exact dependency pins, and one-step synthetic
SFT/DPO integration.

On each new GPU image:

```bash
make test-gpu-gate
```

The GPU gate verifies the pinned CUDA runtime and driver, native SM support,
BF16, eager backward/optimizer behavior, `torch.compile`, and cached versus
uncached generation. It downloads no datasets and loads no trained checkpoint.

## Artifact tests

Artifact targets are strict: a missing requested checkpoint is a failure, not
a successful run containing only skipped tests.

```bash
make test-training SIZE=125m
make test-sft-instruct SIZE=125m
make test-sft-code SIZE=125m
make test-dpo-chat SIZE=125m
```

Use the matching target immediately after its actual pipeline stage. For a
cheap end-to-end rehearsal, create mini artifacts once and validate them:

```bash
make curate-mini
make validate SIZE=mini
make tokenizer SIZE=mini
make tokenize SIZE=mini
make pretrain-mini SIZE=mini GPUS=1
make sft-instruct-mini SIZE=mini GPUS=1
make sft-code-mini SIZE=mini GPUS=1
make dpo-chat-mini SIZE=mini GPUS=1
make test-artifacts SIZE=mini
```

Direct exploratory `pytest` calls may still skip absent stage artifacts.
Makefile artifact targets pass `--require-artifacts` and therefore fail.

## Controlled model comparison

The comparison has a cheap, fail-fast gate:

```bash
make export-base-local SIZE=125m
make compare-sft-preflight
```

The default Tohio input is the native export at
`results/exports/125m/base`; neither model loads remote code. Preflight checks
parameter/vocabulary integrity, prompt sensitivity, and cached/uncached greedy
parity before data is downloaded or training starts.

Only after preflight succeeds:

```bash
make compare-sft
```

Override its bounded defaults explicitly when needed:

```bash
make compare-sft \
  COMPARE_TRAIN_EXAMPLES=32 \
  COMPARE_EVAL_EXAMPLES=32 \
  COMPARE_MAX_STEPS=200
```

The harness uses the same pinned UltraChat record identities, split, order,
optimizer updates, and completion-only objective for both models. Token IDs
cannot be identical across different tokenizers, so it reports each model's
sequence and supervised-token totals instead of claiming equal token exposure.
Canonical response output is uncached; cache correctness is a separate
preflight invariant.

## Test locations

```text
tests/
├── data_pipeline/       # existing curation/validation/tokenizer artifacts
├── gpu_pipeline/        # existing pretrain/SFT/DPO artifacts
├── model/               # CPU architecture, masking, and cache invariants
├── test_export.py       # native Transformers export
├── test_trl_smoke.py    # one-step synthetic SFT and DPO
└── test_sft_comparison.py
```
