# Tests

## Purpose

`tests/` owns repository contracts and checks against artifacts produced by
the data and training pipelines. It does not launch full curation,
pretraining, SFT, or DPO runs.

The project-wide execution policy and commands live in
[`docs/TESTING.md`](../docs/TESTING.md).

## Contents

```text
tests/
├── data_pipeline/       real curator, validation, and tokenizer artifacts
├── gpu_pipeline/        real pretraining, SFT, and DPO artifacts
├── model/               architecture, masking, and KV-cache invariants
├── conftest.py          shared artifact-selection options
├── test_export.py       native Transformers conversion
├── test_trl_smoke.py    one-step synthetic SFT and DPO contracts
└── test_sft_comparison.py
```

## How It Fits In

The Makefile exposes focused and aggregate gates. Tests under
`data_pipeline/` and `gpu_pipeline/` inspect existing run artifacts; the
remaining tests exercise code contracts with bounded fixtures.

Pytest options defined in `conftest.py`:

- `--size=<mini|125m|350m|1b>` selects the artifact profile;
- `--require-artifacts` turns a missing requested artifact into a failure.

Use the Make targets documented in [`docs/TESTING.md`](../docs/TESTING.md) for
normal execution so the correct options and environment variables are applied.
Model, export, TRL, and training-argument tests require the pinned training
stack (`make install-training` on CPU or `make install-gpu` on a GPU host), not
the separate Transformers 4.57.6 curation environment.
