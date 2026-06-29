# SLM Pipeline — Command Reference

This file documents the Makefile-facing command surface. The Makefile is the command source of truth.

Default variables:

```bash
SIZE=125m
GPUS=1
DATA_DIR=data
RESULTS_DIR=results
RUN_ID=
ARTIFACT_STAGES=raw,curated,validated,tokenized,tokenizer,metadata
```

Common overrides:

```bash
make <target> SIZE=350m
make <target> GPUS=4
make <target> DATA_DIR=/data/slm/data
make <target> RUN_ID=125m-20260629-a8f3c9
```

---

## Setup

### Local / CPU setup

```bash
make setup
make setup-data-dir DATA_DIR=/data/slm/data
make install
make install-kenlm
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model    DATA_DIR=/data/slm/data
```

### Alternative environments

```bash
make install-uv
make install-conda
```

### GPU setup

GPU setup restores tokenized artifacts by `RUN_ID`.

```bash
make install-gpu
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m RUN_ID=125m-20260629-a8f3c9
source ~/.bashrc
```

`RUN_ID` is required for `setup-gpu`.

---

## Stage 1 — Curation

```bash
make curate-mini
make curate SIZE=125m WORKERS=62
```

Stage-specific curation targets:

```bash
make curate-download SIZE=125m
make curate-filter   SIZE=125m WORKERS=62
make curate-dedup    SIZE=125m WORKERS=62
make curate-blend    SIZE=125m
```

Legacy direct upload target:

```bash
make curate-upload SIZE=125m
```

The preferred reusable artifact interface is `artifacts-upload`.

---

## Stage 2 — Validation

```bash
make validate SIZE=125m
make validate-datatrove SIZE=125m
```

`validate-upload` is retained for legacy direct upload workflows. Prefer `artifacts-upload` for reusable run artifacts.

---

## Stage 3 — Tokenizer and Tokenization

Train the tokenizer:

```bash
make tokenizer SIZE=125m
make tokenizer-test SIZE=125m
```

Tokenize validated train/val splits:

```bash
make tokenize SIZE=125m
```

---

## Reusable Artifacts

Artifacts are grouped by `RUN_ID`, not date.

Upload artifacts:

```bash
make artifacts-upload SIZE=125m
make artifacts-upload SIZE=125m RUN_ID=125m-20260629-a8f3c9
make artifacts-upload SIZE=125m ARTIFACT_STAGES="tokenized,tokenizer,metadata"
```

Download artifacts:

```bash
make artifacts-download SIZE=125m RUN_ID=125m-20260629-a8f3c9
make artifacts-download SIZE=125m RUN_ID=125m-20260629-a8f3c9 ARTIFACT_STAGES="tokenized,tokenizer,metadata"
```

Valid stages:

```text
raw, curated, validated, tokenized, tokenizer, metadata
```

---

## Config Generation

Generate stage configs for the current GPU and GPU count:

```bash
make config-gen-pretrain SIZE=125m GPUS=1
make config-gen-sft      SIZE=125m GPUS=1
make config-gen-dpo      SIZE=125m GPUS=1
make config-gen          SIZE=125m GPUS=1
```

Override GPU detection:

```bash
make config-gen SIZE=125m GPUS=4 GPU=h200
make config-gen SIZE=1b GPUS=8 GPU=b200 MODE=aggressive
```

Generate Accelerate configs:

```bash
make accel-gen-ddp  GPUS=8
make accel-gen-fsdp GPUS=8
make accelerate-config-single
make accelerate-config-multi GPUS=4
```

Use the same `GPUS` value for Accelerate setup, config generation, and training.

---

## Stage 4 — Pretraining

```bash
make pretrain-mini SIZE=mini GPUS=1
make pretrain      SIZE=125m GPUS=1
make pretrain-resume SIZE=125m GPUS=1
make smoke-gen SIZE=125m
```

Checkpoint output:

```text
results/runs/<size>/pretrain/final
```

Before post-training:

```bash
make reinit-embeds SIZE=125m
```

---

## Stage 5 — SFT

Prepare SFT data:

```bash
make prepare-sft SIZE=125m
```

Instruct SFT:

```bash
make sft-instruct SIZE=125m GPUS=1
make sft-instruct-resume SIZE=125m GPUS=1
make sft-instruct-mini SIZE=mini GPUS=1
```

Compatibility aliases:

```bash
make sft SIZE=125m GPUS=1
make sft-mini SIZE=mini GPUS=1
```

Code SFT:

```bash
make sft-code SIZE=125m GPUS=1
make sft-code-resume SIZE=125m GPUS=1
make sft-code-mini SIZE=mini GPUS=1
```

Raw code-completion SFT:

```bash
make prepare-code-completion SIZE=125m
make sft-code-completion SIZE=125m
make eval-code-completion SIZE=125m
```

Checkpoint outputs:

```text
results/runs/<size>/sft_instruct/final
results/runs/<size>/sft_code/final
results/runs/<size>/sft_code_completion/final
```

---

## Stage 6 — DPO

Prepare DPO data:

```bash
make prepare-dpo SIZE=125m
```

Chat DPO:

```bash
make dpo-chat SIZE=125m GPUS=1
make dpo-chat-resume SIZE=125m GPUS=1
make dpo-chat-mini SIZE=mini GPUS=1
```

Compatibility aliases:

```bash
make dpo SIZE=125m GPUS=1
make dpo-mini SIZE=mini GPUS=1
make dpo-resume SIZE=125m GPUS=1
```

Checkpoint output:

```text
results/runs/<size>/dpo_chat/final
```

---

## Stage 7 — Evaluation

```bash
make eval-base     SIZE=125m
make eval-instruct SIZE=125m
make eval-chat     SIZE=125m
make eval-code     SIZE=125m
make eval          SIZE=125m
make eval-mini     SIZE=mini
```

Sanity evaluation:

```bash
make eval-sanity-base     SIZE=125m
make eval-sanity-instruct SIZE=125m
make eval-sanity-chat     SIZE=125m
make eval-sanity-code     SIZE=125m
make eval-sanity          SIZE=125m
```

`eval` and `eval-sanity` default to the final chat-aligned variant.

---

## Stage 8 — Export

```bash
make export-base     SIZE=125m
make export-instruct SIZE=125m
make export-chat     SIZE=125m
make export-code     SIZE=125m
make export          SIZE=125m
```

Hub names:

```text
tohio/slm-<size>
tohio/slm-<size>-instruct
tohio/slm-<size>-chat
tohio/slm-<size>-code
```

---

## Stage 10 — Serving

```bash
make serve SIZE=125m
make serve-local SIZE=125m
```

`serve` uses the exported Hub chat model. `serve-local` uses the local chat checkpoint.

---

## Tests

CPU/data tests:

```bash
make test-curator
make test-validate
make test-tokenizer
make test-data-pipeline
```

GPU tests:

```bash
make test-training
make test-sft-instruct
make test-sft-code
make test-dpo-chat
make test-gpu-pipeline
```

Compatibility aliases:

```bash
make test-sft-chat
make test-dpo
```

Unit tests:

```bash
make test-model
make test-config-gen
make test-accel-gen
make test-unit
```

---

## Cleanup

```bash
make clean
make clean-data SIZE=125m
make clean-results
make clean-logs
```
