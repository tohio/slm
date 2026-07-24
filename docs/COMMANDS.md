# SLM command reference

This file documents the Makefile command surface. The Makefile is the source of truth.

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

```bash
make setup
make setup-data-dir DATA_DIR=/data/slm/data
make install
make install-gpu
make install-uv
make install-conda
```

Validation prerequisites:

```bash
make install-kenlm
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model DATA_DIR=/data/slm/data
```

GPU restore:

```bash
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m RUN_ID=125m-20260629-a8f3c9
```

`setup-gpu` requires `RUN_ID` when restoring tokenized artifacts from S3.

---

## Curation

```bash
make curate-mini
make curate SIZE=125m WORKERS=62
```

Stage-specific curation:

```bash
make curate-download SIZE=125m
make curate-filter SIZE=125m WORKERS=62
make curate-dedup SIZE=125m WORKERS=62
make curate-blend SIZE=125m
```

To replace a legacy or stale raw source only after a clean staged download
succeeds:

```bash
make curate-download SIZE=125m FORCE=1
```

Upload only curated artifacts through the RUN_ID flow:

```bash
make curate-upload SIZE=125m
```

---

## Validation

```bash
make validate SIZE=125m
```

Upload only validated artifacts through the RUN_ID flow:

```bash
make validate-upload SIZE=125m
```

---

## Tokenizer and tokenization

```bash
make tokenizer SIZE=125m
make tokenizer-test SIZE=125m
make tokenize SIZE=125m
```

---

## Artifacts

Artifacts are grouped by `RUN_ID`.

Upload:

```bash
make artifacts-upload SIZE=125m
make artifacts-upload SIZE=125m RUN_ID=125m-20260629-a8f3c9
make artifacts-upload SIZE=125m ARTIFACT_STAGES="tokenized,tokenizer,metadata"
```

Download:

```bash
make artifacts-download SIZE=125m RUN_ID=125m-20260629-a8f3c9
make artifacts-download SIZE=125m RUN_ID=125m-20260629-a8f3c9 ARTIFACT_STAGES="tokenized,tokenizer,metadata"
```

Valid stages:

```text
raw, curated, validated, tokenized, tokenizer, metadata
```

---

## Config generation

Stage configs:

```bash
make config-gen-pretrain SIZE=125m GPUS=1
make config-gen-sft SIZE=125m GPUS=1
make config-gen-dpo SIZE=125m GPUS=1
make config-gen SIZE=125m GPUS=1
```

Hardware override:

```bash
make config-gen SIZE=125m GPUS=4 GPU=h200
make config-gen SIZE=1b GPUS=8 GPU=b200 MODE=aggressive
```

Accelerate configs:

```bash
make accelerate-config-single
make accelerate-config-multi GPUS=4
make accel-gen-ddp GPUS=8
make accel-gen-fsdp GPUS=8
```

Use the same `GPUS` value for Accelerate setup, config generation, and training.

---

## Pretraining

```bash
make pretrain-mini SIZE=mini GPUS=1
make pretrain SIZE=125m GPUS=1
make pretrain-resume SIZE=125m GPUS=1
make pretrain-smoke SIZE=125m
make smoke-gen SIZE=125m
```

Output:

```text
results/runs/<size>/pretrain/final
```

Before SFT:

```bash
make reinit-embeds SIZE=125m
```

---

## SFT

Prepare data:

```bash
make prepare-sft SIZE=125m
```

Sources and immutable Hub revisions are configured in
`finetune/configs/sft_data_sources.yaml`. Prepared splits include a provenance
and integrity manifest; changing the source contract requires an intentional
rerun with `finetune/data/prepare_sft.py --force`.

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

Raw code-completion path:

```bash
make prepare-code-completion SIZE=125m
make sft-code-completion SIZE=125m
make eval-code-completion SIZE=125m
```

Outputs:

```text
results/runs/<size>/sft_instruct/final
results/runs/<size>/sft_code/final
results/runs/<size>/sft_code_completion/final
```

---

## DPO

Prepare data:

```bash
make prepare-dpo SIZE=125m
```

The pinned source and preference-quality contract are configured in
`alignment/configs/dpo_data_sources.yaml`. Prepared data includes a manifest;
changing the contract requires an intentional
`alignment/data/prepare_dpo.py --force` run.

Train:

```bash
make dpo-chat SIZE=125m GPUS=1
make dpo-chat-resume SIZE=125m GPUS=1
make dpo-chat-mini SIZE=mini GPUS=1
```

Compatibility aliases:

```bash
make dpo SIZE=125m GPUS=1
make dpo-resume SIZE=125m GPUS=1
make dpo-mini SIZE=mini GPUS=1
```

Output:

```text
results/runs/<size>/dpo_chat/final
```

---

## Evaluation

Benchmark evaluation is optional.

```bash
make eval-base SIZE=125m
make eval-instruct SIZE=125m
make eval-chat SIZE=125m
make eval-code SIZE=125m
make eval SIZE=125m
make eval-mini SIZE=mini
```

Sanity evaluation:

```bash
make eval-sanity-base SIZE=125m
make eval-sanity-instruct SIZE=125m
make eval-sanity-chat SIZE=125m
make eval-sanity-code SIZE=125m
make eval-sanity SIZE=125m
```

`eval` and `eval-sanity` default to the chat variant.

---

## Export

```bash
make export-base SIZE=125m
make export-instruct SIZE=125m
make export-chat SIZE=125m
make export-code SIZE=125m
make export SIZE=125m
```

Hub names:

```text
tohio/slm-<size>
tohio/slm-<size>-instruct
tohio/slm-<size>-chat
tohio/slm-<size>-code
```

---

## Serving

```bash
make serve SIZE=125m
make serve-local SIZE=125m
```

---

## Tests

Data pipeline:

```bash
make test-curator
make test-validate
make test-tokenizer
make test-data-pipeline
```

GPU pipeline:

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

## Diagnostics

```bash
make sanity-train
make sanity-train-small
make sanity-train-tiny
make sanity-train-save SANITY_SIZE=tiny
```

---

## Cleanup

```bash
make clean-data SIZE=125m
make clean-results
make clean-logs
make clean
```
