# SLM

From-scratch training pipeline for a family of dense decoder-only language
models, covering corpus construction, tokenizer training, pretraining,
supervised fine-tuning, preference alignment, evaluation, export, inference,
and serving.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

SLM is a stage-based research pipeline for building and evaluating small
language models without starting from an existing foundation-model checkpoint.
It keeps data, tokenizer, model, training, and export contracts explicit so the
same workflow can be exercised with the `mini` profile and scaled to the 125M,
350M, and 1B configurations.

The repository produces four model branches:

| Variant | Lineage | Checkpoint |
|---|---|---|
| Base | pretrained from scratch | `$RESULTS_DIR/runs/<size>/pretrain/final` |
| Instruct | base → instruct SFT | `$RESULTS_DIR/runs/<size>/sft_instruct/final` |
| Chat | instruct → DPO | `$RESULTS_DIR/runs/<size>/dpo_chat/final` |
| Code | instruct → code SFT | `$RESULTS_DIR/runs/<size>/sft_code/final` |

## Architecture

![SLM pipeline and model architecture](docs/architecture.svg)

The data pipeline creates a validated corpus and a size-specific BPE tokenizer.
The training pipeline initializes the decoder from scratch, then branches from
the instruct checkpoint into chat-aligned and code-specialized models. Export
converts each completed branch into a native Transformers Llama package for
standard inference and vLLM serving.

| Size | Parameters | Layers | Hidden size | Q/KV heads | Context |
|---|---:|---:|---:|---:|---:|
| `mini` | 21.7M | 6 | 384 | 6 / 2 | 1,024 |
| `125m` | 125.3M | 16 | 768 | 12 / 4 | 2,048 |
| `350m` | 351.3M | 27 | 1,024 | 16 / 8 | 2,048 |
| `1b` | 1.012B | 21 | 2,048 | 32 / 8 | 4,096 |

All profiles use RoPE, RMSNorm, SwiGLU, grouped-query attention, pre-normalized
residual blocks, tied token embeddings, bias-free projections, and generation
KV caching.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for component ownership,
model lineage, and artifact flow.

## Features

- **Reproducible corpus construction:** source-specific loading, quality
  filtering, deterministic exact and fuzzy deduplication, controlled blending,
  completion manifests, and run-scoped artifact transfer.
- **One tokenizer contract across the pipeline:** a byte-level BPE tokenizer,
  explicit special tokens, a baked-in chat template, tokenizer fingerprints,
  and compatibility checks before training.
- **Hardware-aware training configuration:** generated pretraining, SFT, and
  DPO recipes that calculate micro-batches, gradient accumulation,
  checkpointing, warmup, and pretraining steps from the selected model, GPU,
  GPU count, and VRAM policy.
- **Separate post-training branches:** assistant-only-loss instruct and code
  SFT, followed by either DPO chat alignment or additional code
  specialization.
- **Preflight and artifact gates:** data manifests, split-leakage checks,
  checkpoint audits, controlled model comparison, CPU contract tests, and
  dataset-free CUDA acceptance tests.
- **Portable model packaging:** strict SLM-to-Llama weight conversion,
  tokenizer packaging, source/export parity checks, local artifacts, Hub
  publication, local generation, and vLLM deployment.

## Getting Started

### Prerequisites

- Git and GNU Make.
- Python 3.12.
- Linux for the supported instance setup scripts.
- Persistent storage sized for the selected corpus and checkpoints.
- A Hugging Face token for authenticated dataset and model access.
- Accepted terms for any gated datasets enabled by `config/data_mix.py`.
- AWS credentials and an S3 bucket when data preparation and GPU training use
  separate hosts.
- For GPU training: an NVIDIA GPU with BF16 support, CUDA 13.0 wheels, and an
  NVIDIA driver version supported by `infra/verify_environment.py`.

### Installation

Clone the repository and create the environment file:

```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
```

For local development:

```bash
make install
source .venv/bin/activate
```

For a fresh Ubuntu data-processing host, the setup target installs system and
Python dependencies, KenLM bindings, the spaCy model, and the run directories:

```bash
make setup-data-dir DATA_DIR=/data/slm/data
source .venv/bin/activate
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model DATA_DIR=/data/slm/data
```

For a fresh NVIDIA training host, first populate `.env` with S3 and Hugging
Face credentials, then restore a completed data run:

```bash
make setup-gpu \
  DATA_DIR=/data/slm/data \
  SIZE=125m \
  RUN_ID=125m-YYYYMMDD-abcdef

source .venv/bin/activate
make test-upgrade-gpu
```

### Configuration

The Makefile reads these paths from `.env` and permits command-line overrides:

```bash
DATA_DIR=/data/slm/data
RESULTS_DIR=/data/slm/results
EXPORTS_DIR=/data/slm/exports
```

Populate credentials only for the services used by the selected workflow:

```bash
HF_TOKEN=...
HF_USERNAME=tohio
S3_BUCKET=...
S3_PREFIX=slm/data
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
WANDB_API_KEY=...
WANDB_PROJECT=slm
```

The central pretraining data mix and token targets live in
`config/data_mix.py`. SFT and DPO dataset contracts live in
`finetune/configs/sft_data_sources.yaml` and
`alignment/configs/dpo_data_sources.yaml`.

### Usage

#### 1. Build data artifacts

Run the data stages on the data-processing host:

```bash
make curate SIZE=125m WORKERS=62
make validate SIZE=125m
make tokenizer SIZE=125m
make tokenizer-test SIZE=125m
make tokenize SIZE=125m
```

Validate the produced artifacts:

```bash
make test-curator SIZE=125m
make test-validate SIZE=125m
make test-tokenizer SIZE=125m
```

When training occurs on another host, upload the artifacts under the generated
run ID:

```bash
make artifacts-upload \
  SIZE=125m \
  ARTIFACT_STAGES="tokenized,tokenizer,metadata"
```

Use the mini profile to exercise the complete data path with bounded
per-source inputs:

```bash
make curate-mini
make validate SIZE=mini
make tokenizer SIZE=mini
make tokenizer-test SIZE=mini
make tokenize SIZE=mini
```

#### 2. Generate training configurations

Run configuration generation on the target GPU host so GPU detection reflects
the training hardware:

```bash
make config-gen SIZE=125m GPUS=1
```

Override detection or the VRAM policy when required:

```bash
make config-gen SIZE=350m GPUS=4 GPU=h200 MODE=conservative
```

`MODE` accepts `conservative`, `balanced`, or `aggressive`. Generated files are
written to `pretrain/configs/`, `finetune/configs/`, and
`alignment/configs/`.

#### 3. Train the base model

The GPU setup path restores the run-specific tokenizer and activates it for
training. When training from locally prepared artifacts instead, synchronize
the size-specific tokenizer first:

```bash
make restore-size-tokenizer SIZE=125m
```

Train or resume:

```bash
make pretrain SIZE=125m GPUS=1
make pretrain-resume SIZE=125m GPUS=1
```

`make pretrain` saves periodic checkpoints, writes
`$RESULTS_DIR/runs/125m/pretrain/final`, copies the tokenizer into the final
checkpoint, and runs a raw-generation smoke check.

#### 4. Train instruct, chat, and code branches

Initialize the chat-only special-token embeddings before the first SFT stage:

```bash
make reinit-embeds SIZE=125m
```

Prepare and train the instruct branch:

```bash
make prepare-sft SIZE=125m
make sft-instruct SIZE=125m GPUS=1
```

Both downstream branches start from the instruct checkpoint:

```bash
make sft-code SIZE=125m GPUS=1

make prepare-dpo SIZE=125m
make dpo-chat SIZE=125m GPUS=1
```

Use the corresponding `*-resume` target to resume from the latest checkpoint.
Run `--preflight-only` through the stage entry points when validating SFT or
DPO preprocessing without starting optimization; the component READMEs contain
the exact commands.

#### 5. Evaluate and export

Run benchmark evaluation for completed variants:

```bash
make eval-base SIZE=125m
make eval-instruct SIZE=125m
make eval-chat SIZE=125m
make eval-code SIZE=125m
```

Run deterministic chat-template behavior checks on post-trained variants:

```bash
make eval-sanity-instruct SIZE=125m
make eval-sanity-chat SIZE=125m
make eval-sanity-code SIZE=125m
```

Build and validate local native artifacts:

```bash
make export-base-local SIZE=125m
make export-instruct-local SIZE=125m
make export-chat-local SIZE=125m
make export-code-local SIZE=125m
```

With `HF_USERNAME` and `HF_TOKEN` configured, the non-local targets also
publish the validated artifacts:

```bash
make export-base SIZE=125m
make export-instruct SIZE=125m
make export-chat SIZE=125m
make export-code SIZE=125m
```

#### 6. Run inference or serve the chat model

Interactive inference from a local training checkpoint:

```bash
python inference/chat.py \
  --model results/runs/125m/dpo_chat/final
```

Serve a validated local export:

```bash
make serve-local SIZE=125m
```

See `inference/README.md` and `serve/README.md` for generation controls and
deployment prerequisites.

## Project Structure

```text
slm/
├── accelerate_configs/ Accelerate process-topology configurations
├── alignment/       DPO data preparation and training
├── config/          shared paths, runtime policy, data mix, and token targets
├── config_gen/      hardware-aware training and Accelerate config generation
├── curator/         source loading, filtering, deduplication, and blending
├── docs/            project guides and command reference
├── eval/            benchmark and deterministic behavior evaluation
├── export/          native model conversion, validation, and publication
├── finetune/        SFT data preparation and instruct/code training
├── inference/       interactive and batch generation
├── infra/           data-host and GPU-host setup and validation
├── model/           decoder-only Transformer implementation
├── notebooks/       exploratory artifact analysis
├── pretrain/        binary tokenization and base-model training
├── scripts/         diagnostics and maintenance utilities
├── serve/           vLLM launcher and Kubernetes manifests
├── tests/           CPU, GPU, artifact, and comparison checks
├── tokenizer/       BPE tokenizer training and validation
└── validation/      post-curation document validation
```

Each non-trivial stage has a local README describing its inputs, outputs, and
operating contract.

## Documentation

See [`docs/README.md`](docs/README.md) for the documentation index and
[`docs/COMMANDS.md`](docs/COMMANDS.md) for the Make target reference.

Related repositories:

- [`tohio/slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data) —
  synthetic pretraining, SFT, DPO, and distillation data generation.
- [`tohio/slm-distillation`](https://github.com/tohio/slm-distillation) —
  response and logits distillation workflows.
- [`tohio/slm-reasoning`](https://github.com/tohio/slm-reasoning) — reasoning
  model experiments using SLM checkpoints.

## Testing

Run CPU/API contract tests without pipeline artifacts:

```bash
make test-unit
```

Validate the installed CUDA stack without downloading a dataset or loading a
trained checkpoint:

```bash
make test-gpu-gate
```

Validate existing stage artifacts:

```bash
make test-data-pipeline SIZE=125m
make test-gpu-pipeline SIZE=125m
```

The artifact targets intentionally fail when required outputs are absent.
See [`tests/README.md`](tests/README.md) for the mini pipeline sequence and
controlled SFT comparison workflow.

## License

SLM is licensed under the [MIT License](LICENSE).
