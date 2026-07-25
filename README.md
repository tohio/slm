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
| `mini` | 22M | 6 | 384 | 6 / 2 | 1,024 |
| `125m` | 125M | 16 | 768 | 12 / 4 | 2,048 |
| `350m` | 350M | 27 | 1,024 | 16 / 8 | 2,048 |
| `1b` | 1B | 21 | 2,048 | 32 / 8 | 4,096 |

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

- A fresh Ubuntu instance with `git`, `sudo`, network access, and persistent
  storage sized for the selected corpus.
- A Hugging Face account and token.
- An AWS account, S3 bucket, and credentials.
- A Weights & Biases account and API key.

Before curation, sign in to the Hugging Face account associated with
`HF_TOKEN`, review the conditions, and request access to each gated source used
by the active data mix:

- [`bigcode/the-stack-dedup`](https://huggingface.co/datasets/bigcode/the-stack-dedup)
  — primary source in the code sub-mix.
- [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol)
  — supplemental code source.
- [`nvidia/Nemotron-CC-Math-v1`](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1)
  — mathematical pretraining corpus.

Dataset access is granted to the Hugging Face account, not to an individual
token. Generate `HF_TOKEN` from the same account after accepting the terms.

### Curation process

The following sequence covers infrastructure setup through uploaded,
training-ready artifacts for the 125M profile.

#### 1. Prepare persistent storage

Mount and persist the data volume before installing the repository. Follow
[`docs/DISK_SETUP.md`](docs/DISK_SETUP.md) when using a secondary volume, then
verify the selected path:

```bash
df -h /data
```

#### 2. Clone and configure the repository

```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
vi .env
```

Complete every variable in `.env`. Do not leave blank values, commented
settings, or `...` placeholders. This check must produce no output:

```bash
grep -nE '=\.\.\.|^[A-Z][A-Z0-9_]*=[[:space:]]*(#.*)?$' .env
```

#### 3. Install the data-processing environment

The setup target installs the system and Python dependencies and creates the
run directories. The two model downloads support language identification and
perplexity validation.

```bash
make setup-data-dir DATA_DIR=/data/slm/data
source ~/.bashrc
source .venv/bin/activate
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model DATA_DIR=/data/slm/data
.venv/bin/python infra/verify_environment.py
```

#### 4. Start a persistent session

```bash
tmux new -s slm-curation
nproc
```

Set `WORKERS` below the available CPU count so the operating system and
download processes retain headroom. `WORKERS=62` is the standard starting
point on a 64-vCPU instance.

#### 5. Build and validate the data artifacts

```bash
make curate SIZE=125m WORKERS=62
make test-curator SIZE=125m

make validate SIZE=125m
make test-validate SIZE=125m

make tokenizer SIZE=125m
make tokenizer-test SIZE=125m

make tokenize SIZE=125m
make test-data-pipeline SIZE=125m
```

The final aggregate test inspects the existing curator, validation, tokenizer,
and tokenized artifacts; it does not rerun the data stages. See
[`docs/TESTING.md`](docs/TESTING.md) for the test layers and when each gate
should run.

#### 6. Upload and record the run

```bash
make artifacts-upload \
  SIZE=125m \
  ARTIFACT_STAGES="tokenized,tokenizer,metadata"

cat "$DATA_DIR/runs/125m/RUN_ID"
```

The first upload creates the run ID used to restore these exact artifacts on a
training host. Record the `run_id` value printed by the upload and stored in
the `RUN_ID` file.

If a stage fails, follow [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).
Rerunning the same curation command is the normal resume path because completed
stages are reused only when their manifests still match.

### Usage

For a fresh NVIDIA training host, complete `.env` and restore the curation run
by its recorded ID:

```bash
make setup-gpu \
  DATA_DIR=/data/slm/data \
  SIZE=125m \
  RUN_ID=125m-YYYYMMDD-abcdef

source .venv/bin/activate
make test-upgrade-gpu
```

#### 1. Generate training configurations

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

#### 2. Train the base model

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

#### 3. Train instruct, chat, and code branches

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

Tests are split into CPU contracts, environment acceptance, and checks against
artifacts produced by real data or training runs. Full training is never
launched merely to test it. See [`docs/TESTING.md`](docs/TESTING.md) for the
commands and required execution order.

## License

SLM is licensed under the [MIT License](LICENSE).
