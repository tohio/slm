# slm

End-to-end training and publication pipeline for decoder-only language models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

SLM builds small decoder-only language models from curated pretraining data
through supervised fine-tuning, DPO alignment, evaluation, native Hugging Face
export, inference, and vLLM serving. The repository keeps each stage explicit
so data, checkpoints, and validation artifacts can be inspected independently.

## Architecture

![SLM pipeline architecture](docs/architecture.svg)

The pipeline has four main layers: data preparation, model training,
post-training, and publication. Training checkpoints use the repository's SLM
implementation; publication converts them to native Transformers
`LlamaForCausalLM` packages.

Model lineage:

```text
pretrain/final
  └── sft_instruct/final
      ├── dpo_chat/final
      └── sft_code/final
          └── sft_code_completion/final (optional)
```

See [Architecture](docs/ARCHITECTURE.md) for component ownership and artifact
flow.

## Features

- Configured `mini`, 125M, 350M, and 1B decoder-only model sizes
- Source-aware curation, validation, deduplication, and reproducible tokenization
- Custom BPE tokenizer and chat template
- Full-parameter pretraining, instruct/code SFT, and chat DPO
- Cheap preflight gates separated from expensive artifact validation
- Native Hugging Face export without `trust_remote_code`
- Local inference and OpenAI-compatible vLLM serving
- RUN_ID-based artifact upload and restore

Published variants follow this contract:

| Variant | Training checkpoint | Hub repository |
|---|---|---|
| Base | `$RESULTS_DIR/runs/<size>/pretrain/final` | `tohio/slm-<size>` |
| Instruct | `$RESULTS_DIR/runs/<size>/sft_instruct/final` | `tohio/slm-<size>-instruct` |
| Chat | `$RESULTS_DIR/runs/<size>/dpo_chat/final` | `tohio/slm-<size>-chat` |
| Code | `$RESULTS_DIR/runs/<size>/sft_code/final` | `tohio/slm-<size>-code` |

## Getting Started

### Prerequisites

- Python 3.12
- Git
- Persistent storage for full data runs
- NVIDIA GPU and compatible driver for training
- Hugging Face and S3 credentials only for the stages that use them

See [Hardware](docs/HARDWARE.md) and [Disk setup](docs/DISK_SETUP.md) before
starting a full run.

### Installation

```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
make install
```

On an NVIDIA training host, install and validate the pinned CUDA 13.0 stack:

```bash
make install-gpu
make test-gpu-gate
```

### Configuration

Set the paths and credentials required by your selected stages in `.env`:

```bash
DATA_DIR=/data/slm/data
RESULTS_DIR=results
# EXPORTS_DIR=/data/slm/exports

HF_TOKEN=...
HF_USERNAME=tohio
S3_BUCKET=...
S3_PREFIX=slm/data
```

`EXPORTS_DIR` defaults to `$RESULTS_DIR/exports`. Never commit `.env`.

### Usage

Prepare data:

```bash
make curate SIZE=125m WORKERS=62
make validate SIZE=125m
make tokenizer SIZE=125m
make tokenize SIZE=125m
make artifacts-upload SIZE=125m
```

Restore artifacts and generate hardware-specific configs on the GPU host:

```bash
make setup-gpu \
  DATA_DIR=/data/slm/data \
  SIZE=125m \
  RUN_ID=125m-20260629-a8f3c9
make accelerate-config-single
make config-gen SIZE=125m GPUS=1
```

Train:

```bash
make pretrain SIZE=125m GPUS=1
make reinit-embeds SIZE=125m
make prepare-sft SIZE=125m
make sft-instruct SIZE=125m GPUS=1
make sft-code SIZE=125m GPUS=1
make prepare-dpo SIZE=125m
make dpo-chat SIZE=125m GPUS=1
```

Build a local native export before publishing:

```bash
make export-local SIZE=125m
```

The complete command surface is in the
[command reference](docs/COMMANDS.md).

## Project Structure

```text
slm/
├── alignment/      # DPO data and training
├── config/         # shared paths, data mix, and runtime contracts
├── config_gen/     # hardware-aware training configuration
├── curator/        # pretraining data curation
├── docs/           # project-level guides and references
├── eval/           # benchmark and behavior evaluation
├── export/         # native Hugging Face packaging
├── finetune/       # instruct, code, and code-completion SFT
├── inference/      # local generation and chat
├── model/          # decoder-only Transformer implementation
├── pretrain/       # tokenization and base-model training
├── scripts/        # stage-neutral utilities and diagnostics
├── serve/          # vLLM and Kubernetes serving assets
├── tests/          # unit, GPU-gate, and artifact tests
├── tokenizer/      # tokenizer training and validation
└── validation/     # post-curation validation
```

Each non-trivial stage has a local README describing its ownership and
contracts.

## Documentation

Start with the [documentation index](docs/README.md).

## Testing

Run the cheap CPU gate for normal changes:

```bash
make test-unit
```

Validate a new GPU image or dependency upgrade once:

```bash
make test-gpu-gate
```

Stage artifact tests load existing outputs and do not retrain models:

```bash
make test-data-pipeline SIZE=mini
make test-artifacts SIZE=mini
```

See [Tests](tests/README.md) for the cost-aware test policy.

## Status

The pipeline is configured for `mini`, `125m`, `350m`, and `1b`. `mini` is the
bounded integration target; larger runs require prepared artifacts and
appropriate GPU capacity.

## License

Licensed under the [MIT License](LICENSE).

## Related Projects

- [slm-synthetic-data](https://github.com/tohio/slm-synthetic-data) — synthetic
  data generation for pretraining, SFT, DPO, and distillation
- [slm-distillation](https://github.com/tohio/slm-distillation) — response and
  logits distillation experiments
- [slm-reasoning](https://github.com/tohio/slm-reasoning) — reasoning-model
  experiments
