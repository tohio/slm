# SLM

From-scratch training pipeline for a family of dense decoder-only language
models, from corpus construction through native Hugging Face export.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

SLM is a stage-based research pipeline for building small language models
without starting from an existing foundation-model checkpoint. It keeps data,
tokenizer, model, training, and export contracts explicit so execution can be
rehearsed with `smoke`, functionality can be exercised with `mini`, and the
same workflow can scale to 125M, 350M, and 1B.

The repository produces four model variants:

| Variant | Lineage | Checkpoint |
|---|---|---|
| Base | pretrained from scratch and finalized for post-training | `$RESULTS_DIR/runs/<size>/pretrain/final` |
| Instruct | base → instruct SFT | `$RESULTS_DIR/runs/<size>/sft_instruct/final` |
| Code | instruct → code SFT | `$RESULTS_DIR/runs/<size>/sft_code/final` |
| Chat | instruct → DPO | `$RESULTS_DIR/runs/<size>/dpo_chat/final` |

## Architecture

![SLM pipeline and model architecture](docs/architecture.svg)

The data pipeline creates a validated corpus and size-specific BPE tokenizer.
The training pipeline initializes the decoder from scratch, then branches from
the instruct checkpoint into code-specialized and chat-aligned models. Export
converts each branch into a native Transformers Llama package for standard
inference and vLLM serving.

| Size | Parameters | Layers | Hidden size | Q/KV heads | Context |
|---|---:|---:|---:|---:|---:|
| `smoke` | 22M | 6 | 384 | 6 / 2 | 1,024 |
| `mini` | 69.9M | 17 | 512 | 8 / 4 | 2,048 |
| `125m` | 125M | 16 | 768 | 12 / 4 | 2,048 |
| `350m` | 350M | 27 | 1,024 | 16 / 8 | 2,048 |
| `1b` | 1B | 21 | 2,048 | 32 / 8 | 4,096 |

All profiles use RoPE, RMSNorm, SwiGLU, grouped-query attention,
pre-normalized residual blocks, tied token embeddings, bias-free projections,
and generation KV caching.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for component ownership,
model lineage, and artifact flow.

## Features

- Reproducible multi-source corpus construction with quality filtering,
  deterministic deduplication, controlled blending, completion manifests, and
  run-scoped artifact transfer.
- One byte-level BPE tokenizer contract across curation, pretraining,
  post-training, evaluation, and export.
- Hardware-aware pretraining, SFT, and DPO configuration generation based on
  model size, GPU type/count, and VRAM policy.
- Independent instruct, code, and DPO chat branches with assistant-only SFT
  loss and fixed-reference preference optimization.
- Fail-fast data, tokenizer, checkpoint, model-conversion, and CUDA
  compatibility gates.
- Native Transformers Llama export, local generation, Hub publication, and
  vLLM serving.

## Getting Started

SLM uses separate environments for data curation and model training. Do not
layer the curation and training dependency stacks into the same `.venv`.

### CPU curation server

Use this path on a CPU/data-processing host. It installs the curation stack,
downloads the required FastText and KenLM assets, then validates the pipeline
with `smoke` before running `mini` or a production-size curation.

Prerequisites:

- Ubuntu host with persistent storage for curation.
- Hugging Face account and token.
- AWS account, S3 bucket, and credentials when uploading artifacts.
- Weights & Biases credentials if enabled by the active workflow.

The curation guide lists gated datasets whose terms must be accepted before
downloading them.

```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
vi .env

make setup-data-dir DATA_DIR=/data/slm/data
source .venv/bin/activate

make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model    DATA_DIR=/data/slm/data
make check-curation-prereqs  DATA_DIR=/data/slm/data

make curate-smoke DATA_DIR=/data/slm/data
make validate SIZE=smoke DATA_DIR=/data/slm/data

make curate-mini DATA_DIR=/data/slm/data
make validate SIZE=mini DATA_DIR=/data/slm/data
```

`curate-smoke` is the first pipeline-validation run. `curate-mini` is the
functional mini-scale curation run. Curation time depends on CPU count, network
bandwidth, cache state, storage throughput, and Common Crawl availability, so
the repository does not publish fixed runtime estimates.

All curation entry points fail before source processing begins when the
FastText or KenLM model files are missing. Install them explicitly with the two
download targets above.

For a production-size run after smoke and mini are healthy:

```bash
make curate SIZE=125m WORKERS=62 DATA_DIR=/data/slm/data
# or SIZE=350m / SIZE=1b
```

Use [`docs/CURATION.md`](docs/CURATION.md) for gated-source prerequisites,
worker sizing, resume behavior, validation, tokenization, and artifact upload.

### GPU training server

Use this path on a supported NVIDIA GPU host after the curation host has
produced and uploaded a run-scoped artifact set. The training stack is pinned
separately from the curation stack.

```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
vi .env

make setup-gpu \
  SIZE=125m \
  RUN_ID=125m-YYYYMMDD-abcdef \
  DATA_DIR=/data/slm/data

make check-training-env
```

Then use the stage-specific commands or the complete new-run workflow:

```bash
make train-all \
  SIZE=125m \
  GPUS=1 \
  RUN_ID=125m-YYYYMMDD-abcdef \
  DATA_DIR=/data/slm/data
```

Use [`docs/TRAIN.md`](docs/TRAIN.md) for pretraining, instruct SFT, code SFT,
DPO, resume procedures, evaluation, and export.

## Project Structure

```text
slm/
├── alignment/       DPO data preparation and training
├── config/          shared paths, runtime policy, data mix, and token targets
├── config_gen/      hardware-aware training configuration generation
├── curator/         source loading, filtering, deduplication, and blending
├── docs/            operational guides, architecture, and command reference
├── eval/            benchmark and deterministic behavior evaluation
├── export/          native model conversion, validation, and publication
├── finetune/        SFT data preparation and instruct/code training
├── inference/       interactive and batch generation
├── infra/           data-host and GPU-host setup and validation
├── model/           decoder-only Transformer implementation
├── pretrain/        binary tokenization and base-model training
├── serve/           vLLM launcher and Kubernetes manifests
├── tests/           CPU, GPU, artifact, and comparison checks
├── tokenizer/       BPE tokenizer training and validation
└── validation/      post-curation document validation
```

## Documentation

See [`docs/README.md`](docs/README.md) for the documentation index and
[`docs/COMMANDS.md`](docs/COMMANDS.md) for the complete Make target reference.

Related repositories:

- [`tohio/slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data) —
  synthetic pretraining, SFT, DPO, and distillation data generation.
- [`tohio/slm-distillation`](https://github.com/tohio/slm-distillation) —
  response and logits distillation workflows.
- [`tohio/slm-reasoning`](https://github.com/tohio/slm-reasoning) — reasoning
  model experiments using SLM checkpoints.

## Troubleshooting

Use [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for environment,
dataset access, storage, resume, artifact transfer, CUDA, and training
problems. Add new operational failure procedures there instead of expanding
the root README.

## Testing

Tests are separated into CPU model/training contracts, environment
acceptance, and checks against existing data or model artifacts. Model-facing
CPU tests use the pinned training stack rather than the separate curation stack.
Full curation and training are not launched merely to test the repository.

See [`docs/TESTING.md`](docs/TESTING.md) for test order, commands, and artifact
requirements.

## License

SLM is licensed under the [MIT License](LICENSE).
