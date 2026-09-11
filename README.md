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

Each dataset run carries a matched byte-level BPE tokenizer and tokenized
train/validation/test artifacts. The corpus is English-focused, including code
and technical text; Unicode support does not imply multilingual capability.
The training pipeline initializes the decoder from scratch, then branches from
the instruct checkpoint into code-specialized and chat-aligned models. Export
converts each branch into a native Transformers Llama package for standard
inference and vLLM serving.

| Size | Approx. parameters | Layers | Hidden size | Q/KV heads | Context |
|---|---:|---:|---:|---:|---:|
| `smoke` | 21.7M | 6 | 384 | 6 / 2 | 1,024 |
| `mini` | 69.9M | 17 | 512 | 8 / 4 | 2,048 |
| `125m` | 125.3M | 16 | 768 | 12 / 4 | 2,048 |
| `350m` | 351.3M | 27 | 1,024 | 16 / 8 | 2,048 |
| `1b` | 1.012B | 21 | 2,048 | 32 / 8 | 4,096 |

All profiles use RoPE, RMSNorm, SwiGLU, grouped-query attention,
pre-normalized residual blocks, tied token embeddings, bias-free projections,
and generation KV caching. Smoke and Mini are development profiles, not
production-export profiles.

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
- Opt-in, single-call web-search runtime and tool-use SFT support using the
  existing tool tokens; see [tool calling](docs/TOOL_CALLING.md).

## Pretraining data and dataset reuse

Train, validation, and final-only test use one matched artifact contract.
`DATASET_SIZE` can differ from model `SIZE` while model-specific budgets still
control training. Artifact stages remain user-selected. With `ARTIFACT_BACKEND=hf`,
curated/validated text goes to Dataset repositories and operational objects go
to Storage Buckets; models are published separately after training.
See [pretraining data and reuse](docs/PRETRAINING_DATA.md).

## Getting Started

SLM uses separate curation and training environments. The public setup commands
are `make setup-curate` and `make setup-train`; both accept
`INSTALLER=pip|uv|conda` (pip/venv by default). Each includes shared
`requirements.txt` through its role-specific requirements file. Do not layer the
two roles into the same `.venv`. See [installer details](infra/README.md).

### CPU curation server

Use an Ubuntu host with persistent storage, Python 3 available for bootstrap,
and administrative access for system-package installation. Configure your
Hugging Face token and accept gated-source terms listed in the curation guide.
**W&B is required:** set `WANDB_API_KEY` and `WANDB_PROJECT` in `.env`.
Storage credentials are required only when uploading/restoring selected
artifacts; `HF_USERNAME` is required for model publication, not local curation.

```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
vi .env

make setup-curate DATA_DIR=/data/slm/data

make curate SIZE=smoke
make validate SIZE=smoke
make tokenizer SIZE=smoke
make tokenizer-test SIZE=smoke
make tokenize SIZE=smoke
```

Setup installs the curation software, prepares and checks the FastText and matched
KenLM/SentencePiece model assets, and persists `DATA_DIR` in `.env`. No separate
model-download step or shell activation is required for these Make commands.
Existing valid assets are reused; missing or invalid assets make setup fail rather
than report success. See [installer details](infra/README.md) for interactive use.

Smoke is a bounded execution check through tokenization, not a model-quality
measurement. Mini is an **optional** end-to-end development profile requiring
at least **1.4B usable training tokens**; it is not a prerequisite for choosing
125M, 350M, or 1B. Curation time depends on CPU, network, cache, storage, and
source availability; no fixed runtime is promised.

For a complete data-preparation workflow, explicitly select what to upload:

```bash
make curate-all \
  SIZE=125m WORKERS=62 DATA_DIR=/data/slm/data \
  ARTIFACT_STAGES=validated,tokenized,tokenizer,metadata
```

This runs curation → validation → tokenizer training/checks → tokenization →
artifact checks → selected upload, then prints the source dataset `RUN_ID`.
Choose workers for your host; 62 is an example for a 64-vCPU machine.
`make curate SIZE=...` alone only constructs the corpus. See
[`docs/CURATION.md`](docs/CURATION.md) for stage-by-stage execution, optional
Mini, gated sources, reuse, and transfer configuration.

### GPU training server

Use a supported NVIDIA GPU host with a working compatible driver, the CUDA
13.0 development toolkit for the FA3 build, and administrative access.
Prepare a separate checkout/environment and `.env` as
above, including the required W&B settings and credentials for the source
artifact backend. Training dependencies are separate from the curation stack.

For a **complete new training run**, use:

```bash
make train-all \
  SIZE=125m GPUS=1 \
  RUN_ID=125m-YYYYMMDD-abcdef \
  DATA_DIR=/data/slm/data
```

Replace the placeholder with the recorded **source dataset** run. `train-all`
already invokes `setup-train`, restores selected artifacts, generates configs,
and runs pretraining/instruct/code/DPO stages. Do not run setup separately first
when using this route. Evaluation, export, publication, and serving are explicit
subsequent operations.

**Alternatively**, set up and launch individual stages:

```bash
make setup-train \
  SIZE=125m RUN_ID=125m-YYYYMMDD-abcdef DATA_DIR=/data/slm/data
make config-gen SIZE=125m GPUS=1
make pretrain SIZE=125m GPUS=1
```

Use the same chosen `GPUS` count for configuration and launch; 1 is only an
example. With no source run selected, `setup-train` installs the environment
only. Restore defaults to `tokenized,tokenizer,metadata`, not every uploaded
stage. Include `validated` in `ARTIFACT_STAGES` for test-document completions.
`DATASET_SIZE` and `DATASET_RUN_ID` explicitly select a different source profile
and run without changing the model output size.

Use [`docs/TRAIN.md`](docs/TRAIN.md) for post-training, resume, final evaluation,
and export, and [`infra/README.md`](infra/README.md) for installer prerequisites.

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
acceptance, and checks against existing data or model artifacts.
CPU-executed model tests use an already installed training environment; the
supported training setup targets NVIDIA GPU hosts, not CPU-only workstations.
Curation-only suites use the curation environment. Tests do not launch full
curation or training; some bounded model tests perform synthetic optimizer steps.

See [`docs/TESTING.md`](docs/TESTING.md) for test order, commands, and artifact
requirements.

For an isolated model-versus-data learning control, see the existing
[HF diagnostic scripts](scripts/README.md). They use the selected model recipe
and never replace the completed training run.

## License

SLM is licensed under the [MIT License](LICENSE).
