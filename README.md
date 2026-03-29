# SLM — Small Language Model Pipeline

A production-minded, end-to-end pipeline for training a small language model from scratch using NVIDIA NeMo. Covers the full lifecycle: large-scale data curation, pre-training, supervised fine-tuning, and alignment via DPO — with infrastructure designed for cost efficiency on cloud GPU instances.

> Built to demonstrate practical expertise across data engineering, distributed training, and LLM alignment — not just theory.

---

## Overview

Most LLM projects start from a pretrained checkpoint. This one doesn't. SLM is built from raw, imperfect web data through to an aligned, instruction-following model capable of general conversation and code generation.

The pipeline is modular — each stage is independently runnable, reproducible, and documented with the design decisions that shaped it.

**Model:** GPT-style decoder-only transformer, ~125M parameters (scalable to 350M and 1B)
**Domain:** General chat → coding (sequential fine-tuning)
**Alignment:** Direct Preference Optimization (DPO)
**Framework:** NVIDIA NeMo + NeMo Curator + NeMo Aligner
**Infrastructure:** AWS Spot (data) + Cloud GPU instances (training)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│               Stage 1: Data Curation                │
│                  (NeMo Curator)                     │
│  Common Crawl (WARC) ──┐                            │
│  Wikipedia (EN)    ────┼──► Curator Pipeline ──► S3 │
│  CodeSearchNet     ────┘                            │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│             Stage 2: Pre-Training                   │
│              (NeMo + Megatron-Core)                 │
│  GPT ~125M params, BF16, single/multi GPU           │
│  Trained on general text corpus (~2.5B tokens)      │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│          Stage 3: Supervised Fine-Tuning            │
│                 (NeMo Aligner)                      │
│  SFT-1: General chat (OpenAssistant / Dolly)        │
│  SFT-2: Coding (CodeSearchNet / The Stack)          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Stage 4: Alignment (DPO)               │
│                 (NeMo Aligner)                      │
│  Preference data: UltraFeedback / HH-RLHF           │
│  No separate reward model required                  │
└─────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
slm/
├── Dockerfile           Self-contained NeMo image (no NGC auth required)
├── Makefile
├── requirements.txt
├── environment.yml
├── curator/             Stage 1: data curation pipeline
├── tokenizer/           Custom BPE tokenizer training
├── pretrain/            Stage 2: pre-training
├── finetune/            Stage 3: supervised fine-tuning
├── alignment/           Stage 4: DPO alignment
├── eval/                Evaluation suite (all stages)
└── infra/               GPU instance setup
```

---

## Quick Start

### Docker (recommended)

The image is self-contained and built from public sources only — **no NGC account or API token required**.

```bash
# Build once
make docker-build

# Data curation (CPU — runs on a cheap spot instance)
make docker-shell-cpu
# then inside the container:
make download-data
make curate
make tokenizer
make upload-data S3_BUCKET=my-bucket

# Training (GPU)
make docker-shell-gpu
# then inside the container:
make setup-instance S3_BUCKET=my-bucket
make prepare-sft-data
make prepare-dpo-data
make pretrain
make sft
make dpo

# Evaluate & export
make eval-dpo
make convert-hf
```

### Local (no Docker)

```bash
# 1. Install PyTorch first — version must match your CUDA driver
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Install remaining dependencies
pip install -r requirements.txt
```

---

## Scaling Path

The architecture scales by config change only — no code changes required:

| Scale | Layers | Hidden | Heads | Tokens | GPUs |
|---|---|---|---|---|---|
| **125M** (baseline) | 12 | 768 | 12 | ~2.5B | 1 |
| 350M | 24 | 1024 | 16 | ~7B | 4 |
| 1B | 32 | 2048 | 16 | ~20B | 4 (TP=2) |

```bash
# Override scale and GPU count at runtime
make pretrain GPUS=4 CONFIG=pretrain/configs/gpt_350m.yaml
```

---

## Infrastructure

### Docker Image

The Dockerfile builds a single image that serves both pipeline roles:

| Target | Command | GPU required |
|---|---|---|
| Data curation | `make docker-shell-cpu` | No |
| Training / alignment | `make docker-shell-gpu` | Yes |

The image is based on `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` (public DockerHub). No NVIDIA NGC account is needed.

**Validated version stack:**

| Package | Version | Notes |
|---|---|---|
| `nemo_toolkit` | `2.2.0` | `[core]` extra — excludes `mamba-ssm` (requires `nvcc`) |
| `nemo-aligner` | `0.7.0` | No `nemo_toolkit[nlp]` dep — avoids `mamba-ssm` entirely |
| `nemo-curator` | `0.7.1` | Earliest version on public PyPI (0.5.0 was NGC-only) |
| `transformers` | `>=4.48.0,<=4.48.3` | Pinned by `nemo_toolkit 2.2.0`; curator needs `>=4.48.0` |
| `fasttext` | `0.9.3` | `nemo-curator` pins this exactly; 0.9.2 will conflict |
| AWS CLI | v2 binary | `awscli` v1 (pip) hard-pins `botocore` and conflicts with `boto3` |

**Key decisions:**

- `nemo_toolkit[core]` not `[all]` or `[nlp]` — both `[all]` and `[nlp]` include `mamba-ssm==2.2.2` which requires `nvcc` to compile from source. The `runtime` base image does not ship `nvcc` (only `devel` does). `[core]` excludes it entirely.
- `huggingface-hub`, `transformers`, `pytorch-lightning`, `omegaconf`, `hydra-core`, `sentencepiece`, and `datasets` are not pinned directly — owned and versioned by `nemo_toolkit` and `nemo-curator`. Re-pinning them causes resolution conflicts across environments.
- GPU-accelerated curator ops (`cudf`, `dask-cuda`) are excluded. They require the NVIDIA PyPI index and are not needed for the CPU curation pipeline.

### AWS

Data curation runs on spot instances (CPU, storage-optimized). Training runs on on-demand GPU instances. The `upload-data` and `setup-instance` targets handle the handoff between the two via S3.

```bash
make upload-data S3_BUCKET=my-bucket S3_PREFIX=slm/data
make setup-instance S3_BUCKET=my-bucket
```

---

## Design Decisions & Tradeoffs

**Why from scratch instead of a pretrained base?**
Starting from a pretrained checkpoint is the right production choice. Starting from scratch here is intentional — it exercises every stage of the pipeline and provides full visibility into how data quality, tokenizer design, and training dynamics interact.

**Why DPO over PPO?**
At small model scale on limited hardware, PPO's actor-critic setup requires running four models simultaneously and is sensitive to reward scaling and KL penalty tuning. DPO achieves comparable alignment with a simpler training loop and no separate reward model.

**Why a custom tokenizer?**
A tokenizer trained on your specific data mix (general + code) encodes domain patterns more efficiently than GPT-2 or LLaMA tokenizers. Special tokens (`<|user|>`, `<|assistant|>`, `<|code|>`) are baked in from the start rather than retrofitted.

**Why sequential SFT (chat → code)?**
Sequential fine-tuning lets each stage be evaluated independently and makes it easier to diagnose regressions. The code SFT uses a lower learning rate specifically to reduce catastrophic forgetting of chat capabilities.

**Why a self-contained Docker image instead of the official NeMo NGC image?**
The official `nvcr.io/nvidia/nemo` image requires an NGC account and API token, which adds friction for open development and CI pipelines. The Dockerfile here reproduces the same stack from public sources only, with all transitive dependencies resolved and locked against the public PyPI index. Resolving this stack required pinning specific intermediate versions — notably upgrading `nemo_toolkit` to `2.2.0` (where `transformers>=4.48` support was added) and `nemo-aligner` to `0.7.0` (which dropped the `nemo_toolkit[nlp]` dependency that pulled in `mamba-ssm`).

---

## References

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)
- [NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner)
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022
- [DPO](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023

---

## License

MIT