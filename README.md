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
**Framework:** NVIDIA NeMo 2.x + NeMo-Aligner 0.7.0 + NeMo Curator 0.7.1
**Infrastructure:** Cloud CPU instances (data) + Cloud GPU instances (training)
**Container:** `nvcr.io/nvidia/nemo:25.02` (requires NGC account)

---

## Pipeline Architecture

![Pipeline Architecture](docs/architecture.svg)

---

## Repository Structure

```
slm/
├── Dockerfile                        NeMo 25.02-based image (NGC auth required)
├── Makefile
├── requirements.txt
├── environment.yml
├── .env.sample                       Environment variable template (copy to .env)
├── .dockerignore
├── HARDWARE.md                       GPU/instance recommendations
├── docs/
│   ├── architecture.svg
│   └── screenshots/
├── notebooks/
│   ├── inference.ipynb
│   ├── training_run.ipynb
│   ├── data_exploration.ipynb
│   ├── training_curves.ipynb
│   ├── model_comparison.ipynb
│   ├── eval_analysis.ipynb
│   ├── tokenizer_analysis.ipynb
│   ├── dataset_blend.ipynb
│   └── README.md
├── curator/                          Stage 1: data curation pipeline
│   ├── configs/
│   │   └── curator.yaml
│   ├── pipelines/
│   │   ├── pipeline.py
│   │   ├── extract.py
│   │   ├── language_filter.py
│   │   ├── heuristic_filter.py
│   │   ├── quality_filter.py
│   │   ├── dedup.py
│   │   ├── pii.py
│   │   └── tokenize_data.py
│   ├── scripts/
│   │   ├── download_cc.sh
│   │   └── upload_s3.sh
│   └── README.md
├── tokenizer/                        Custom BPE tokenizer training
│   ├── configs/
│   │   └── tokenizer.yaml
│   ├── train_tokenizer.py
│   └── README.md
├── pretrain/                         Stage 2: pre-training (NeMo 2.x)
│   ├── configs/                      Reference configs (documentation only)
│   │   ├── gpt_125m.yaml
│   │   ├── gpt_350m.yaml
│   │   └── gpt_1b.yaml
│   ├── scripts/
│   │   ├── train.sh
│   │   ├── convert_pretrain.sh       NeMo 2.x ckpt → mcore_gpt.nemo
│   │   └── convert_ckpt.sh           Export to HuggingFace format
│   ├── train.py                      NeMo 2.x entry point
│   └── README.md
├── finetune/                         Stage 3: supervised fine-tuning (NeMo-Aligner)
│   ├── configs/
│   │   ├── sft_chat.yaml
│   │   └── sft_code.yaml
│   ├── data/
│   │   └── prepare_sft.py
│   ├── scripts/
│   │   └── train_sft.sh
│   ├── train_sft.py
│   └── README.md
├── alignment/                        Stage 4: DPO alignment (NeMo-Aligner)
│   ├── configs/
│   │   └── dpo.yaml
│   ├── data/
│   │   └── prepare_dpo.py
│   ├── scripts/
│   │   └── train_dpo.sh
│   ├── train_dpo.py
│   └── README.md
├── eval/
│   ├── run_eval.py
│   ├── perplexity.py
│   ├── mmlu.py
│   ├── generation.py
│   ├── win_rate.py
│   └── README.md
├── inference.py
└── infra/
    └── setup_gpu_instance.sh
```

---

## Prerequisites

### NGC Account

This pipeline uses `nvcr.io/nvidia/nemo:25.02` as its base container. This image is hosted on NVIDIA GPU Cloud (NGC) and requires a free NGC account and API key.

1. Create a free account at [ngc.nvidia.com](https://ngc.nvidia.com)
2. Generate an API key under **Setup → Generate API Key**
3. Accept the NeMo container license — visit [catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo), find the `25.02` tag, and accept the license agreement. Without this step, `docker build` will fail with a `412 Precondition Failed` error even after a successful NGC login.
4. Add your API key to `.env`:

```bash
cp .env.sample .env
# Edit .env and set NGC_API_KEY=your-key-here
```

The `make setup-instance` command handles NGC login non-interactively using this key — no manual `docker login` required.

---

## Quick Start

### Docker (recommended)

#### Data Curation (CPU instance)

```bash
# 1. First-time host setup
make init-dirs

# 2. Build the Docker image
#    Requires NGC_API_KEY in .env — handled automatically by setup_gpu_instance.sh
make docker-build

# 3. Download fasttext language ID model
make download-models

# 4. Download Common Crawl WARC files
#    Default: 20 files. Override: make download-data N_WARC_FILES=2
make download-data

# 5. Run the full two-pass curation pipeline
make curate-full
#    Checkpointed — re-running skips completed steps automatically.

# 6. Upload all curation artifacts to S3
make upload-data                    # reads S3_BUCKET from .env
#    Uploads: curated JSONL, tokenized .bin/.idx, tokenizer model, quality classifier
#    Skip flags: --skip-jsonl, --skip-bin, --skip-tokenizer, --skip-classifier
```

#### Training (GPU instance)

```bash
# 1. First-time setup on GPU instance
#    Handles: NGC login, S3 data pull, Docker build
make setup-instance

# 2. Prepare fine-tuning datasets
make prepare-sft-data
make prepare-dpo-data

# 3. Pre-train (NeMo 2.x)
make pretrain                    # 125M, 1 GPU (default)
make pretrain SIZE=350m GPUS=4   # 350M, 4 GPUs
make pretrain SIZE=1b GPUS=4     # 1B, tensor parallel

# 4. Convert pretrain checkpoint for NeMo-Aligner
#    NeMo 2.x saves distributed checkpoints; NeMo-Aligner requires .nemo format.
make convert-pretrain

# 5. Supervised fine-tuning (NeMo-Aligner)
make sft

# 6. DPO alignment (NeMo-Aligner)
make dpo

# 7. Evaluate and export
make eval-dpo
make convert-hf
```

#### Interactive shells

```bash
make docker-shell-cpu   # CPU container — curation and data prep
make docker-shell-gpu   # GPU container — training
```

---

## Scaling Path

The architecture scales by config change only — no code modifications required:

| Scale | Layers | Hidden | Heads | Tokens | GPUs |
|---|---|---|---|---|---|
| **125M** (baseline) | 12 | 768 | 12 | ~2.5B | 1 |
| 350M | 24 | 1024 | 16 | ~7B | 4 |
| 1B | 32 | 2048 | 16 | ~20B | 4 (TP=2) |

```bash
make pretrain SIZE=350m GPUS=4
make pretrain SIZE=1b GPUS=4
```

---

## Infrastructure

### Docker Image

The Dockerfile is based on `nvcr.io/nvidia/nemo:25.02` — NVIDIA's official NeMo Framework container. It ships with all LLM training dependencies pre-compiled and tested together.

| Target | Command | GPU required |
|---|---|---|
| Data curation | `make docker-shell-cpu` | No |
| Training / alignment | `make docker-shell-gpu` | Yes |

**Component versions in `nvcr.io/nvidia/nemo:25.02`:**

| Component | Version | Notes |
|---|---|---|
| `nemo-toolkit` | `2.2.1` | NeMo 2.x LLM collection |
| `nemo-aligner` | `0.7.0` | SFT and DPO alignment |
| `megatron-core` | `0.11.1` | Megatron Core distributed training |
| `apex` | `0.1` | Pre-compiled NVIDIA Apex |
| `transformer-engine` | `1.14.0` | Pre-compiled Transformer Engine |
| `nemo-curator` | `0.7.1` | Data curation pipeline |
| Python | `3.10` | Ubuntu 22.04 base |

The image adds only three curation-specific packages not present in the base: `trafilatura`, `langdetect`, `datasketch`.

### Training Stack

Pre-training uses the **NeMo 2.x API** (`nemo.collections.llm.GPTModel`) for its Python-native configuration and cleaner megatron-core integration. SFT and DPO use **NeMo-Aligner 0.7.0**, which requires a `.nemo` format checkpoint. A conversion step (`make convert-pretrain`) bridges the two.

```
make pretrain        →  NeMo 2.x distributed checkpoint
make convert-pretrain →  mcore_gpt.nemo  (NeMo-Aligner input)
make sft             →  megatron_gpt_sft.nemo
make dpo             →  megatron_gpt_dpo.nemo
make convert-hf      →  HuggingFace format
```

### AWS

Data curation runs on CPU instances. Training runs on GPU instances. All curation artifacts are pushed to S3 from the CPU instance and pulled down on the GPU instance.

**Artifacts managed via S3:**

| Artifact | S3 path | Needed for |
|---|---|---|
| Curated JSONL | `curated/pii/*.jsonl` | Tokenizer training, pass 2 |
| Tokenized mmap | `curated/tokenized/*.bin, *.idx` | Pre-training |
| Tokenizer model | `tokenizer/` | SFT, DPO, inference |
| Quality classifier | `models/quality_classifier.bin` | Pass 2 quality filter |

```bash
# On CPU instance — push everything to S3
make upload-data                         # reads S3_BUCKET from .env
make upload-data --skip-bin              # skip if pass 2 not yet complete

# On GPU instance — pull everything from S3
make setup-instance                      # reads S3_BUCKET and NGC_API_KEY from .env
make setup-instance --skip-jsonl         # skip JSONL if only training
```

### Dask Dashboard

Available at port `8787` during curation. All `docker-shell-*` and `docker-curate` targets expose this port automatically.

**SSH tunnel:**
```bash
ssh -L 8787:localhost:8787 <your-instance>
# then open http://localhost:8787
```

---

## Inference

```bash
make inference
make inference-compare PROMPT="Write a Python function to check if a number is prime."
```

**Generation parameters:**

| Flag | Default | Notes |
|---|---|---|
| `--temperature` | `0.7` | Lower = more deterministic |
| `--max-tokens` | `512` | Maximum new tokens |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--no-chat-template` | off | Raw prompts — useful for pretrain checkpoint |

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

**Why NeMo 2.x for pretraining and NeMo-Aligner for SFT/DPO?**
NeMo 2.x provides a cleaner Python-native API for pretraining with better megatron-core integration. NeMo-Aligner 0.7.0 is the stable, production-tested path for SFT and DPO in the `nemo:25.02` container — full NeMo 2.x SFT/DPO was not yet stable in this release. A conversion step (`make convert-pretrain`) bridges the checkpoint formats between the two APIs.

**Why NGC instead of a self-contained public image?**
The NeMo training stack requires Apex and Transformer Engine, which must be compiled against a specific CUDA/PyTorch version. Building from scratch takes 20-30 minutes and is fragile across CUDA driver updates. The NGC container ships these pre-compiled and validated — significantly reducing build time and dependency hell. A free NGC account is the only requirement.

---

## References

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)
- [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner)
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022
- [DPO](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023

---

## License

MIT