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
**Framework:** NVIDIA NeMo 1.x + NeMo-Aligner 0.7.0 + NeMo Curator 0.7.1
**Infrastructure:** Cloud CPU instances (data) + Cloud GPU instances (training)
**Container:** `nvcr.io/nvidia/nemo:25.02` (requires NGC account)

---

## Pipeline Architecture

```
Common Crawl WARCs
      │
      ▼
 NeMo Curator          data curation (CPU instance)
      │
      ▼
 Custom Tokenizer      BPE, 32k vocab, sentencepiece
      │
      ▼
 Pre-Training          NeMo 1.x megatron_gpt_pretraining.py → .nemo
      │
      ▼
 SFT (chat → code)    NeMo-Aligner GPTSFTModel
      │
      ▼
 DPO Alignment         NeMo-Aligner GPTDPOModel
      │
      ▼
 HuggingFace Export    for inference and evaluation
```

---

## Repository Structure

```
slm/
├── Dockerfile                        NeMo 25.02-based image (NGC auth required)
├── Makefile
├── requirements.txt
├── .env.sample
├── curator/                          Stage 1: data curation pipeline
├── tokenizer/                        Custom BPE tokenizer training
├── pretrain/                         Stage 2: pre-training (NeMo 1.x)
│   ├── configs/
│   │   ├── gpt_125m.yaml
│   │   ├── gpt_350m.yaml
│   │   └── gpt_1b.yaml
│   ├── scripts/
│   │   ├── train.sh
│   │   └── convert_ckpt.sh
│   ├── train.py
│   └── README.md
├── finetune/                         Stage 3: SFT (NeMo-Aligner)
├── alignment/                        Stage 4: DPO (NeMo-Aligner)
├── eval/
└── infra/
    └── setup_gpu_instance.sh
```

---

## Prerequisites

### NGC Account

This pipeline uses `nvcr.io/nvidia/nemo:25.02` as its base container. Requires a free NGC account and API key.

1. Create a free account at [ngc.nvidia.com](https://ngc.nvidia.com)
2. Generate an API key under **Setup → Generate API Key**
3. Accept the NeMo container license — visit [catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo), find the `25.02` tag, and accept the license agreement. Without this step, `docker build` will fail with a `412 Precondition Failed` error even after a successful NGC login.
4. Add your API key to `.env`:

```bash
cp .env.sample .env
# Edit .env and set NGC_API_KEY=your-key-here
```

---

## Quick Start

### Data Curation (CPU instance)

```bash
make init-dirs
make docker-build
make download-models
make download-data
make curate-full
make upload-data
```

### Training (GPU instance)

```bash
# 1. First-time setup — NGC login, S3 pull, Docker build
make setup-instance

# 2. Prepare fine-tuning datasets
make prepare-sft-data
make prepare-dpo-data

# 3. Pre-train (NeMo 1.x — saves .nemo directly)
make pretrain                    # 125M, 1 GPU (default)
# make pretrain SIZE=350m GPUS=4
# make pretrain SIZE=1b GPUS=4

# 4. Supervised fine-tuning (NeMo-Aligner)
make sft

# 5. DPO alignment (NeMo-Aligner)
make dpo

# 6. Evaluate and export
make eval-dpo
make convert-hf
```

---

## Scaling Path

| Scale | Layers | Hidden | Heads | Tokens | GPUs |
|---|---|---|---|---|---|
| **125M** (baseline) | 12 | 768 | 12 | ~2.5B | 1 |
| 350M | 24 | 1024 | 16 | ~7B | 4 |
| 1B | 32 | 2048 | 16 | ~20B | 4 (TP=2) |

```bash
make pretrain SIZE=350m GPUS=4
make pretrain SIZE=1b   GPUS=4
```

---

## Infrastructure

### Container

`nvcr.io/nvidia/nemo:25.02` — NVIDIA's official NeMo Framework container.

| Component | Version |
|---|---|
| `nemo-toolkit` | `2.2.1` |
| `nemo-aligner` | `0.7.0` |
| `megatron-core` | `0.11.1` |
| `apex` | `0.1` (pre-compiled) |
| `transformer-engine` | `1.14.0` (pre-compiled) |
| `nemo-curator` | `0.7.1` |
| Python | `3.10`, Ubuntu 22.04 |

### Training Stack

Pre-training uses **NeMo 1.x** (`megatron_gpt_pretraining.py` from `/opt/NeMo/examples/`) which saves checkpoints directly in `.nemo` format. SFT and DPO use **NeMo-Aligner 0.7.0**, which loads `.nemo` checkpoints natively.

```
make pretrain   →  megatron_gpt_pretraining.py  →  last.nemo
make sft        →  NeMo-Aligner GPTSFTModel      →  loads last.nemo
make dpo        →  NeMo-Aligner GPTDPOModel      →  loads SFT .nemo
make convert-hf →  HuggingFace format
```

No checkpoint conversion step is needed.

### AWS

| Artifact | S3 path | Needed for |
|---|---|---|
| Curated JSONL | `curated/pii/*.jsonl` | Tokenizer training, pass 2 |
| Tokenized mmap | `curated/tokenized/*.bin, *.idx` | Pre-training |
| Tokenizer model | `tokenizer/` | All training stages |
| Quality classifier | `models/quality_classifier.bin` | Pass 2 quality filter |

---

## Design Decisions

**Why from scratch instead of continued pre-training?**
Starting from an existing checkpoint is the right production choice. We start from scratch deliberately — it exercises every stage of the pipeline and provides full visibility into how data quality and tokenizer design interact with training dynamics.

**Why DPO over PPO?**
At small model scale, PPO's actor-critic setup requires four models simultaneously and is sensitive to reward scaling. DPO achieves comparable alignment with a simpler training loop and no separate reward model.

**Why a custom tokenizer?**
A tokenizer trained on your specific data mix encodes domain patterns more efficiently. Special tokens (`<|user|>`, `<|assistant|>`, `<|code|>`) are baked in from the start.

**Why sequential SFT (chat → code)?**
Sequential fine-tuning produces independently evaluable checkpoints at each stage, making regressions immediately visible. The code SFT uses a lower learning rate to reduce catastrophic forgetting.

**Why NeMo 1.x for pretraining (not NeMo 2.x)?**
NeMo 2.x (`nemo.collections.llm.GPTModel`) saves distributed checkpoints in a format that NeMo-Aligner 0.7.0 cannot load directly — there is no export connector registered for the generic `GPTModel` in `nemo:25.02`. NeMo 1.x (`megatron_gpt_pretraining.py`) saves `.nemo` tarballs that NeMo-Aligner loads natively, giving a clean end-to-end pipeline with no conversion step. NeMo 2.x end-to-end support (pretrain + SFT + DPO) becomes viable in later containers (`25.04+`) with NeMo-RL replacing NeMo-Aligner.

**Why NGC instead of a self-contained public image?**
The NeMo training stack requires Apex and Transformer Engine pre-compiled against a specific CUDA/PyTorch version. The NGC container ships these pre-compiled and validated — significantly reducing build time and dependency fragility.

---

## References

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)
- [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner)
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)
- [DPO](https://arxiv.org/abs/2305.18290)

---

## License

MIT