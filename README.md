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

## Scaling Path

The architecture scales by config change only — no code changes required:

| Scale | Layers | Hidden | Heads | Tokens | GPUs |
|---|---|---|---|---|---|
| **125M** (baseline) | 12 | 768 | 12 | ~2.5B | 1 |
| 350M | 24 | 1024 | 16 | ~7B | 4 |
| 1B | 32 | 2048 | 16 | ~20B | 4 (TP=2) |

---

## Quick Start

```bash
# 1. Install dependencies (install PyTorch manually first)
make setup

# 2. On spot instance — curate data
make download-data
make curate
make tokenizer
make upload-data S3_BUCKET=my-bucket

# 3. On GPU instance — train
make setup-instance S3_BUCKET=my-bucket
make prepare-sft-data
make prepare-dpo-data
make pretrain
make sft
make dpo

# 4. Evaluate
make eval-dpo

# 5. Export
make convert-hf
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
