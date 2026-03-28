# SLM — Small Language Model Pipeline

A production-minded, end-to-end pipeline for training a small language model from scratch using NVIDIA NeMo. Covers the full lifecycle: large-scale data curation, pre-training, supervised fine-tuning, and alignment via DPO — with infrastructure designed for cost efficiency on cloud GPU instances.

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

![Architecture](docs/architecture.svg)

---

## Repository Structure

```
slm/
├── README.md            Overview and quick start
├── HARDWARE.md          Hardware recommendations and scaling guide
├── Makefile
├── requirements.txt
├── environment.yml
├── curator/             Stage 1: data curation pipeline
├── tokenizer/           Custom BPE tokenizer training
├── pretrain/            Stage 2: pre-training
├── finetune/            Stage 3: supervised fine-tuning
├── alignment/           Stage 4: DPO alignment
├── eval/                Evaluation suite (all stages)
├── infra/               GPU instance setup
└── notebooks/
    └── exploration.ipynb
        ├── Section 1    Dataset Exploration
        ├── Section 2    Training Run
        └── Section 3    Inference
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

## Hardware Recommendations

**See [HARDWARE.md](HARDWARE.md) for a detailed guide.**

Quick summary — train in four runs:

| Run | Goal | Model | Hardware | Duration | Cost |
|---|---|---|---|---|---|
| 1 | Validate pipeline | 125M | 2x A100 | 30-45 min | $3-5 |
| 2 | Full baseline | 125M | 4x A100 or 2x H100 | 4-5 hrs | $12-22 |
| 3 | Prove scaling | 350M | 6-8x H100 | 5-7 hrs | $60-100 |
| 4 | Production scale | 1B | 8-16x H100 | 12-26 hrs | $240-330 |

**Total:** ~40-60 hours wall-clock, $365-555

Each run demonstrates a key principle: *validation → efficiency → scaling → tensor parallelism.*

---

## Getting Started

**Prerequisites**
- Python 3.10+
- NVIDIA GPU (recommended: A10G or better for pretraining)
- NVIDIA NeMo dependencies (see `environment.yml`)
- AWS account (for spot instance data curation and S3 storage)

**Installation — pip**

```bash
git clone https://github.com/tohio/slm.git
cd slm

python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
cp .env.sample .env
# Add your API keys and S3 config to .env
```

**Installation — conda (recommended for NeMo)**

```bash
git clone https://github.com/tohio/slm.git
cd slm

conda env create -f environment.yml
conda activate slm
cp .env.sample .env
# Add your API keys and S3 config to .env
```

**Installation — uv**

```bash
git clone https://github.com/tohio/slm.git
cd slm

uv venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

uv pip install -r requirements.txt
cp .env.sample .env
# Add your API keys and S3 config to .env
```

**Run the pipeline**

```bash
# 1. On spot instance — curate data
make download-data
make curate
make tokenizer
make upload-data S3_BUCKET=my-bucket

# 2. On GPU instance — train (see HARDWARE.md for which GPUs to use)
make setup-instance S3_BUCKET=my-bucket
make prepare-sft-data
make prepare-dpo-data

# Run 1: Pipeline validation (2x A100)
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=2

# Run 2: 125M full training (choose one)
# Option A: 4x A100 (cost-conscious)
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=4
# Option B: 2x H100 (speed-focused)
# make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=2

# Run 3: 350M full training (choose one)
# Option A: 6x H100 (balanced)
make pretrain CONFIG=pretrain/configs/gpt_350m.yaml GPUS=6
# Option B: 8x H100 (maximum efficiency)
# make pretrain CONFIG=pretrain/configs/gpt_350m.yaml GPUS=8

# Run 4: 1B production-scale training (choose one)
# Option A: 8x H100 with tensor parallelism (balanced)
make pretrain CONFIG=pretrain/configs/gpt_1b.yaml GPUS=8
# Option B: 16x H100 with tensor parallelism (fastest)
# make pretrain CONFIG=pretrain/configs/gpt_1b.yaml GPUS=16

# 3. Evaluate
make eval-dpo

# 4. Export
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

## Production Considerations

This project is intentionally scoped for demonstration. In a production system:

- **Data pipeline** — the NeMo Curator pipeline would run on a distributed Spark cluster (AWS EMR or Databricks) rather than a single spot instance, reducing curation time for trillion-token datasets from days to hours.
- **Training infrastructure** — multi-node training would use AWS EFA networking for low-latency GPU-to-GPU communication. Spot instance interruptions would be handled via NeMo's checkpoint resumption rather than restarting from scratch.
- **Tokenizer** — the custom BPE tokenizer would be versioned and stored in a model registry (MLflow or Weights & Biases) alongside the model checkpoints it was trained with, to prevent train/serve skew.
- **Experiment tracking** — Weights & Biases or MLflow would track loss curves, gradient norms, and throughput (tokens/sec) across all training stages, making it possible to diagnose instability or compare runs systematically.
- **Checkpoint management** — checkpoints would be stored in S3 with versioning enabled and a retention policy. Only the top-k checkpoints by validation loss would be kept to manage storage costs at scale.
- **Evaluation** — the eval suite would run automatically after each stage (pretrain, SFT, DPO) in CI, gating promotion to the next stage on a minimum score threshold rather than relying on manual inspection.
- **Serving** — the exported HuggingFace checkpoint would be served via vLLM for high-throughput inference with continuous batching, exposed behind a FastAPI layer with async request handling.
- **Observability** — token throughput, GPU utilisation, and training loss would be streamed to CloudWatch or Grafana during training runs for real-time monitoring and alerting on divergence.

---

## References

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)
- [NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner)
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022
- [DPO](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023

---

## Related Projects

This repo is part of a broader AI engineering portfolio:

- [rag-pipeline](https://github.com/tohio/rag-pipeline) — modular RAG pipeline: ingestion, embedding, retrieval, and generation
- [agentic-rag](https://github.com/tohio/agentic-rag) — extends the RAG pipeline with tool use, query routing, and multi-step reasoning
- [multi-agent](https://github.com/tohio/multi-agent) — autonomous multi-agent investment research system using CrewAI

---

## License

MIT