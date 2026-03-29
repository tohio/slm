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
**Infrastructure:** Cloud CPU instances (data) + Cloud GPU instances (training)

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
├── Dockerfile                        Self-contained NeMo image (no NGC auth required)
├── Makefile
├── requirements.txt
├── environment.yml
├── .env.sample                       Environment variable template
├── .dockerignore
├── HARDWARE.md                       GPU/instance recommendations
├── docs/
│   └── architecture.svg
├── notebooks/
│   └── exploration.ipynb
├── curator/                          Stage 1: data curation pipeline
│   ├── configs/
│   │   └── curator.yaml
│   ├── pipelines/
│   │   ├── pipeline.py               Orchestrator — runs all stages in order
│   │   ├── extract.py                WARC → clean text (trafilatura)
│   │   ├── language_filter.py        fastText language ID
│   │   ├── heuristic_filter.py       Gopher-style rule filters
│   │   ├── quality_filter.py         fastText quality classifier
│   │   ├── dedup.py                  Exact (MD5) + fuzzy (MinHash) dedup
│   │   ├── pii.py                    Regex-based PII redaction
│   │   └── tokenize_data.py          SentencePiece → NeMo mmap format
│   ├── scripts/
│   │   ├── download_cc.sh            Download Common Crawl WARCs
│   │   └── upload_s3.sh              Upload curated data to S3
│   └── README.md
├── tokenizer/                        Custom BPE tokenizer training
│   ├── configs/
│   │   └── tokenizer.yaml
│   ├── train_tokenizer.py
│   └── README.md
├── pretrain/                         Stage 2: pre-training
│   ├── configs/
│   │   ├── gpt_125m.yaml
│   │   ├── gpt_350m.yaml
│   │   └── gpt_1b.yaml
│   ├── scripts/
│   │   ├── train.sh
│   │   └── convert_ckpt.sh           Export to HuggingFace format
│   ├── train.py
│   └── README.md
├── finetune/                         Stage 3: supervised fine-tuning
│   ├── configs/
│   │   ├── sft_chat.yaml
│   │   └── sft_code.yaml
│   ├── data/
│   │   └── prepare_sft.py
│   ├── scripts/
│   │   └── train_sft.sh
│   ├── train_sft.py
│   └── README.md
├── alignment/                        Stage 4: DPO alignment
│   ├── configs/
│   │   └── dpo.yaml
│   ├── data/
│   │   └── prepare_dpo.py
│   ├── scripts/
│   │   └── train_dpo.sh
│   ├── train_dpo.py
│   └── README.md
├── eval/                             Evaluation suite (all stages)
│   ├── run_eval.py                   Main eval entry point
│   ├── perplexity.py
│   ├── mmlu.py
│   ├── generation.py
│   ├── win_rate.py
│   └── README.md
└── infra/
    └── setup_gpu_instance.sh
```

---

## Quick Start

### Docker (recommended)

The image is self-contained and built from public sources only — **no NGC account or API token required**.

#### Data Curation (CPU instance)

```bash
# 1. First-time host setup — creates /data, /results, /logs on the host
make init-dirs

# 2. Build the Docker image
make docker-build

# 3. Download fasttext language ID model (lid.176.bin)
make download-models

# 4. Download Common Crawl WARC files
#    Start with 2 files to validate the pipeline before a full run
make download-data N_WARC_FILES=2

# 5. Run the full curation pipeline in Docker
make docker-curate
#    Stages: extract → language_filter → heuristic_filter →
#            exact_dedup → fuzzy_dedup → pii
#    NOTE: quality_filter and tokenization are disabled on first pass
#    See curator/configs/curator.yaml for details

# 6. Train the custom BPE tokenizer on curated output
make tokenizer

# 7. Upload curated dataset to S3
make upload-data S3_BUCKET=my-bucket
```

#### Training (GPU instance)

```bash
# 1. First-time setup on GPU instance
make init-dirs
make docker-build

# 2. Pull curated data from S3
make setup-instance S3_BUCKET=my-bucket

# 3. Prepare fine-tuning datasets
make prepare-sft-data
make prepare-dpo-data

# 4. Train
make pretrain
make sft
make dpo

# 5. Evaluate and export
make eval-dpo
make convert-hf
```

#### Interactive shells

```bash
# CPU container — for curation and data prep
make docker-shell-cpu

# GPU container — for training
make docker-shell-gpu
```

### Local (no Docker)

```bash
# 1. Install PyTorch first — version must match your CUDA driver
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Install NeMo stack first (sets dependency floor)
pip install "nemo_toolkit[core]==2.2.0" "nemo-aligner==0.7.0" "dask[distributed]==2024.4.1"
pip install "nemo-curator==0.7.1"

# 3. Install remaining dependencies
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

- `nemo_toolkit[core]` not `[all]` or `[nlp]` — both pull in `mamba-ssm==2.2.2` which requires `nvcc` to compile. The `runtime` base image does not ship `nvcc`.
- `huggingface-hub`, `transformers`, `pytorch-lightning`, `omegaconf`, `hydra-core`, `sentencepiece`, and `datasets` are not pinned directly — owned by `nemo_toolkit` and `nemo-curator`. Re-pinning causes resolution conflicts.
- Common Crawl WARC files are downloaded via `curl` over HTTPS (`data.commoncrawl.org`) rather than `aws s3 cp`. The `--no-sign-request` flag returns 403 when instance IAM credentials are present. `curl` bypasses this entirely.
- GPU-accelerated curator ops (`cudf`, `dask-cuda`) are excluded. They require the NVIDIA PyPI index and are not needed for the CPU curation pipeline.

### Curator Config

`curator/configs/curator.yaml` is tuned for a 32 vCPU / 128GB instance:

```yaml
dask:
  n_workers: 32
  threads_per_worker: 1
  memory_limit: "3GB"    # 32 × 3GB = 96GB
```

For smaller instances, scale `n_workers` to your vCPU count and set `memory_limit` to `(total_RAM_GB - 2) / n_workers`.

**Two-pass curation:** `quality_filter` and `tokenization` are disabled on the first curation pass because they depend on artifacts produced later (`quality_classifier.bin` and `slm_tokenizer.model`). Enable them after running `make tokenizer` and training a quality classifier.

### AWS

Data curation runs on CPU instances. Training runs on GPU instances. The `upload-data` and `setup-instance` targets handle the handoff via S3.

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
The official `nvcr.io/nvidia/nemo` image requires an NGC account and API token, which adds friction for open development and CI pipelines. The Dockerfile here reproduces the same stack from public sources only, with all transitive dependencies resolved and locked against the public PyPI index. Resolving this required upgrading `nemo_toolkit` to `2.2.0` (where `transformers>=4.48` support was added) and `nemo-aligner` to `0.7.0` (which dropped the `nemo_toolkit[nlp]` dependency that pulled in `mamba-ssm`).

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