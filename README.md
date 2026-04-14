# slm

A decoder-only language model trained from scratch — raw web data through to an aligned, serving-ready model. Covers the full lifecycle: data curation, validation, tokenizer training, pretraining, supervised fine-tuning, preference alignment, evaluation, and production serving.

---

## Overview

Most LLM projects start from a pretrained checkpoint. This one doesn't. SLM is built entirely from scratch — from unstructured web crawl data to an instruction-following, chat-capable model deployed on Kubernetes.

The pipeline is modular and independently runnable at each stage. Every design decision is documented and justified.

**Models:** `tohio/slm-125m` · `tohio/slm-125m-instruct` · `tohio/slm-125m-chat` · `tohio/slm-350m` · `tohio/slm-350m-instruct` · `tohio/slm-350m-chat` · `tohio/slm-1b` · `tohio/slm-1b-instruct` · `tohio/slm-1b-chat`

![Architecture](docs/architecture.svg)

---

## Architecture

The model is a dense decoder-only transformer with a modern architecture:

| Component | Choice | Rationale |
|---|---|---|
| Positional encoding | RoPE | Better length generalisation, relative position awareness |
| Normalization | RMSNorm | Faster than LayerNorm, modern standard |
| Activation | SwiGLU | Better gradient flow, used by LLaMA, Mistral, Qwen |
| Attention | GQA | Reduces KV memory overhead at inference |
| Bias | None | Simpler, modern standard |
| Embeddings | Tied | Reduces parameters, effective at small scale |

**Model sizes:**

| Model | Layers | Hidden | Q heads | KV heads | Context |
|---|---|---|---|---|---|
| `slm-125m` | 12 | 768 | 12 | 4 | 2048 |
| `slm-350m` | 24 | 1024 | 16 | 8 | 2048 |
| `slm-1b` | 32 | 2048 | 32 | 8 | 4096 |

---

## Tech Stack

| Stage | Tool |
|---|---|
| Data curation | HuggingFace `datasets` + `datatrove` + custom scripts |
| Data validation | `datatrove` |
| Tokenizer | HuggingFace `tokenizers` (BPE) |
| Pretraining | HuggingFace `accelerate` + `transformers` |
| Experiment tracking | Weights & Biases |
| SFT | HuggingFace `trl` (`SFTTrainer`) |
| DPO | HuggingFace `trl` (`DPOTrainer`) |
| Evaluation | `lm-evaluation-harness` |
| Export | HuggingFace `transformers` |
| Inference | HuggingFace `transformers` |
| Serving | `vLLM` on Kubernetes via `ai-infra` |

---

## Repo Structure

```
slm/
├── model/                        Custom decoder-only transformer architecture
│   ├── config.py                 SLMConfig — hyperparameters for 125M/350M/1B
│   ├── attention.py              Grouped Query Attention + RoPE
│   ├── mlp.py                    SwiGLU feed-forward network
│   ├── norm.py                   RMSNorm
│   ├── block.py                  Pre-norm transformer block
│   └── model.py                  SLMModel + SLMForCausalLM
│
├── curator/                      Stage 1: data curation
│   ├── sources/
│   │   ├── wikipedia.py          Wikipedia EN via HuggingFace datasets
│   │   ├── code_search_net.py    CodeSearchNet via HuggingFace datasets
│   │   └── common_crawl.py       Common Crawl WARCs via HTTPS + trafilatura
│   ├── filters/
│   │   ├── quality.py            Heuristic quality filters (FineWeb/Gopher-style)
│   │   └── dedup.py              Exact + datatrove disk-based MinHash deduplication
│   └── scripts/
│       ├── curate.py             Main pipeline entry point
│       └── upload_s3.py          S3 upload/download utilities
│
├── validation/                   Stage 2: data validation
│   └── scripts/
│       ├── validate.py           Quality filter + perplexity filtering
│       └── upload_validated.py   Upload validated data to S3 (versioned by target + date)
│
├── tokenizer/                    Stage 3: tokenizer training
│   ├── train_tokenizer.py        BPE tokenizer — 32k vocab, 16 special tokens
│   └── test_tokenizer.py         Roundtrip, fertility, chat template tests
│
├── pretrain/                     Stage 4: pretraining
│   ├── configs/                  gpt_mini.yaml, gpt_125m.yaml, gpt_350m.yaml, gpt_1b.yaml
│   ├── data/
│   │   ├── tokenize_data.py      JSONL → uint16 memory-mapped binary
│   │   ├── upload_tokenized.py   Upload/download tokenized binary to/from S3
│   │   └── dataset.py            PretrainingDataset wrapping .bin file
│   └── train.py                  Pretraining loop
│
├── finetune/                     Stage 5: supervised fine-tuning
│   ├── configs/                  sft_chat/code × 125m/350m/1b (6 configs)
│   ├── data/prepare_sft.py       Chat + code dataset preparation
│   └── train_sft.py              SFT training loop
│
├── alignment/                    Stage 6: preference alignment
│   ├── configs/                  dpo_125m.yaml, dpo_350m.yaml, dpo_1b.yaml
│   ├── data/prepare_dpo.py       Preference dataset blending
│   └── train_dpo.py              DPO training loop
│
├── eval/                         Stage 7: benchmark evaluation
│   └── eval.py                   HellaSwag, ARC, MMLU, TruthfulQA, HumanEval
│
├── export/                       Stage 8: model export
│   └── export.py                 Hub push + model card generation
│
├── inference/                    Stage 9: local inference
│   ├── chat.py                   Interactive multi-turn chat CLI
│   └── generate.py               Batch text generation
│
├── serve/                        Stage 10: production serving
│   ├── manifests/                Kubernetes deployment, service, HPA
│   └── serve.sh                  Local server launch script
│
├── notebooks/                    Exploratory analysis — one per pipeline stage
│   ├── 01_model_exploration.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_validation_exploration.ipynb
│   ├── 04_tokenizer_exploration.ipynb
│   ├── 05_pretrain_exploration.ipynb
│   ├── 06_sft_exploration.ipynb
│   ├── 07_dpo_exploration.ipynb
│   ├── 08_eval_exploration.ipynb
│   └── 09_inference_exploration.ipynb
│
├── docs/
│   ├── COMMANDS.md               Full make target reference
│   ├── architecture.svg          Pipeline architecture diagram
│   └── screenshots/              Pipeline stage screenshots
│       ├── 01_blend_stats.png    Stage 1 — source mix breakdown
│       ├── 02_validation_report.png  Stage 2 — validation report
│       ├── 03_tokenizer_test.png Stage 3 — special tokens and fertility
│       ├── 04_pretrain_loss.png  Stage 4b — W&B pretraining loss curve
│       ├── 05_sft_loss.png       Stage 5 — W&B SFT loss curve
│       ├── 06_dpo_loss.png       Stage 6 — W&B DPO loss curve
│       ├── 07_eval_results.png   Stage 7 — benchmark results
│       ├── 08_hf_hub.png         Stage 8 — HuggingFace Hub model page
│       ├── 09_chat_session.png   Stage 9 — multi-turn chat session
│       └── 10_vllm_curl.png      Stage 10 — vLLM API response
│
├── infra/
│   ├── setup.sh                  CPU instance bootstrap — curation environment
│   └── setup_gpu_instance.sh     GPU instance bootstrap — safe to re-run after preemptible restart
│
├── accelerate_configs/           Accelerate launch configs for GPU training
│   ├── single_gpu.yaml           Single GPU — mini validation runs
│   └── multi_gpu.yaml            Multi-GPU — full pretraining and fine-tuning
├── Makefile                      Full pipeline automation
├── requirements.txt              Python dependencies
├── environment.yml               Conda environment
└── .env.sample                   Environment variable template
```

---

## Getting Started

**Prerequisites**
- Python 3.12+
- CUDA-capable GPU (H100 or A100 recommended for pretraining)
- AWS account (S3 for data storage)
- Weights & Biases account

**Installation**

On a fresh Ubuntu 22.04 cloud instance (recommended):
```bash
git clone https://github.com/tohio/slm.git /data/slm
cd /data/slm

# Populate credentials before running setup
cp .env.sample .env

# fill in S3_BUCKET, AWS credentials, WANDB_API_KEY, HF_TOKEN
vi .env

# make is the only manual prerequisite — setup handles everything else
sudo apt install -y make

# Custom data dir — recommended when using a separate EBS volume
make setup-data-dir DATA_DIR=/data/slm/data

# Default data dir (repo/data) — use if no separate EBS volume
# make setup
```

Using pip (recommended — creates `.venv` automatically):
```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
# Add your credentials to .env
make install          # creates .venv and installs all dependencies
make install-kenlm    # kenlm not on PyPI — install from source (curation instance only)
```

Using uv:
```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
# Add your credentials to .env
make install-uv
make install-kenlm
```

Using conda:
```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
# Add your credentials to .env
make install-conda
make install-kenlm
```

**GPU training instance only** — fasttext and kenlm not needed:
```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
# Add your credentials to .env
make install-gpu
```

Then fill in credentials in `.env`:
```
S3_BUCKET=your-bucket
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
```

**Validate the pipeline with a mini run (~30 min)**

Before committing to a full run, validate the pipeline end to end:

```bash
make download-fasttext-model        # one-time: download fasttext language ID model (~1MB)
make curate-mini
```

This caps Wikipedia at 5k docs, CodeSearchNet at 10k samples, and Common
Crawl at 2 WARC segments. Exercises every stage without the wait.

**Run the full pipeline**

```bash
# ── One-time setup ────────────────────────────────────────────────────────────
make download-fasttext-model DATA_DIR=/data/slm/data   # fasttext language ID model (~1MB)
make download-kenlm-model    DATA_DIR=/data/slm/data   # KenLM perplexity model (~4GB)
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m DATE=2026-04-12  # GPU instance setup
make pretrain-mini GPUS=1                           # validate training loop
make prepare-sft                                    # download SFT datasets
make sft-mini GPUS=1                                # validate SFT
make sft-code-mini GPUS=1                           # validate code SFT
make prepare-dpo                                    # download DPO datasets
make dpo-mini GPUS=1                                # validate DPO

# ── Data ──────────────────────────────────────────────────────────────────────
make curate SIZE=125m WORKERS=16    # Stage 1: download, curate, upload to S3
make validate                       # Stage 2: perplexity filter
make validate-upload SIZE=125m      # Stage 2: push validated data to S3

# ── Tokenizer ─────────────────────────────────────────────────────────────────
make tokenizer                      # Stage 3: train BPE tokenizer
make tokenizer-upload               # Stage 3: push tokenizer to S3
make tokenize                       # Stage 4a: tokenize to binary
make tokenize-upload SIZE=125m      # Stage 4a: push tokenized binary to S3

# ── Training ──────────────────────────────────────────────────────────────────
make pretrain-mini GPUS=1            # Stage 4b: validate training loop (~5-10 min)
make pretrain GPUS=4                # Stage 4b: pretrain slm-125m
make prepare-sft                    # Stage 5a: download SFT datasets
make sft      GPUS=4                # Stage 5b: chat SFT
make sft-code GPUS=4                # Stage 5c: code SFT
make prepare-dpo                    # Stage 6a: download DPO datasets
make dpo      GPUS=2                # Stage 6b: DPO alignment

# ── Ship ──────────────────────────────────────────────────────────────────────
make eval                           # Stage 7: evaluate on benchmarks
make export-base SIZE=125m          # Stage 8: push base model
make export-instruct SIZE=125m      # Stage 8: push instruct model
make export-chat SIZE=125m          # Stage 8: push chat model
make serve                          # Stage 10: launch vLLM server
```

For full documentation of every `make` target see [docs/COMMANDS.md](docs/COMMANDS.md).

**Run curation sub-stages individually**

```bash
make curate-download SIZE=125m
make curate-filter   SIZE=125m
make curate-dedup    SIZE=125m WORKERS=16
make curate-blend    SIZE=125m
make curate-upload   SIZE=125m
```

**Multi-GPU training**

```bash
make pretrain SIZE=125m GPUS=4
make pretrain SIZE=350m GPUS=6
make pretrain SIZE=1b   GPUS=8

# Override config directly
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=4
make sft      CONFIG=finetune/configs/sft_chat_125m.yaml GPUS=4
make dpo      CONFIG=alignment/configs/dpo_125m.yaml GPUS=2
```

**Interactive chat**

```bash
python inference/chat.py --model tohio/slm-125m
```

---

## Infrastructure

### Data Curation (CPU)

Stages 1–4a (download, filter, dedup, blend, validate, tokenize) run on CPU instances. No GPU required.

| Target | vCPUs | RAM | Est. runtime |
|---|---|---|---|
| mini (validation) | Any | 4GB+ | ~30–45 min |
| 125m | 16 vCPU | 32GB | ~22–26 hrs (curation ~12–16 hrs + tokenize ~9 hrs) |
| 350m | 16 vCPU | 32GB | ~55–65 hrs |
| 1b | 32 vCPU | 64GB | ~110–130 hrs |

Any cloud provider works for curation. Run close to `us-east-1` (AWS) or `us-east1` (GCP) to minimise Common Crawl egress latency. Attach a persistent disk (500GB) for `DATA_DIR` so data survives spot/preemptible interruptions — the pipeline is fully resumable at every stage.

Use `tmux` to keep the pipeline running through session timeouts:
```bash
tmux new -s curate
make curate SIZE=125m WORKERS=16    # .venv/bin/python used automatically
# Ctrl+B, D to detach — tmux attach -t curate to reattach
```

### Training (GPU)

Stages 4b–6 (pretrain, SFT, DPO) require a CUDA-capable GPU instance. Any cloud provider with H100 or A100 instances works. Providers like [Nebius](https://nebius.com), [Lambda Labs](https://lambdalabs.com), and [CoreWeave](https://coreweave.com) typically offer better GPU pricing than hyperscalers.

Before committing to a full pretraining run, validate the training loop with `make pretrain-mini` — it uses a tiny 6-layer model and runs to 500 steps in minutes on a single GPU.

| Target | GPUs | VRAM | Est. pretrain runtime |
|---|---|---|---|
| mini (validation) | 1× any GPU | 8GB+ | ~5–10 min |
| 125m | 4× H100 or A100 | 320GB+ | ~12–18 hrs |
| 350m | 8× H100 or A100 | 640GB+ | ~24–36 hrs |
| 1b | 8× H100 or A100 | 640GB+ | ~72–96 hrs |

SFT and DPO runtimes are roughly 20–30% of pretraining time at the same model size. Use spot/preemptible instances where possible — all training loops support `--resume` from the last checkpoint.

Run `make accelerate-config` once on the GPU instance before training to configure multi-GPU settings. Pull the tokenized binary with `make tokenize-download` to avoid re-tokenizing on expensive GPU hardware.

---

## Screenshots

Captured at each pipeline stage as proof of a working end-to-end run.

| Screenshot | Stage | Description |
|---|---|---|
| `docs/screenshots/01_blend_stats.png` | Stage 1 | `blend_stats.json` showing 70/20/10 source mix |
| `docs/screenshots/02_validation_report.png` | Stage 2 | Validation report — total, kept, and rejection breakdown |
| `docs/screenshots/03_tokenizer_test.png` | Stage 3 | Tokenizer test output — special tokens table and fertility score |
| `docs/screenshots/04_pretrain_loss.png` | Stage 4 | W&B pretraining loss curve |
| `docs/screenshots/05_sft_loss.png` | Stage 5 | W&B chat SFT loss curve |
| `docs/screenshots/06_dpo_loss.png` | Stage 6 | W&B DPO loss curve |
| `docs/screenshots/07_eval_results.png` | Stage 7 | Benchmark results — HellaSwag, ARC, MMLU, TruthfulQA, HumanEval |
| `docs/screenshots/08_hf_hub.png` | Stage 8 | HuggingFace Hub model page for `tohio/slm-125m` |
| `docs/screenshots/09_chat_session.png` | Stage 9 | Interactive multi-turn chat session via `inference/chat.py` |
| `docs/screenshots/10_vllm_curl.png` | Stage 10 | `curl` request to vLLM server with response |

### Stage 1 — Data Curation

Source mix breakdown from `blend_stats.json` — confirming the 70/20/10 Wikipedia / Common Crawl / code split.

![Blend stats](docs/screenshots/01_blend_stats.png)

### Stage 4 — Pretraining

W&B loss curve showing steady convergence over the full pretraining run.

![Pretraining loss](docs/screenshots/04_pretrain_loss.png)

### Stage 7 — Evaluation

Benchmark results across HellaSwag, ARC, MMLU, TruthfulQA, and HumanEval.

![Eval results](docs/screenshots/07_eval_results.png)

### Stage 9 — Inference

Multi-turn chat session via `inference/chat.py` using the aligned chat model.

![Chat session](docs/screenshots/09_chat_session.png)

---

## Evaluation

Models are evaluated on standard benchmarks via `lm-evaluation-harness`:

| Benchmark | Measures |
|---|---|
| HellaSwag | Commonsense reasoning |
| ARC-Easy / ARC-Challenge | Science QA |
| MMLU | Broad knowledge |
| TruthfulQA | Factual accuracy |
| HumanEval | Code generation |

---

## Key Design Decisions

**Why from scratch?** Starting from an existing checkpoint is the right production choice. We start from scratch deliberately — it exercises every stage of the pipeline and provides full visibility into how data quality and tokenizer design interact with training dynamics.

**Why a custom tokenizer?** A tokenizer trained on your specific data mix encodes domain patterns more efficiently. Special tokens (`<|user|>`, `<|assistant|>`, `<|code|>`, `<|endofturn|>`) are baked in from the start, giving the model a clean chat template without retrofitting.

**Why GQA over MHA?** At inference time, KV cache is the primary memory bottleneck. GQA reduces KV heads from 12 to 4 (125m) — a 3x reduction in KV memory with negligible quality loss. Directly improves throughput in vLLM.

**Why DPO over PPO?** At small model scale, PPO's actor-critic setup requires multiple models simultaneously and is sensitive to reward scaling. DPO achieves comparable alignment with a simpler training loop and no separate reward model.

**Why sequential SFT (chat → code)?** Sequential fine-tuning produces independently evaluable checkpoints at each stage, making regressions immediately visible. The code SFT uses a lower learning rate to reduce catastrophic forgetting of chat capability.

**Why vLLM for serving?** PagedAttention enables continuous batching and efficient KV cache management. The OpenAI-compatible API means any client built against the OpenAI SDK works out of the box — no custom client code.

**Why datatrove for dedup instead of datasketch?** datasketch's `MinHashLSH` is in-memory — at 350m scale it requires ~32GB RAM, at 1b it requires ~85GB and cannot fit on a single instance. datatrove's disk-based pipeline uses a sort-based approach where RAM usage is bounded by shard size, not corpus size. Same approach used by FineWeb at trillion-token scale.

**Why HTTPS for Common Crawl instead of S3?** Direct S3 access to the `commoncrawl` bucket fails on EC2 instances with IAM roles attached — the instance role credentials are rejected by the bucket policy. HTTPS via `data.commoncrawl.org` works reliably regardless of instance credentials.

**Why fasttext for language detection instead of langdetect?** Language detection runs on every document in the Common Crawl pipeline — at 125m scale that is tens of millions of HTML pages. `langdetect` is pure Python and adds ~5–10ms per document, which compounds to 50–100+ hours of wall time on a single instance. fasttext's language identification model (`lid.176.ftz`) is C-backed, covers 176 languages, and runs ~1000x faster with equivalent accuracy. The model file is ~1MB and is downloaded once via `make download-fasttext-model`.

---

## Production Serving

The `serve/manifests/` directory contains Kubernetes manifests deployed via [ai-infra](https://github.com/tohio/ai-infra) using ArgoCD. The vLLM server exposes an OpenAI-compatible REST API and a Prometheus `/metrics` endpoint scraped by the cluster monitoring stack.

```bash
# Query the model via OpenAI-compatible API
curl http://slm-service:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slm-125m",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Production Considerations

This project is scoped as a complete end-to-end training pipeline and demonstration. In a larger production system:

- **Data scale** — the curation pipeline would run on a distributed compute cluster over petabyte-scale crawl data rather than a single CPU instance.
- **Training scale** — multi-node training with FSDP or tensor parallelism across 8+ GPUs for the 1B model.
- **Continual learning** — a data flywheel feeding new curated data back into periodic pretraining runs.
- **Reward modelling** — a trained reward model enabling PPO or online DPO for more sophisticated alignment.
- **Observability** — per-request latency, token throughput, and generation quality metrics surfaced in Grafana.

---

## Related Projects

- [ai-infra](https://github.com/tohio/ai-infra) — Kubernetes platform that deploys and operates this model in production
- [rag-pipeline](https://github.com/tohio/rag-pipeline) — RAG pipeline that can use slm as the base LLM
- [multi-agent](https://github.com/tohio/multi-agent) — autonomous multi-agent investment research
- [data-flywheel](https://github.com/tohio/data-flywheel) — self-improving data pipeline feeding into future SLM training runs

---

## License

MIT