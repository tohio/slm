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
| Data validation | `datatrove` + KenLM perplexity filtering |
| Tokenizer | HuggingFace `tokenizers` (BPE, 32k vocab) |
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
│   │   ├── code_search_net.py    CodeSearchNet Python via HuggingFace datasets
│   │   └── common_crawl.py       Common Crawl WARCs via HTTPS + trafilatura
│   ├── filters/
│   │   ├── quality.py            Heuristic quality filters + fasttext language detection
│   │   └── dedup.py              Exact + datatrove disk-based MinHash deduplication
│   └── scripts/
│       ├── curate.py             Main pipeline entry point
│       └── upload_s3.py          S3 upload/download utilities
│
├── validation/                   Stage 2: data validation
│   └── scripts/
│       ├── validate.py           Quality filter + perplexity filtering (KenLM)
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
- CUDA-capable GPU (for pretraining stages)
- AWS account (S3 for data storage)
- Weights & Biases account

**Installation**

On a fresh Ubuntu 22.04 cloud instance (recommended):
```bash
git clone https://github.com/tohio/slm.git /data/slm
cd /data/slm

cp .env.sample .env
vi .env   # fill in S3_BUCKET, AWS credentials, WANDB_API_KEY, HF_TOKEN

sudo apt install -y make

# Custom data dir — recommended when using a separate disk volume
make setup-data-dir DATA_DIR=/data/slm/data

# Default data dir (repo/data)
# make setup
```

Using pip:
```bash
make install          # creates .venv and installs all dependencies
make install-kenlm    # kenlm not on PyPI — curation instance only
```

Using uv:
```bash
make install-uv
make install-kenlm
```

Using conda:
```bash
make install-conda
make install-kenlm
```

GPU training instance only:
```bash
make install-gpu      # skips kenlm and other curation-only dependencies
```

---

**Run the full pipeline**

```bash
# ── Step 1: Curation instance (CPU) ──────────────────────────────────────────
make download-fasttext-model DATA_DIR=/data/slm/data   # language ID model (~1MB)
make download-kenlm-model    DATA_DIR=/data/slm/data   # perplexity model (~4GB)
make curate SIZE=125m WORKERS=24    # Stage 1: download, filter, dedup, blend, upload
make validate                       # Stage 2: perplexity filter
make validate-upload SIZE=125m      # Stage 2: push validated data to S3
make tokenizer                      # Stage 3: train BPE tokenizer
make tokenizer-upload               # Stage 3: push tokenizer to S3
make tokenize                       # Stage 4a: tokenize to binary
make tokenize-upload SIZE=125m      # Stage 4a: push tokenized binary to S3

# ── Step 2: GPU instance setup ───────────────────────────────────────────────
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m DATE=YYYY-MM-DD
source ~/.bashrc

# ── Step 3: Validate mini pipeline ───────────────────────────────────────────
# Exercises every stage end-to-end on a single GPU. Takes ~15–20 min.
make accelerate-config-single
make pretrain-mini  GPUS=1
make prepare-sft
make sft-mini       GPUS=1
make sft-code-mini  GPUS=1
make prepare-dpo
make dpo-mini       GPUS=1
make eval-mini

# ── Step 4: Full training ─────────────────────────────────────────────────────
# Before running, update gradient_accumulation_steps and max_steps in
# pretrain/configs/gpt_125m.yaml, aligment/configs/dpo_125m.yal and finetune/configs/sft_*_125m.yaml for your GPU count.
# See pretrain/README.md — Multi-GPU Config Scaling for exact values.
make accelerate-config-single.        # Use make accelerate-config-multi GPUS=x for multi gpu
make pretrain  SIZE=125m GPUS=1       # Stage 4b: pretrain from scratch
make export-base     SIZE=125m        # Stage 8:  push base model to Hub
make sft       SIZE=125m GPUS=1       # Stage 5b: chat SFT
make sft-code  SIZE=125m GPUS=1       # Stage 5c: code SFT
make export-instruct SIZE=125m        # Stage 8:  push instruct model to Hub
make dpo       SIZE=125m GPUS=1       # Stage 6b: DPO alignment
make eval      SIZE=125m              # Stage 7:  benchmark evaluation
make export-chat     SIZE=125m        # Stage 8:  push chat model to Hub
make serve                            # Stage 10: launch vLLM server
```

For full documentation of every `make` target see [docs/COMMANDS.md](docs/COMMANDS.md).

---

## Multi-GPU Config Scaling

> **Important:** All training configs — pretrain, SFT, and DPO — are written
> for **1 GPU**. Before running multi-GPU training at any stage, scale the
> config to preserve the global batch size and token budget:
>
> ```
> gradient_accumulation_steps = original / num_gpus
> max_steps                   = original / num_gpus   # pretrain only
> ```
>
> Each config file includes a comment with exact values for 1, 4, and 8 GPUs.
> This must be done for **every stage** — pretrain, SFT chat, SFT code, and DPO.

| Stage | Config location | Scaling fields |
|---|---|---|
| Pretrain | `pretrain/configs/gpt_{size}.yaml` | `gradient_accumulation_steps`, `max_steps` |
| SFT chat | `finetune/configs/sft_chat_{size}.yaml` | `gradient_accumulation_steps` |
| SFT code | `finetune/configs/sft_code_{size}.yaml` | `gradient_accumulation_steps` |
| DPO | `alignment/configs/dpo_{size}.yaml` | `gradient_accumulation_steps` |

Note: SFT and DPO use `epochs` not `max_steps` — only `gradient_accumulation_steps` needs adjusting for those stages.

```bash
make pretrain  SIZE=125m GPUS=8
make sft       SIZE=125m GPUS=8
make sft-code  SIZE=125m GPUS=8
make dpo       SIZE=125m GPUS=8

# Override config directly
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=4
```

---

## Data

### Source Mix

| Source | Mix | Tokens (1b) | Notes |
|---|---|---|---|
| Common Crawl | 55% | 16.5B | Broad web coverage, aggressively filtered |
| Wikipedia EN | 25% | 7.5B | High quality, factual, structured |
| CodeSearchNet | 20% | 6B | Python only |

### Token Targets

| Model | Total tokens | Epochs |
|---|---|---|
| `slm-125m` | 5B | 2 |
| `slm-350m` | 15B | 2 |
| `slm-1b` | 30B | 2 |

---

## Infrastructure

### Data Curation (CPU) — Stages 1–4a

Runs on CPU instances. No GPU required.

| Target | vCPUs | RAM | Est. runtime |
|---|---|---|---|
| mini (validation) | Any | 4GB+ | ~30–45 min |
| 125m | 16 vCPU | 32GB | ~8–12 hrs |
| 350m | 32 vCPU | 64GB | ~20–28 hrs |
| 1b | 64 vCPU | 256GB | ~30–40 hrs |

Run close to `us-east-1` (AWS) or `us-east1` (GCP) to minimise Common Crawl egress latency. Attach a persistent disk (500GB+) for `DATA_DIR` — the pipeline is fully resumable at every stage.

Use `tmux` to keep the pipeline running through session timeouts:
```bash
tmux new -s curate
make curate SIZE=125m WORKERS=16
# Ctrl+B, D to detach — tmux attach -t curate to reattach
```

### Training (GPU) — Stages 4b–6

Requires a CUDA-capable GPU instance. The pipeline uses **pure data parallelism** throughout all model sizes — no tensor parallelism or model parallelism is needed. The model is replicated on each GPU and the batch is split across GPUs.

> **Before running multi-GPU training at any stage:** Update
> `gradient_accumulation_steps` in the pretrain, SFT, and DPO configs for
> your GPU count. For pretrain, also update `max_steps`. Each config includes
> a scaling comment with exact values for 1, 4, and 8 GPUs.

Runtime varies significantly by GPU type and count. Use `make pretrain-mini GPUS=1` first to validate the training loop and measure your actual throughput before committing to a full run.

| Target | Min VRAM | Notes |
|---|---|---|
| mini (validation) | 8GB+ | Any GPU — confirms training loop works |
| 125m | 16GB+ per GPU | Fits on any modern data center GPU |
| 350m | 24GB+ per GPU | A100 40GB or better recommended |
| 1b | 40GB+ per GPU | A100 80GB / H100 / H200 recommended; gradient checkpointing enabled |

SFT and DPO runtimes are roughly 20–30% of pretraining time at the same model size. Use spot/preemptible instances — all training loops support `--resume` from the last checkpoint.

---

## Screenshots

| Screenshot | Stage | Description |
|---|---|---|
| `docs/screenshots/01_blend_stats.png` | Stage 1 | `blend_stats.json` showing 55/25/20 source mix |
| `docs/screenshots/02_validation_report.png` | Stage 2 | Validation report — total, kept, and rejection breakdown |
| `docs/screenshots/03_tokenizer_test.png` | Stage 3 | Tokenizer test output — special tokens and fertility score |
| `docs/screenshots/04_pretrain_loss.png` | Stage 4 | W&B pretraining loss curve |
| `docs/screenshots/05_sft_loss.png` | Stage 5 | W&B chat SFT loss curve |
| `docs/screenshots/06_dpo_loss.png` | Stage 6 | W&B DPO loss curve |
| `docs/screenshots/07_eval_results.png` | Stage 7 | Benchmark results — HellaSwag, ARC, MMLU, TruthfulQA, HumanEval |
| `docs/screenshots/08_hf_hub.png` | Stage 8 | HuggingFace Hub model page for `tohio/slm-125m` |
| `docs/screenshots/09_chat_session.png` | Stage 9 | Interactive multi-turn chat session via `inference/chat.py` |
| `docs/screenshots/10_vllm_curl.png` | Stage 10 | `curl` request to vLLM server with response |

### Stage 1 — Data Curation

Source mix breakdown from `blend_stats.json` — confirming the 55/25/20 Common Crawl / Wikipedia / code split.

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
| HumanEval | Python code generation |

---

## Key Design Decisions

**Why from scratch?** Starting from an existing checkpoint is the right production choice. We start from scratch deliberately — it exercises every stage of the pipeline and provides full visibility into how data quality and tokenizer design interact with training dynamics.

**Why a custom tokenizer?** A tokenizer trained on your specific data mix encodes domain patterns more efficiently. Special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|code|>`, `<|endofturn|>` and more) are baked in from the start with a Jinja2 chat template, giving the model a clean and consistent format across pretraining, SFT, DPO, and inference.

**Why GQA over MHA?** At inference time, KV cache is the primary memory bottleneck. GQA reduces KV heads from 12 to 4 (125m) — a 3× reduction in KV memory with negligible quality loss. Directly improves throughput in vLLM.

**Why DPO over PPO?** At small model scale, PPO's actor-critic setup requires multiple models simultaneously and is sensitive to reward scaling. DPO achieves comparable alignment with a simpler training loop and no separate reward model.

**Why sequential SFT (chat → code)?** Sequential fine-tuning produces independently evaluable checkpoints at each stage, making regressions immediately visible. The code SFT uses a lower learning rate to reduce catastrophic forgetting of chat capability.

**Why Python-only for code?** CodeSearchNet's Go and Rust corpora are thin, and including weak language coverage adds noise without meaningful benefit. Python has the strongest coverage, the highest quality docstrings, and the best downstream evaluation benchmarks (HumanEval). A focused 20% Python corpus outperforms a diluted multi-language mix at this scale.

**Why vLLM for serving?** PagedAttention enables continuous batching and efficient KV cache management. The OpenAI-compatible API means any client built against the OpenAI SDK works out of the box.

**Why datatrove for dedup instead of datasketch?** datasketch's `MinHashLSH` is in-memory — at 350m scale it requires ~32GB RAM. datatrove's disk-based pipeline uses a sort-based approach where RAM usage is bounded by shard size, not corpus size. Same approach used by FineWeb at trillion-token scale.

**Why HTTPS for Common Crawl instead of S3?** Direct S3 access to the `commoncrawl` bucket fails on EC2 instances with IAM roles attached — the instance role credentials are rejected by the bucket policy. HTTPS via `data.commoncrawl.org` works reliably regardless of instance credentials.

**Why fasttext for language detection?** Language detection runs on every Common Crawl document — tens of millions of pages. `langdetect` is pure Python and adds ~5–10ms per document. fasttext's `lid.176.ftz` model is C-backed, covers 176 languages, and runs ~1000× faster with equivalent accuracy.

---

## Production Serving

The `serve/manifests/` directory contains Kubernetes manifests deployed via [ai-infra](https://github.com/tohio/ai-infra) using ArgoCD. The vLLM server exposes an OpenAI-compatible REST API:

```bash
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
- **Training scale** — multi-node training with FSDP across 8+ nodes for the 1B model and beyond.
- **Continual learning** — a data flywheel feeding new curated data back into periodic pretraining runs.
- **Reward modelling** — a trained reward model enabling online DPO for more sophisticated alignment.
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