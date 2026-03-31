# slm

A decoder-only language model trained from scratch — raw web data through to an aligned, serving-ready model. Covers the full lifecycle: data curation, validation, tokenizer training, pretraining, supervised fine-tuning, preference alignment, evaluation, and production serving.

---

## Overview

Most LLM projects start from a pretrained checkpoint. This one doesn't. SLM is built entirely from scratch — from unstructured web crawl data to an instruction-following, chat-capable model deployed on Kubernetes.

The pipeline is modular and independently runnable at each stage. Every design decision is documented and justified.

**Models:** `tohio/slm-125m` · `tohio/slm-350m` · `tohio/slm-1b`

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
| Data curation | HuggingFace `datasets` + custom scripts |
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
│   ├── sources/                  Wikipedia, CodeSearchNet, Common Crawl
│   ├── filters/                  Quality heuristics + MinHash deduplication
│   └── scripts/                  curate.py pipeline + upload_s3.py
│
├── validation/                   Stage 2: data validation
│   └── scripts/validate.py       Quality filter + perplexity filtering
│
├── tokenizer/                    Stage 3: tokenizer training
│   ├── train_tokenizer.py        BPE tokenizer — 32k vocab, 16 special tokens
│   └── test_tokenizer.py         Roundtrip, fertility, chat template tests
│
├── pretrain/                     Stage 4: pretraining
│   ├── configs/                  gpt_125m.yaml, gpt_350m.yaml, gpt_1b.yaml
│   ├── data/
│   │   ├── tokenize.py           JSONL → uint16 memory-mapped binary
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
│   ├── architecture.svg          Pipeline architecture diagram
│   └── screenshots/              Pipeline stage screenshots
│
├── infra/
│   └── setup_gpu_instance.sh     GPU instance bootstrap — install deps, pull data from S3
│
├── Makefile                      Full pipeline automation
├── requirements.txt              Python dependencies
├── environment.yml               Conda environment
└── .env.sample                   Environment variable template
```

---

## Getting Started

**Prerequisites**
- Python 3.10+
- CUDA-capable GPU (H100 or A100 recommended for pretraining)
- AWS account (S3 for data storage)
- Weights & Biases account

**Installation**

Using pip:
```bash
git clone https://github.com/tohio/slm.git
cd slm
pip install -r requirements.txt
cp .env.sample .env
# Add your credentials to .env
```

Using uv:
```bash
git clone https://github.com/tohio/slm.git
cd slm
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.sample .env
# Add your credentials to .env
```

Using conda:
```bash
git clone https://github.com/tohio/slm.git
cd slm
conda create -n slm python=3.10 -y
conda activate slm
pip install -r requirements.txt
cp .env.sample .env
# Add your credentials to .env
```

**Run the full pipeline**

```bash
make curate          # Stage 1: download and curate data
make validate        # Stage 2: quality filter and validate
make tokenizer       # Stage 3: train tokenizer
make tokenize        # Stage 4a: tokenize dataset
make pretrain        # Stage 4b: pretrain slm-125m (single GPU)
make pretrain GPUS=4 # Stage 4b: pretrain on 4 GPUs
make sft             # Stage 5: chat SFT
make sft-code        # Stage 5: code SFT
make dpo             # Stage 6: DPO alignment
make eval            # Stage 7: evaluate on benchmarks
make export          # Stage 8: export to HuggingFace Hub
make serve           # Stage 10: launch vLLM server (Hub model)
make serve-local     # Stage 10: launch vLLM server (local checkpoint)
```

**Multi-GPU training**

```bash
# Specify GPU count with GPUS=N
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

## Key Design Decisions

**Why from scratch?** Starting from an existing checkpoint is the right production choice. We start from scratch deliberately — it exercises every stage of the pipeline and provides full visibility into how data quality and tokenizer design interact with training dynamics.

**Why a custom tokenizer?** A tokenizer trained on your specific data mix encodes domain patterns more efficiently. Special tokens (`<|user|>`, `<|assistant|>`, `<|code|>`, `<|endofturn|>`) are baked in from the start, giving the model a clean chat template without retrofitting.

**Why GQA over MHA?** At inference time, KV cache is the primary memory bottleneck. GQA reduces KV heads from 12 to 4 (125m) — a 3x reduction in KV memory with negligible quality loss. Directly improves throughput in vLLM.

**Why DPO over PPO?** At small model scale, PPO's actor-critic setup requires multiple models simultaneously and is sensitive to reward scaling. DPO achieves comparable alignment with a simpler training loop and no separate reward model.

**Why sequential SFT (chat → code)?** Sequential fine-tuning produces independently evaluable checkpoints at each stage, making regressions immediately visible. The code SFT uses a lower learning rate to reduce catastrophic forgetting of chat capability.

**Why vLLM for serving?** PagedAttention enables continuous batching and efficient KV cache management. The OpenAI-compatible API means any client built against the OpenAI SDK works out of the box — no custom client code.

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