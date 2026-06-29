# slm

End-to-end decoder-only language model pipeline: curation, validation, tokenizer training, pretraining, SFT, DPO, evaluation, export, inference, and serving.

Models:
- `tohio/slm-125m`
- `tohio/slm-125m-instruct`
- `tohio/slm-125m-chat`
- `tohio/slm-125m-code`

The same pipeline structure supports `125m`, `350m`, and `1b` runs.

![Architecture](docs/architecture.png)

---

## Choosing a size

All model sizes use the same pipeline. Choose based on available compute.

| Size | Use case |
|---|---|
| `125m` | Pipeline validation, debugging, and low-cost experiments |
| `350m` | Higher-capacity research runs |
| `1b` | Larger small-model runs when more compute is available |

---

## Architecture

Dense decoder-only Transformer.

| Component | Choice |
|---|---|
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Attention | GQA |
| Bias | None |
| Embeddings | Tied |

| Model | Layers | Hidden | Q heads | KV heads | Context |
|---|---:|---:|---:|---:|---:|
| `slm-125m` | 12 | 768 | 12 | 4 | 2048 |
| `slm-350m` | 24 | 1024 | 16 | 8 | 2048 |
| `slm-1b` | 32 | 2048 | 32 | 8 | 4096 |

---

## Tech Stack

| Stage | Tool |
|---|---|
| Data curation | Hugging Face Datasets, Datatrove |
| Data validation | Datatrove, KenLM |
| Tokenizer | Hugging Face Tokenizers |
| Pretraining | Accelerate, Transformers |
| Experiment tracking | Weights & Biases |
| SFT | TRL |
| DPO | TRL |
| Evaluation | lm-evaluation-harness |
| Export | Transformers |
| Inference | Transformers |
| Serving | vLLM |

---

## Repo Structure

```text
slm/
├── config/
│   └── data_mix.py
├── config_gen/
│   ├── config_gen.py
│   └── accel_gen.py
├── model/
│   ├── config.py
│   ├── attention.py
│   ├── mlp.py
│   ├── norm.py
│   ├── block.py
│   └── model.py
├── curator/
│   ├── constants.py
│   ├── sources/
│   ├── filters/
│   └── scripts/
│       ├── curate.py
│       ├── sample_source.py
│       └── upload_s3.py
├── validation/
│   └── scripts/
│       ├── validate.py
│       └── upload_validated.py
├── tokenizer/
│   ├── train_tokenizer.py
│   └── test_tokenizer.py
├── pretrain/
│   ├── configs/
│   ├── data/
│   └── train.py
├── finetune/
│   ├── configs/
│   ├── data/
│   │   ├── prepare_sft.py
│   │   └── response_control.py
│   └── train_sft.py
├── alignment/
│   ├── configs/
│   ├── data/
│   │   └── prepare_dpo.py
│   └── train_dpo.py
├── eval/
│   ├── eval.py
│   ├── sanity_eval.py
│   └── sanity_prompts.jsonl
├── export/
│   └── export.py
├── inference/
│   ├── utils.py
│   ├── chat.py
│   └── generate.py
├── serve/
│   ├── manifests/
│   └── serve.sh
├── scripts/
├── tests/
├── docs/
├── infra/
│   ├── setup.sh
│   └── setup_gpu_instance.sh
├── accelerate_configs/
├── Makefile
├── requirements.txt
├── environment.yml
└── .env.sample
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Ubuntu 24.04 recommended
- AWS account and S3 bucket for data artifacts
- Weights & Biases account
- Hugging Face account and token
- CUDA-capable GPU for training stages

Before the first curation run, accept the terms for gated datasets used by the active mix:
- `bigcode/the-stack-dedup`
- `bigcode/the-stack-smol`
- `nvidia/Nemotron-CC-Math-v1`
- `nvidia/Nemotron-Pretraining-Specialized-v1.1`

### Install

```bash
git clone https://github.com/tohio/slm.git /data/slm
cd /data/slm

cp .env.sample .env
vi .env

sudo apt install -y make

make setup-data-dir DATA_DIR=/data/slm/data
make install
```

Alternative install paths:

```bash
make install-uv
make install-conda
```

CPU curation prerequisites:

```bash
make install-kenlm
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model    DATA_DIR=/data/slm/data
```

GPU training instance only:

```bash
make install-gpu
```

---

## Reusable curation artifacts

Curation artifacts are grouped by `RUN_ID`, not calendar date. A single logical run should use one `RUN_ID` across upload, download, and GPU setup.

Generated run IDs use this format:

```text
{SIZE}-{YYYYMMDD}-{random_hex}
```

Example:

```text
125m-20260629-a8f3c9
```

`artifacts-upload` resolves the run ID as follows:

```text
provided RUN_ID wins
else today's local RUN_ID file wins
else create a new RUN_ID file for today
```

The local run ID file is written here:

```text
data/runs/<size>/RUN_ID
```

Run layout:

```text
data/runs/<size>/raw/<source>/
data/runs/<size>/curated/
data/runs/<size>/validated/
data/runs/<size>/tokenized/
data/runs/<size>/tokenizer/
data/runs/<size>/metadata/
```

S3 layout:

```text
<S3_PREFIX>/<size>/<run_id>/raw/
<S3_PREFIX>/<size>/<run_id>/curated/
<S3_PREFIX>/<size>/<run_id>/validated/
<S3_PREFIX>/<size>/<run_id>/tokenized/
<S3_PREFIX>/<size>/<run_id>/tokenizer/
<S3_PREFIX>/<size>/<run_id>/metadata/
```

Normal upload:

```bash
make artifacts-upload SIZE=125m
```

Explicit restore/reuse:

```bash
make artifacts-download SIZE=125m RUN_ID=125m-20260629-a8f3c9
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m RUN_ID=125m-20260629-a8f3c9
```

Valid `ARTIFACT_STAGES` values:

```text
raw, curated, validated, tokenized, tokenizer, metadata
```

---

## Pipeline Commands

### CPU prerequisites

```bash
make setup-data-dir DATA_DIR=/data/slm/data
make install-kenlm
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model    DATA_DIR=/data/slm/data
```

### Mini curation

```bash
make curate-mini
make test-curator

make validate SIZE=mini
make test-validate SIZE=mini

make tokenizer SIZE=mini
make tokenize SIZE=mini
make test-tokenizer SIZE=mini

make artifacts-upload SIZE=mini ARTIFACT_STAGES="tokenized,tokenizer,metadata"
```

### Full curation: 125m

```bash
make curate SIZE=125m WORKERS=62
make validate SIZE=125m
make tokenizer SIZE=125m
make tokenize SIZE=125m
make artifacts-upload SIZE=125m
```

### GPU prerequisites

```bash
make setup-gpu DATA_DIR=/data/slm/data SIZE=mini RUN_ID=<mini-run-id>
source ~/.bashrc
make accelerate-config-single
```

### Mini training

```bash
make pretrain-mini SIZE=mini GPUS=1
make test-training SIZE=mini

make reinit-embeds SIZE=mini

make prepare-sft SIZE=mini
make sft-instruct-mini SIZE=mini GPUS=1
make test-sft-instruct SIZE=mini

make sft-code-mini SIZE=mini GPUS=1
make test-sft-code SIZE=mini

make prepare-dpo SIZE=mini
make dpo-chat-mini SIZE=mini GPUS=1
make test-dpo-chat SIZE=mini

make eval-mini SIZE=mini
```

### Full training: 125m base

```bash
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m RUN_ID=<125m-run-id>
source ~/.bashrc

make accelerate-config-single
make config-gen SIZE=125m GPUS=1

make pretrain SIZE=125m GPUS=1
make reinit-embeds SIZE=125m

make eval-base SIZE=125m
make export-base SIZE=125m
```

### Post-training: instruct and chat branch

```bash
make prepare-sft SIZE=125m
make sft-instruct SIZE=125m GPUS=1

make eval-instruct SIZE=125m
make export-instruct SIZE=125m

make prepare-dpo SIZE=125m
make dpo-chat SIZE=125m GPUS=1

make eval-chat SIZE=125m
make eval-sanity-chat SIZE=125m
make export-chat SIZE=125m
```

### Post-training: code branch

```bash
make sft-code SIZE=125m GPUS=1

make eval-code SIZE=125m
make eval-sanity-code SIZE=125m
make export-code SIZE=125m
```

For full documentation of every `make` target see [docs/COMMANDS.md](docs/COMMANDS.md).

---

## Checkpoint Lineage

```text
pretrain/final
  ↓
sft_instruct/final
  ├── dpo_chat/final
  └── sft_code/final
```

Model variants:

| Variant | Path | Hub model |
|---|---|---|
| Base | `results/runs/{size}/pretrain/final` | `tohio/slm-{size}` |
| Instruct | `results/runs/{size}/sft_instruct/final` | `tohio/slm-{size}-instruct` |
| Chat | `results/runs/{size}/dpo_chat/final` | `tohio/slm-{size}-chat` |
| Code | `results/runs/{size}/sft_code/final` | `tohio/slm-{size}-code` |

---

## Tests

### CPU pipeline tests

```bash
make curate-mini   && make test-curator
make validate      && make test-validate
make tokenize      && make test-tokenizer

make test-data-pipeline
```

### GPU pipeline tests

GPU pipeline test targets default to `mini`. Pass `SIZE=125m`, `350m`, or `1b` only when validating full-run artifacts.

```bash
make pretrain-mini       GPUS=1 && make test-training
make sft-instruct-mini   GPUS=1 && make test-sft-instruct
make sft-code-mini       GPUS=1 && make test-sft-code
make dpo-chat-mini       GPUS=1 && make test-dpo-chat

make test-gpu-pipeline
```

Full-run artifact checks:

```bash
make test-training     SIZE=125m
make test-sft-instruct SIZE=125m
make test-sft-code     SIZE=125m
make test-dpo-chat     SIZE=125m
```

### Unit tests

```bash
make test-model
make test-config-gen
make test-accel-gen
make test-unit
```

| Target | Validates |
|---|---|
| `test-curator` | curation outputs |
| `test-validate` | validation outputs |
| `test-tokenizer` | tokenizer and tokenized binary outputs |
| `test-training` | pretraining outputs |
| `test-sft-instruct` | instruct SFT outputs |
| `test-sft-code` | code SFT outputs |
| `test-dpo-chat` | chat DPO outputs |
| `test-model` | model architecture |
| `test-config-gen` | training config generation |
| `test-accel-gen` | accelerate config generation |

---

## Multi-GPU Config Scaling

Generate configs before training:

```bash
make config-gen-pretrain SIZE=125m GPUS=8
make config-gen-sft      SIZE=125m GPUS=8
make config-gen-dpo      SIZE=125m GPUS=8
make config-gen          SIZE=125m GPUS=8
```

Choose an accelerate config:

```bash
make accelerate-config-single
make accelerate-config-multi GPUS=8
```

For FSDP:

```bash
make accel-gen-fsdp GPUS=8
```

Use the same `GPUS` value for accelerate setup, config generation, and training:

```bash
make accelerate-config-multi GPUS=4
make config-gen SIZE=125m GPUS=4
make pretrain   SIZE=125m GPUS=4
```

---

## Data

### Source Mix

The source mix is defined in `config/data_mix.py`. It is the source of truth for curation, export, and notebooks.

| Source | Target Share | Notes |
|---|---:|---|
| Common Crawl | 5% | direct WARC via trafilatura |
| FineWeb | 10% | broad web text |
| FineWeb-Edu | 31.5% | educational/explanatory web text |
| Wikipedia | 10% | encyclopedia text |
| pg19 | 2.5% | public-domain books |
| peS2o | 5% | academic/scientific prose |
| Nemotron CC Math | 7% | math/STEM text |
| StackExchange | 1% | Q&A-style web text |
| Synthetic arithmetic | 0.1475% | arithmetic signal |
| Synthetic task code | 0.3934% | task-shaped code examples |
| Educational QA/MCQ (math) | 0.1475% | math MCQ examples |
| Educational QA/MCQ (general) | 0.2459% | general MCQ examples |
| Factual restraint | 0.0657% | uncertainty/restraint examples |
| Nemotron Specialized | 12% | specialized supplement |
| Code (total) | 15% | split across code sub-sources |

### Run-specific realized mix

Completed curation runs write realized mix metadata to:

```text
data/runs/<size>/curated/blend_stats.json
data/runs/<size>/metadata/blend_stats.json
```

Use `blend_stats.json` as the source of truth for completed runs and exported model cards.

### Token Targets

| Model | Curation target | Epochs | Consumed target |
|---|---:|---:|---:|
| `slm-125m` | 10B | 2 | 20B |
| `slm-350m` | 25B | 2 | 50B |
| `slm-1b` | 75B | 1 | 75B |

### Train / val split

The train and val splits are produced by the curator blend stage. Validation and tokenization process each split independently, so `train.bin` and `val.bin` receive the same quality filters.

---

## Infrastructure

The model may fit on smaller GPUs, but full runs can become impractically slow. The recommendations below are for practical end-to-end runs, not minimum loadability.

### Data curation

| Target | Recommended vCPUs | Recommended RAM | Notes |
|---|---:|---:|---|
| `mini` | 4+ | 16 GB+ | pipeline validation |
| `125m` | 64+ | 256–384 GB | recommended full 125m curation |
| `350m` | 64–96+ | 384 GB+ | larger curation run |
| `1b` | 96+ | 512 GB+ | largest supported run |

Use a persistent disk for `DATA_DIR`. Run long jobs inside `tmux`.

### Training

| Target | Practical GPU recommendation | Notes |
|---|---|---|
| `mini` | 1× 16 GB+ GPU | training-loop validation only |
| `125m` | 1× A100 80GB / H100 / H200, or better | practical full 125m run |
| `350m` | 1–4× A100 80GB / H100 / H200, or better | use multi-GPU if available |
| `1b` | 4–8× A100 80GB / H100 / H200, or better | prefer FSDP/multi-GPU |

---

## Screenshots

Example pipeline outputs and run screenshots are available in `docs/screenshots/`.

---

## Evaluation

```bash
make eval-base     SIZE=125m
make eval-instruct SIZE=125m
make eval-chat     SIZE=125m
make eval-code     SIZE=125m
make eval-sanity   SIZE=125m
```

| Benchmark | Measures |
|---|---|
| HellaSwag | Commonsense reasoning |
| ARC-Easy / ARC-Challenge | Science QA |
| MMLU | Broad knowledge |
| TruthfulQA | Factual accuracy |
| HumanEval | Python code generation |
| MBPP | Basic Python programming problems |

---

## Post-training Objective

Post-training produces three downstream variants from the pretrained base:

| Variant | Objective |
|---|---|
| `instruct` | general instruction following and response control |
| `chat` | preference-aligned assistant behavior |
| `code` | code generation and code-specific instruction following |

Behavior goals:

- follow the requested format
- answer simple questions directly
- give useful explanations when needed
- avoid unsupported factual claims
- produce code when code is requested
- stop cleanly after the answer is complete

---

## Post-training data policy

### Instruct SFT

Instruct SFT uses a SmolTalk backbone plus the local `response_control` dataset generated by `finetune/data/response_control.py`.

| Model | External backbone | Local custom dataset |
|---|---|---|
| `slm-125m` | 50% of `HuggingFaceTB/smol-smoltalk` | `response_control` |
| `slm-350m` | full `HuggingFaceTB/smol-smoltalk` | `response_control` |
| `slm-1b` | full `HuggingFaceTB/smoltalk` | `response_control` |

### Chat DPO

Chat DPO starts from the instruct checkpoint and applies general preference alignment.

DPO uses `HuggingFaceH4/ultrafeedback_binarized` plus local targeted preference pairs from `alignment/data/prepare_dpo.py`.

### Code SFT

Code SFT starts from the instruct checkpoint and applies code-specific SFT.

Code SFT uses `ise-uiuc/Magicoder-OSS-Instruct-75K` plus local code examples generated by `finetune/data/prepare_sft.py`.

---

## Key Design Decisions

- **From scratch:** trains from curated data instead of starting from an external pretrained checkpoint.
- **Custom tokenizer:** trained on the project data mix with chat/code special tokens.
- **Staged post-training:** `instruct`, `chat`, and `code` are separate checkpoints with separate eval/export paths.
- **DPO alignment:** chat alignment uses DPO instead of PPO.
- **Streaming-first curation:** large sources are processed without full in-memory materialization.
- **Config generation:** `config-gen` produces hardware-aware configs for pretraining, SFT, and DPO.
- **vLLM serving:** exported models are served through an OpenAI-compatible vLLM endpoint.

---

## Production Serving

The `serve/` directory contains vLLM serving assets. The server exposes an OpenAI-compatible chat completions API.

```bash
curl http://slm-service:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slm-125m-chat",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Kubernetes manifests live in `serve/manifests/`.

---

## Related Projects

- [ai-infra](https://github.com/tohio/ai-infra) — Kubernetes infrastructure for model serving
- [rag-pipeline](https://github.com/tohio/rag-pipeline) — RAG pipeline using SLM-compatible models
- [multi-agent](https://github.com/tohio/multi-agent) — multi-agent research workflows
- [data-flywheel](https://github.com/tohio/data-flywheel) — data feedback pipeline for future training runs

---

## License

MIT
