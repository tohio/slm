# slm

A decoder-only language model trained from scratch — raw web data through to an aligned, serving-ready model. Covers the full lifecycle: data curation, validation, tokenizer training, pretraining, supervised fine-tuning, preference alignment, evaluation, and production serving.

> **Status:** This project is under active development. The pipeline is operational at 125m; 350m and 1b runs are pending. Items marked _TBD_ will be filled in as empirical data becomes available.

---

## Overview

Most LLM projects start from a pretrained checkpoint. This one doesn't. SLM is built entirely from scratch — from unstructured web crawl data to an instruction-following, chat-capable model deployed on Kubernetes.

The pipeline is modular and independently runnable at each stage. Every design decision is documented and justified.

**Models:** `tohio/slm-125m` · `tohio/slm-125m-instruct` · `tohio/slm-125m-chat` · `tohio/slm-350m` · `tohio/slm-350m-instruct` · `tohio/slm-350m-chat` · `tohio/slm-1b` · `tohio/slm-1b-instruct` · `tohio/slm-1b-chat`

![Architecture](docs/architecture.svg)

---

## Choosing a size

All three sizes run through the same code path — the only differences are config values and target token counts. Choose based on your time and compute budget:

| Size | Curation time | Training time | Rough cost | Suits |
|---|---|---|---|---|
| `slm-125m` | ~16 hrs (measured) | ~3–4 hrs (1× H200, with `make config-gen`) | _TBD_ | learning the pipeline, single-GPU runs |
| `slm-350m` | _TBD — pending 350m run_ | _TBD_ | _TBD_ | serious research budget, multi-GPU |
| `slm-1b` | _TBD — pending 1b run_ | _TBD_ | _TBD_ | production-useful small model, GPU cluster |

Most readers will find `125m` fits their budget. The `1b` path is here for readers with the compute — it uses the same commands and same config structure, and produces a more capable model. The pipeline is designed for all three to work reliably; the choice is about what you can afford, not what you can trust.

---

## Architecture

The model is a dense decoder-only transformer with a modern architecture:

| Component | Choice | Rationale |
|---|---|---|
| Positional encoding | RoPE | Better length generalisation, relative position awareness |
| Normalization | RMSNorm | Faster than LayerNorm, modern standard |
| Activation | SwiGLU | Better gradient flow, used by Llama, Mistral, Qwen |
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
├── config/
│   └── data_mix.py
│
├── config_gen/
│   ├── config_gen.py        utility: auto-generate per-GPU training configs
│   └── accel_gen.py         utility: auto-generate accelerate launch configs (DDP/FSDP)
│
├── model/
│   ├── config.py
│   ├── attention.py
│   ├── mlp.py
│   ├── norm.py
│   ├── block.py
│   └── model.py
│
├── curator/
│   ├── constants.py
│   ├── sources/
│   │   ├── common_crawl.py
│   │   ├── fineweb.py
│   │   ├── wikipedia.py
│   │   ├── pg19.py
│   │   ├── pes2o.py
│   │   ├── open_web_math.py
│   │   ├── stackexchange.py
│   │   ├── code_search_net.py
│   │   ├── stack_smol.py
│   │   ├── stack_v1.py
│   │   ├── stack_v2.py          disabled — see file header
│   │   ├── jupyter.py
│   │   └── conala.py
│   ├── filters/
│   │   ├── quality.py
│   │   └── dedup.py
│   └── scripts/
│       ├── curate.py
│       └── upload_s3.py
│
├── validation/
│   └── scripts/
│       ├── validate.py
│       └── upload_validated.py
│
├── tokenizer/
│   ├── train_tokenizer.py
│   └── test_tokenizer.py
│
├── pretrain/
│   ├── configs/
│   ├── data/
│   │   ├── tokenize_data.py
│   │   ├── upload_tokenized.py
│   │   └── dataset.py
│   └── train.py
│
├── finetune/
│   ├── configs/
│   ├── data/prepare_sft.py
│   └── train_sft.py
│
├── alignment/
│   ├── configs/
│   ├── data/prepare_dpo.py
│   └── train_dpo.py
│
├── eval/
│   └── eval.py
│
├── export/
│   └── export.py
│
├── inference/
│   ├── utils.py
│   ├── chat.py
│   └── generate.py
│
├── serve/
│   ├── manifests/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── hpa.yaml
│   │   └── pvc.yaml
│   └── serve.sh
│
├── scripts/
│   └── sanity_train.py
│
├── tests/
│   ├── conftest.py
│   ├── README.md
│   ├── test_config_gen.py        unit tests for config_gen/config_gen.py
│   ├── test_accel_gen.py         unit tests for config_gen/accel_gen.py
│   ├── data_pipeline/
│   │   ├── test_pipeline_curator.py
│   │   ├── test_pipeline_validate.py
│   │   └── test_pipeline_tokenizer.py
│   ├── model/
│   │   └── test_model.py
│   └── gpu_pipeline/
│       ├── conftest.py            adds --size pytest option for GPU pipeline tests
│       ├── test_pipeline_training.py
│       ├── test_pipeline_sft.py
│       └── test_pipeline_dpo.py
│
├── notebooks/
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
│   ├── COMMANDS.md
│   ├── DISK_SETUP.md
│   ├── architecture.svg
│   └── screenshots/
│
├── infra/
│   ├── setup.sh
│   └── setup_gpu_instance.sh
│
├── accelerate_configs/
│   ├── single_gpu.yaml
│   └── multi_gpu.yaml
│
├── Makefile
├── pytest.ini
├── requirements.txt
├── environment.yml
└── .env.sample
```

---

## Getting Started

**Prerequisites**
- Python 3.12+
- Ubuntu 24.04 (recommended — `setup.sh` targets noble)
- CUDA-capable GPU (for pretraining stages)
- AWS account (S3 for data storage)
- Weights & Biases account
- HuggingFace account + token (several sources are gated: FineWeb, the-stack-smol, the-stack-dedup)

**Disk setup (separate data volume)**

If you are attaching a secondary disk for your data directory (recommended for curation — you need 500GB+), mount it before cloning:

→ [docs/DISK_SETUP.md](docs/DISK_SETUP.md)

If you are using the boot disk only, skip this step.

**Installation**

On a fresh Ubuntu 24.04 cloud instance (recommended):
```bash
# Clone into /data/slm — requires /data to exist and be writable.
# If using a separate disk, complete docs/DISK_SETUP.md first.
git clone https://github.com/tohio/slm.git /data/slm
cd /data/slm

cp .env.sample .env
vi .env   # fill in S3_BUCKET, AWS credentials, WANDB_API_KEY, HF_TOKEN, HF_USERNAME, SWH_AUTH_TOKEN

sudo apt install -y make

# Custom data dir — recommended when using a separate disk volume
make setup-data-dir DATA_DIR=/data/slm/data

# Default data dir (repo/data) — boot disk only
# make setup
```

Using pip / uv / conda:
```bash
make install          # creates .venv and installs all dependencies
make install-kenlm    # kenlm not on PyPI — curation instance only

make install-uv       # alternative: uv
make install-conda    # alternative: conda
```

GPU training instance only:
```bash
make install-gpu     # training/eval/serving deps only (no curation deps)
```

**Accept dataset Terms of Use**

Before first run, visit and accept terms on these HuggingFace dataset pages (required for gated datasets used in curation):
- https://huggingface.co/datasets/HuggingFaceFW/fineweb
- https://huggingface.co/datasets/bigcode/the-stack-smol
- https://huggingface.co/datasets/bigcode/the-stack-dedup

---

**Run the full pipeline**

```bash
# Custom data dir — recommended when using a separate disk volume
make setup-data-dir DATA_DIR=/data/slm/data

# ── Step 1: Data Curation instance (CPU) ──────────────────────────────────────────
make download-fasttext-model DATA_DIR=/data/slm/data   # language ID model (~1MB)
make download-kenlm-model    DATA_DIR=/data/slm/data   # perplexity model (~4GB)

# ── Step 2: Validate curation pipeline ───────────────────────────────────────
# Exercises every curation stage end-to-end on tiny data — all 12 sources.
# All tests run here — catch issues before spending hours on the full run.
make curate-mini && make test-curator
make validate    && make test-validate
make tokenizer   && make test-tokenizer
make tokenize                     # produces train.bin + val.bin
make tokenize-upload SIZE=mini    # push mini tokenized binaries to S3 for GPU instance
make tokenizer-upload             # push tokenizer to S3 (shared across all sizes)

# ── Step 3: Full curation ─────────────────────────────────────────────────────
make curate SIZE=125m WORKERS=62    # Stage 1: download, filter, dedup, blend (→ train.jsonl + val.jsonl), upload
make validate                       # Stage 2: perplexity filter (applied to both splits)
make validate-upload SIZE=125m      # Stage 2: push validated data to S3
make tokenizer                      # Stage 3: train BPE tokenizer
make tokenizer-upload               # Stage 3: push tokenizer to S3
make tokenize                       # Stage 4a: tokenize both splits to binary
make tokenize-upload SIZE=125m      # Stage 4a: push tokenized binaries to S3

# ── Step 4: GPU instance setup ───────────────────────────────────────────────
# For mini validation — pulls mini tokenized binaries and tokenizer from S3
make setup-gpu DATA_DIR=/data/slm/data SIZE=mini DATE=YYYY-MM-DD
source ~/.bashrc

# ── Step 5: Validate training pipeline ───────────────────────────────────────
# Exercises every training stage end-to-end on a single GPU.
# All tests run here — catch issues before spending hours on the full run.
# GPU pipeline test targets default to SIZE=mini, so no flag needed here.
make accelerate-config-single       # single GPU for mini validation
make pretrain-mini  GPUS=1 && make test-training  SIZE=mini
make reinit-embeds  SIZE=mini       # Stage 4c: re-init chat special-token embeds before SFT
make prepare-sft
make sft-mini       GPUS=1 && make test-sft-chat  SIZE=mini
make sft-code-mini  GPUS=1 && make test-sft-code  SIZE=mini
make prepare-dpo
make dpo-mini       GPUS=1 && make test-dpo       SIZE=mini
make eval-mini

# ── Step 6: Full training ─────────────────────────────────────────────────────
# All training configs are auto-generated for the current GPU by `make config-gen`.
# That writes pretrain, SFT chat, SFT code, and DPO configs in one shot.
# To skip the auto-tune for a specific stage, edit the YAML by hand — see
# the "Multi-GPU Config Scaling" section below for the formula.
# Re-run setup-gpu to pull the 125m tokenized binaries before training.
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m DATE=YYYY-MM-DD
make accelerate-config-single        # single GPU — change to: make accelerate-config-multi GPUS=x for multi-GPU
make config-gen      SIZE=125m GPUS=1   # Stage 4-6: auto-tune pretrain + sft + dpo configs for current GPU
make pretrain        SIZE=125m GPUS=1   # Stage 4b: pretrain from scratch
make reinit-embeds   SIZE=125m          # Stage 4c: re-init chat special-token embeds before SFT
make eval-base       SIZE=125m          # Stage 7:  evaluate base variant
make export-base     SIZE=125m          # Stage 8:  push base model to Hub
make prepare-sft
make sft             SIZE=125m GPUS=1   # Stage 5b: chat SFT
make sft-code        SIZE=125m GPUS=1   # Stage 5c: code SFT
make eval-instruct   SIZE=125m          # Stage 7:  evaluate instruct variant
make export-instruct SIZE=125m          # Stage 8:  push instruct model to Hub
make prepare-dpo
make dpo             SIZE=125m GPUS=1   # Stage 6b: DPO alignment
make eval-chat       SIZE=125m          # Stage 7:  evaluate chat variant (also: make eval)
make export-chat     SIZE=125m          # Stage 8:  push chat model to Hub
make serve                              # Stage 10: launch vLLM server
```

For full documentation of every `make` target see [docs/COMMANDS.md](docs/COMMANDS.md).

---

## Tests

Tests validate real pipeline outputs at each stage. Each test target is paired with the make stage that produces the outputs it checks. See [tests/README.md](tests/README.md) for full documentation.

**CPU curation instance:**

```bash
make curate-mini   && make test-curator      # validate curation outputs (all 12 sources)
make validate      && make test-validate     # validate validation outputs
make tokenizer     && make test-tokenizer    # validate tokenizer outputs

make test-data-pipeline                      # run all three at once
```

**GPU training instance:**

GPU pipeline test targets accept `SIZE=<size>` to validate any model size. Default is `mini` (matches the pipeline-validation flow). Pass `SIZE=125m` (or `350m`, `1b`) after a full run.

```bash
# After mini runs (default SIZE=mini)
make pretrain-mini  GPUS=1  && make test-training
make sft-mini       GPUS=1  && make test-sft-chat
make sft-code-mini  GPUS=1  && make test-sft-code
make dpo-mini       GPUS=1  && make test-dpo

# After full runs
make test-training  SIZE=125m
make test-sft-chat  SIZE=125m
make test-sft-code  SIZE=125m
make test-dpo       SIZE=125m

make test-gpu-pipeline                               # run all four at once (mini)
```

**Unit tests — no pipeline outputs needed, runs anywhere:**

```bash
make test-model           # model architecture
make test-config-gen      # config generator
make test-accel-gen       # accelerate config generator
make test-unit            # all of the above
```

| Target | Stage | Validates |
|---|---|---|
| `test-curator` | `curate-mini` | Raw shards exist for all 12 sources, filter quality, dedup correctness, blend output, stats |
| `test-validate` | `validate` | Retention rate, subset correctness, quality of retained docs |
| `test-tokenizer` | `tokenizer` | Special token IDs, roundtrip, fertility, chat template |
| `test-data-pipeline` | all three above | Runs curator + validate + tokenizer tests |
| `test-training` | `pretrain` (any size) | Model loads, loss finite and below random init, dataset indexing |
| `test-sft-chat` | `sft` (any size) | SFT data format, model loads, chat template preserved, generation runs |
| `test-sft-code` | `sft-code` (any size) | Code model loads, loss finite, code special tokens present |
| `test-dpo` | `dpo` (any size) | DPO data format, chosen ≠ rejected, model loads, generation runs |
| `test-gpu-pipeline` | all four above | Runs training + sft-chat + sft-code + dpo tests |
| `test-model` | none | RMSNorm, SwiGLU, GQA, causal mask, weight tying, parameter count |
| `test-config-gen` | none | `config_gen/config_gen.py` algorithm, invariants, YAML rendering |
| `test-accel-gen` | none | `config_gen/accel_gen.py` DDP and FSDP YAML rendering |
| `test-unit` | none | All unit tests above |

---

## Multi-GPU Config Scaling

The training pipeline uses **pure data parallelism** at all model sizes — no tensor or pipeline parallelism. Adding GPUs splits the batch across them; each GPU keeps a full copy of the model.

The invariant to preserve when changing GPU count is the **global batch size**:

```
global_batch = micro_batch_size × gradient_accumulation_steps × num_gpus
```

If you double the GPUs, halve `gradient_accumulation_steps` to keep the global batch (and therefore the model and recipe) identical. For pretrain only, also rescale `max_steps` to preserve the number of training steps over the corpus — `max_steps_new = max_steps_old × old_gpus / new_gpus`.

### Per-size reference tables (pretrain)

The committed configs are written for 1 GPU. For multi-GPU pretraining, scale these values:

**125m** — global batch 32 sequences, 5B corpus × 2 epochs:

| GPUs | gradient_accumulation_steps | max_steps |
|---|---|---|
| 1 | 8 | 152,000 |
| 4 | 2 | 38,000 |
| 8 | 1 | 19,000 |

**350m** — global batch 128 sequences, 15B corpus × 2 epochs:

| GPUs | gradient_accumulation_steps | max_steps |
|---|---|---|
| 1 | 16 | 230,000 |
| 4 | 4 | 57,500 |
| 8 | 2 | 28,750 |

**1b** — global batch 128 sequences, 30B corpus × 1 epoch:

| GPUs | gradient_accumulation_steps | max_steps |
|---|---|---|
| 1 | 64 | 57,000 |
| 4 | 16 | 14,250 |
| 8 | 8 | 7,125 |

### Per-stage scaling fields

| Stage | Config location | Scaling fields |
|---|---|---|
| Pretrain | `pretrain/configs/gpt_{size}.yaml` | `gradient_accumulation_steps`, `max_steps` |
| SFT chat | `finetune/configs/sft_chat_{size}.yaml` | `gradient_accumulation_steps` |
| SFT code | `finetune/configs/sft_code_{size}.yaml` | `gradient_accumulation_steps` |
| DPO | `alignment/configs/dpo_{size}.yaml` | `gradient_accumulation_steps` |

SFT and DPO use `epochs` not `max_steps` — only `gradient_accumulation_steps` needs adjusting for those stages.

### Auto-tune the math (recommended)

`make config-gen-*` reads your GPU model and count and emits configs with the right values automatically — no manual scaling. Same math, just done by the script:

```bash
make config-gen-pretrain SIZE=125m GPUS=8     # writes pretrain/configs/gpt_125m.yaml
make config-gen-sft      SIZE=125m GPUS=8     # writes BOTH sft_chat and sft_code
make config-gen-dpo      SIZE=125m GPUS=8     # writes alignment/configs/dpo_125m.yaml
make config-gen          SIZE=125m GPUS=8     # convenience: all three
```

The script also picks `micro_batch_size` based on GPU memory (a bigger H200 fits a bigger micro batch than an A100 40GB), and decides whether to enable gradient checkpointing. For 1b on multi-GPU, prefer FSDP over DDP via `make accel-gen-fsdp GPUS=8`.

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

12 sources total — 7 non-code top-level sources plus 5 code sub-sources that share the 10% code budget. Scale-invariant percentages — the same mix applies at every size. Defined in `config/data_mix.py` and referenced by the curator, export, and notebooks — do not duplicate these numbers elsewhere.

| Source | Target Share | Notes |
|---|---|---|
| Common Crawl | 10% | direct WARC via trafilatura |
| FineWeb | 47.5% | `HuggingFaceFW/fineweb` sample-100BT, overflow sink |
| Wikipedia | 10% | `wikimedia/wikipedia` EN |
| pg19 | 2.5% | public-domain books pre-1919 |
| peS2o | 5% | `allenai/peS2o` v2 — academic papers |
| open-web-math | 10% | math-heavy web filtered from CC |
| StackExchange | 5% | Q+A across dozens of sites |
| Code (total) | 10% | split across 5 code sub-sources (see curator/README.md) |

When supply-constrained sources (peS2o, jupyter, and at 1b also Wikipedia / pg19 / open_web_math / stack_smol) fall short of their character budget, the deficit is automatically routed to FineWeb as an overflow sink. The mix shape is preserved; the token target is hit.

### Realized mix at 125m

The 125m run produces a corpus with the following actual breakdown. Supply-bound sources (peS2o, jupyter) under-fill their char target; the deficit routes to FineWeb, inflating its realized share above the 47.5% target. All other sources land on target.

| Source | Target Share | Realized Share |
|---|---|---|
| `common_crawl` | 10.00% | 10.00% |
| `fineweb` | 47.50% | 49.39% |
| `wikipedia` | 10.00% | 10.00% |
| `pg19` | 2.50% | 2.50% |
| `pes2o` | 5.00% | 3.22% ⚠ |
| `open_web_math` | 10.00% | 10.00% |
| `stackexchange` | 5.00% | 5.00% |
| `stack_v1` | 5.00% | 5.00% |
| `codesearchnet` | 3.50% | 3.50% |
| `stack_smol` | 1.00% | 1.00% |
| `jupyter` | 0.40% | 0.29% ⚠ |
| `conala` | 0.10% | 0.10% |

Realized total: ~5.00B corpus tokens (8.31M train + 41.8K val docs). The `blend_stats.json` written by the curator records these realized numbers; `export.py` reads from it to produce the per-model card.

### Token Targets

| Model | Corpus tokens | Epochs |
|---|---:|---:|
| `slm-125m` | 5B | 2 |
| `slm-350m` | 15B | 2 |
| `slm-1b` | 30B | 1 |

Why 1b uses 1 epoch: at 30B corpus tokens, every source stays below its supply
ceiling, so no repetition is needed. 125m and 350m use 2 epochs because their
smaller corpora leave comfortable headroom on every source.

Why 1b uses 1 epoch: at 30B tokens / 1 epoch, every source stays below its supply ceiling, so no repetition. Modern small-model training (Llama, Phi) follows the same pattern — fresh tokens outperform repeated ones. 125m and 350m retain 2 epochs because their smaller budgets leave comfortable headroom.

### Train / val split

The train and val splits are produced by the curator's blend stage, not at training time. After the blend stage shuffles the corpus, val is sampled uniformly across all sources via reservoir sampling and the rest goes to `train.jsonl`. Validation (KenLM perplexity filtering) and tokenization both process each split independently, so `val.bin` receives the same quality treatment as `train.bin`. At 125m the realized val mix matches train within ±0.25pp per source — see `blend_stats.json`'s `val_docs` field.

Splitting at blend time (rather than at training time) avoids two correctness bugs: runtime splitting silently drifts out of sync with the underlying tokenization, and the tail-of-stream slice isn't a uniform sample when the shuffle is disk-chunked at 1b scale. Splitting right after the blend shuffle — where order is provably random — gives a clean uniform sample and eliminates the staleness concern by construction.

See `curator/README.md` for full details on the mix, sub-source breakdowns, cap-and-redistribute behavior, and scaling beyond 1b.

---

## Infrastructure

### Data Curation (CPU) — Stages 1–4a

Runs on CPU instances. No GPU required. Hardware recommendations below, not floors — the pipeline streams everywhere and runs on less RAM with longer wall time.

| Target | vCPUs | RAM | Curation runtime |
|---|---|---|---|
| `mini` | 4+ | 8 GB | 30–60 min |
| `slm-125m` | 64 | 256 GB | ~16 hrs (measured: 11h25m download + 16m filter + 3h6m dedup + 3m blend) |
| `slm-350m` | 64 | 256 GB | _TBD — pending 350m run_ |
| `slm-1b` | 64 | 256 GB | _TBD — pending 1b run_ |

> **Measure your own throughput before committing.** Many variables dominate:
> network peering between your cloud and Common Crawl's AWS `us-east-1` origin,
> per-WARC CloudFront throughput at your time of day, disk IOPS, CPU generation,
> and CC's own throttling behavior. Cross-cloud (Nebius → AWS, GCP → AWS) runs
> can be 2–3× faster or slower than same-region runs. Before committing to a
> full run, time a `curate-mini` or `curate SIZE=125m` run to calibrate.

Run close to `us-east-1` (AWS) or `us-east1` (GCP) to minimise Common Crawl egress latency. Attach a persistent disk (500GB+) for `DATA_DIR` — the pipeline is fully resumable at every stage.

Use `tmux` to keep the pipeline running through session timeouts:
```bash
tmux new -s curate
make curate SIZE=125m WORKERS=62
# Ctrl+B, D to detach — tmux attach -t curate to reattach
```

### Training (GPU) — Stages 4b–6

Requires a CUDA-capable GPU instance. The pipeline uses **pure data parallelism** throughout all model sizes — no tensor parallelism or model parallelism is needed. The model is replicated on each GPU and the batch is split across GPUs.

> **Run `make config-gen` before pretrain.** It reads your GPU model and count
> and emits a tuned `pretrain/configs/gpt_$(SIZE).yaml` — no manual scaling
> needed for pretraining. SFT and DPO configs still need manual scaling for
> multi-GPU; see Multi-GPU Config Scaling above.

Runtime varies significantly by GPU type and count. Use `make pretrain-mini GPUS=1` first to validate the training loop and measure your actual throughput before committing to a full run.

| Target | Min VRAM | Notes |
|---|---|---|
| `mini` | 8 GB+ | any modern GPU — confirms training loop works |
| `slm-125m` | 16 GB+ per GPU | fits on any modern data center GPU; ~3–4 hrs on 1× H200 |
| `slm-350m` | 24 GB+ per GPU | A100 40GB or better recommended |
| `slm-1b` | 40 GB+ per GPU | A100 80GB / H100 / H200 recommended; gradient checkpointing enabled by `config-gen` when needed |

SFT and DPO runtimes are roughly 20–30% of pretraining time at the same model size. Use spot/preemptible instances — all training loops support `--resume` from the last checkpoint.

---

## Screenshots

| Screenshot | Stage | Description |
|---|---|---|
| `docs/screenshots/01_blend_stats.png` | Stage 1 | `blend_stats.json` showing source mix |
| `docs/screenshots/02_validation_report.png` | Stage 2 | Validation report — total, kept, and rejection breakdown |
| `docs/screenshots/03_tokenizer_test.png` | Stage 3 | Tokenizer test output — special tokens and fertility score |
| `docs/screenshots/04_pretrain_loss.png` | Stage 4 | W&B pretraining loss curve |
| `docs/screenshots/05_sft_loss.png` | Stage 5 | W&B chat SFT loss curve |
| `docs/screenshots/06_dpo_loss.png` | Stage 6 | W&B DPO loss curve |
| `docs/screenshots/07_eval_results.png` | Stage 7 | Benchmark results — HellaSwag, ARC, MMLU, TruthfulQA, HumanEval |
| `docs/screenshots/08_hf_hub.png` | Stage 8 | HuggingFace Hub model page for `tohio/slm-125m` |
| `docs/screenshots/09_chat_session.png` | Stage 9 | Interactive multi-turn chat session via `inference/chat.py` |
| `docs/screenshots/10_vllm_curl.png` | Stage 10 | `curl` request to vLLM server with response |

---

## Evaluation

Models are evaluated on standard benchmarks via `lm-evaluation-harness`. Each variant — base, instruct, chat — has its own eval target so the per-variant model cards can carry real benchmark scores:

```bash
make eval-base     SIZE=125m   # after pretrain
make eval-instruct SIZE=125m   # after SFT (chat + code)
make eval-chat     SIZE=125m   # after DPO (also: make eval)
```

| Benchmark | Measures |
|---|---|
| HellaSwag | Commonsense reasoning |
| ARC-Easy / ARC-Challenge | Science QA |
| MMLU | Broad knowledge |
| TruthfulQA | Factual accuracy |
| HumanEval | Python code generation |
| MBPP | Basic Python programming problems |

**Contamination stance.** None of these benchmarks appear in any training source. HumanEval, MBPP, APPS, HellaSwag, ARC, MMLU, and TruthfulQA are all absent from the curated data — the earlier `codeparrot/apps` source was explicitly dropped to keep APPS clean. Model cards can claim clean eval results without asterisks.

---

## Key Design Decisions

**Why from scratch?** Starting from an existing checkpoint is the right production choice. We start from scratch deliberately — it exercises every stage of the pipeline and provides full visibility into how data quality and tokenizer design interact with training dynamics.

**Why a custom tokenizer?** A tokenizer trained on your specific data mix encodes domain patterns more efficiently. Special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|code|>`, `<|endofturn|>` and more) are baked in from the start with a Jinja2 chat template, giving the model a clean and consistent format across pretraining, SFT, DPO, and inference.

**Why GQA over MHA?** At inference time, KV cache is the primary memory bottleneck. GQA reduces KV heads from 12 to 4 (125m) — a 3× reduction in KV memory with negligible quality loss. Directly improves throughput in vLLM.

**Why DPO over PPO?** At small model scale, PPO's actor-critic setup requires multiple models simultaneously and is sensitive to reward scaling. DPO achieves comparable alignment with a simpler training loop and no separate reward model.

**Why sequential SFT (chat → code)?** Sequential fine-tuning produces independently evaluable checkpoints at each stage, making regressions immediately visible. The code SFT uses a lower learning rate to reduce catastrophic forgetting of chat capability.

**Why per-variant eval targets?** `eval-base`, `eval-instruct`, and `eval-chat` evaluate the three checkpoints written by the pipeline (`results/slm-{size}/final`, `results/slm-{size}-chat-code/final`, `results/slm-{size}-dpo/final`). Running each one writes its own JSON output, which `export.py` then reads when building per-variant model cards on the Hub. A single combined `eval` target would either skip the base and instruct cards or require running them all in series at the end — splitting them out lets eval run inline with each pipeline stage.

**Why 12 data sources?** Distribution coverage. A model pretrained only on web scrape (even filtered) has characteristic weaknesses: poor factual recall on niche topics, no long-range coherence over book-length spans, weak technical/academic prose, weak math reasoning, weak Q+A structure, weak code. Each of the 12 sources covers a specific gap — 7 non-code top-level sources for prose breadth, plus 5 code sub-sources for code coverage from raw files (stack-v1) through curated function/notebook/intent corpora (CodeSearchNet, stack-smol, jupyter, CoNaLa). See [curator/README.md](curator/README.md) for the full mix and sub-source rationale.

**Why scale-invariant mix percentages?** A reader scaling from 125m to 1b changes one number (`target_tokens`) and gets proportionally more of everything — no per-scale mix tuning. Supply variance is handled by cap-and-redistribute, not by per-scale knobs.

**Why `rope_theta=500000` across all sizes?** RoPE's base period is the slow axis of the position encoding — larger values give the model room to extrapolate to longer contexts than it was trained on. Using 500000 uniformly across 125m, 350m, and 1b means any size can be length-extended later (via YaRN, dynamic scaling, or similar) without retraining from scratch. The tradeoff at 2048 context (125m, 350m) is negligible — large base values don't hurt in-context quality at short sequence lengths, and consistency across sizes is worth more than micro-optimising each tier. Llama 3 and Qwen follow this same pre-stretched-base pattern.

**Why different epoch counts per scale?** Corpus size versus per-source supply. At 125m (5B corpus tokens), 2 epochs is comfortable; at 1b (30B corpus tokens), 1 epoch leaves every source below its supply ceiling, so no repetition. Modern small-model training (Llama, Phi, Qwen) follows the single-epoch pattern at scale — fresh tokens outperform repeated ones.

**Why streaming-first curation?** At 1b with 30B+ tokens, materializing sources in memory is infeasible on reasonable hardware. FineWeb and stack-v1 require streaming; the other sources use it for consistency. RAM is not the load-bearing scaling axis — vCPU count and network throughput are. This means readers on modest hardware (32 GB RAM) can still run 1b, just slower.

**Why cap-and-redistribute?** Several sources have finite supply at large scales — peS2o (abstracts only) and jupyter are supply-bound at 350m+; Wikipedia, pg19, open_web_math, and stack_smol become supply-bound at 1b. Rather than add per-scale knobs or accept repetition, the deficit routes to FineWeb — which has 15T tokens of headroom — preserving mix shape and hitting the token target.

**Why a single `config/` package for locked values?** The data mix, token targets, CHARS_PER_TOKEN, CC_CHARS_PER_SEGMENT, PRETRAIN_VAL_FRACTION, and a few other constants are each read by multiple stages (curator, export, pretrain, notebooks, tests). Previously they were duplicated — and the duplicates drifted, most visibly in an export pipeline that was writing stale 3-source pretraining tables to the Hub while the curator was actually running the 12-source mix. Centralising into `config/data_mix.py` with an import-time `validate()` makes drift impossible: every consumer sees the same values, and percentages sum-to-100 at the moment the module is loaded.

**Why a separate `config_gen/` package for `config_gen.py`?** The pretrain configs need to be tuned per GPU — `micro_batch_size` that fits on H200 fits trivially on B200 but not on A100 40GB, and the right `gradient_accumulation_steps` depends on both GPU memory and the count. Hand-tuning these for every (size, GPU, num_gpus) combination is error-prone and stale configs silently waste GPU hours. Centralising the math in `config_gen/config_gen.py` — keyed off measured GPU specs and per-size memory profiles — makes "tune for this hardware" a one-line `make config-gen` rather than a careful manual edit. The script intentionally leaves LR, schedule, and architecture untouched: those are recipe decisions, not hardware decisions.

**Why parametrize GPU pipeline tests by `--size`?** A test pinned to `results/slm-mini/final` skips on any larger checkpoint — defeating the purpose after a real run. The four GPU pipeline test targets (`test-training`, `test-sft-chat`, `test-sft-code`, `test-dpo`) accept `SIZE=<size>` and pass `--size=<size>` to pytest, where a fixture in `tests/gpu_pipeline/conftest.py` derives the model directory. Same tests, same Makefile pattern as everything else, no new targets per size.

**Why vLLM for serving?** PagedAttention enables continuous batching and efficient KV cache management. The OpenAI-compatible API means any client built against the OpenAI SDK works out of the box.

**Why datatrove for dedup instead of datasketch?** datasketch's `MinHashLSH` is in-memory — at 350m it requires ~32GB; at 1b ~85GB and may not fit on a single instance. datatrove's disk-based pipeline uses a sort-based approach (signatures → buckets → cluster → filter) where RAM usage is bounded by shard size, not corpus size. Same approach used by FineWeb at trillion-token scale.

**Why HTTPS for Common Crawl instead of S3?** Direct S3 access to the `commoncrawl` bucket fails on EC2 instances with IAM roles attached — the instance role credentials are rejected by the bucket policy. HTTPS via `data.commoncrawl.org` works reliably regardless of instance credentials.

**Why fasttext for language detection?** Language detection runs on every Common Crawl document — tens of millions of pages. `langdetect` is pure Python and adds ~5–10ms per document. fasttext's `lid.176.ftz` model is C-backed, covers 176 languages, and runs ~1000× faster with equivalent accuracy.

---

## Scaling Beyond 1b

The pipeline is designed to extend past 1b. Scale-invariant percentages, streaming-first code, and cap-and-redistribute all generalise. As compute gets cheaper and faster, larger sizes become accessible.

To run at 3b or beyond:

1. Add a new entry to `TARGET_CONFIGS` in `config/data_mix.py` with the new `total_tokens`, `epochs`, and `cc_crawls` list.
2. Add a matching entry to `SIZE_PROFILES` in `config_gen/config_gen.py` with `state_gb`, `act_per_seq_gb_*`, `ctx`, `ref_global_batch`, `tokens`, `lr`, `hidden`, `layers`, and head counts. After this, `make config-gen SIZE=3b GPUS=N` produces a tuned pretrain config automatically.
3. Add hand-written SFT and DPO configs for the new size in `finetune/configs/` and `alignment/configs/`.
4. Review Wikipedia and pg19 supply: at budgets approaching 40B × 1 epoch, Wikipedia repetition approaches 1.6×. Options: drop Wikipedia's share, add multilingual Wikipedia, or accept the repetition.
5. Consider adding a second bulk-code source to avoid stack-v1 over-epoching at 5B+ code tokens.
6. Consider upgrading FineWeb from `sample-100BT` to a larger sample if overflow consumption gets close to 100B.

No core code changes are required for scaling — the target config, source mix, and cap-and-redistribute handle supply variance automatically. See [curator/README.md](curator/README.md) for full details.

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
- **Training scale** — multi-node training with FSDP across 8+ nodes for models beyond 1b.
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
