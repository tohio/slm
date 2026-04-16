# SLM Pipeline — Command Reference

Complete reference for all `make` targets. For a high-level overview of the pipeline see the [README](README.md).

---

## Variables

All targets accept these variables as overrides:

| Variable | Default | Description |
|---|---|---|
| `SIZE` | `125m` | Model size target — controls data volume and config selection. One of `125m`, `350m`, `1b`. |
| `GPUS` | `1` | Number of GPUs for `accelerate launch`. Used by `pretrain`, `sft`, `sft-code`, `dpo`. |
| `WORKERS` | _(cpu_count - 2)_ | Parallel workers for filter, dedup, and blend stages. Defaults to `cpu_count - 2` automatically — only set this to override. |
| `DATA_DIR` | `data` | Root data directory. Override when using a separate disk volume. |
| `CONFIG` | _(derived from SIZE)_ | Explicit path to a YAML config file. Overrides the SIZE-derived default. |

---

## One-Time Setup

These targets are run once per instance before the pipeline starts.

---

### `make setup`

Bootstraps a fresh Ubuntu instance for curation. Installs system dependencies, configures the environment, and sets up the default data directory at `repo/data`.

```bash
make setup
```

**Use instead:** `make setup-data-dir` when using a separate disk volume (recommended).

---

### `make setup-data-dir`

Same as `setup` but configures a custom data directory. Use when data lives on a separate disk volume so it survives spot interruptions.

```bash
make setup-data-dir DATA_DIR=/data/slm/data
```

**Produces:** Configured instance with `DATA_DIR` set in the environment.

---

### `make install`

Creates a `.venv` virtualenv and installs Python dependencies from `requirements.txt`, including `orjson` and `fasttext-wheel`. Does not require a pre-activated virtualenv — safe to run on a fresh instance.

```bash
make install
```

All subsequent `make` targets use `.venv/bin/python` and `.venv/bin/accelerate` automatically — no need to activate the venv manually.

---

### `make install-uv`

Installs Python dependencies using `uv` (faster alternative to pip), including `orjson` and `fasttext-wheel`.

```bash
make install-uv
```

---

### `make install-conda`

Creates a conda environment named `slm` and installs dependencies including `orjson` and `fasttext-wheel`.

```bash
make install-conda
```

---

### `make install-gpu`

Installs Python dependencies for a GPU training instance. Includes `orjson` and `fasttext-wheel`. fasttext and kenlm validation dependencies are not needed on the GPU instance and are not installed separately.

```bash
make install-gpu
```

**Use on:** GPU training instances (pretrain, SFT, DPO, eval, export).
**Use `make install` on:** CPU curation instances (full dependency set including fasttext and kenlm).

---

### `make install-kenlm`

Installs the KenLM Python bindings from source. Required for the validation stage. KenLM is not on PyPI so this cannot be included in `requirements.txt`.

```bash
make install-kenlm
```

**Must run before:** `make validate` or `make download-kenlm-model`.

---

### `make install-orjson`

Installs `orjson` and `fasttext-wheel` into the existing virtualenv. Use this on instances where `make install` was run before these dependencies were added.

```bash
make install-orjson
```

---

### `make download-fasttext-model`

Downloads the fasttext language identification model (`lid.176.ftz`, ~1MB) to `DATA_DIR/models/`. Required by the quality filter stage for English language detection on Common Crawl documents.

```bash
make download-fasttext-model
make download-fasttext-model DATA_DIR=/data/slm/data
```

**Produces:** `data/models/lid.176.ftz`
**Must run before:** Any `curate` or `curate-filter` target.

---

### `make download-kenlm-model`

Downloads the KenLM English language model (`en.arpa.bin`, ~4GB) to `DATA_DIR/models/`. Required by the validation stage for perplexity filtering.

```bash
make download-kenlm-model
make download-kenlm-model DATA_DIR=/data/slm/data
```

**Produces:** `data/models/en.arpa.bin`
**Must run before:** `make validate`.

---

### `make accelerate-config`

Launches the interactive `accelerate config` wizard. Use this when you need a custom configuration not covered by the presets below.

```bash
make accelerate-config
```

---

### `make accelerate-config-single`

Copies `accelerate_configs/single_gpu.yaml` into place as the active accelerate config. Use for mini validation runs on a single GPU before committing to a full multi-GPU run.

```bash
make accelerate-config-single
```

**Configures:** 1 process, bf16, no distributed training.

---

### `make accelerate-config-multi`

Copies `accelerate_configs/multi_gpu.yaml` into place with `num_processes` set to `GPUS`. Use before full pretraining, SFT, and DPO runs on multi-GPU instances.

```bash
make accelerate-config-multi GPUS=8
make accelerate-config-multi GPUS=4
```

**Configures:** MULTI_GPU distributed training, bf16, all GPUs.

---

## Stage 1 — Data Curation

Downloads raw data from three sources (Wikipedia, CodeSearchNet Python, Common Crawl), applies quality filters, deduplicates, blends to target token ratios, and uploads to S3.

### Source mix

| Source | Mix | Tokens (1b target) | Notes |
|---|---|---|---|
| Common Crawl | 55% | 16.5B | Broad web coverage, aggressive filtering |
| Wikipedia EN | 25% | 7.5B | High quality, factual, structured |
| CodeSearchNet | 20% | 6B | Python only |

### Token targets

| Model | Total tokens | CC segments |
|---|---|---|
| `mini` | 1M | 2 |
| `125m` | 5B | 459 |
| `350m` | 15B | 916 (split across 2 crawls) |
| `1b` | 30B | 1,833 (split across 3 crawls) |

---

### `make curate`

Runs the full curation pipeline end-to-end: download → filter → dedup → blend → upload to S3. Worker count defaults to `cpu_count - 2` automatically.

```bash
make curate SIZE=125m
make curate SIZE=1b
make curate SIZE=125m WORKERS=32   # override worker count
```

**Requires:** fasttext model downloaded, AWS credentials in `.env`, S3 bucket configured.
**Produces:** `data/curated/train.jsonl`, `data/curated/blend_stats.json`, uploaded to S3.
**Note:** Already includes the S3 upload step — do not follow with `make curate-upload`.

---

### `make curate-mini`

Runs a minimal curation pipeline for pipeline validation. Caps Wikipedia at 5k docs, CodeSearchNet at 10k Python samples, and Common Crawl at 2 WARC segments. Total runtime ~30–45 min.

```bash
make curate-mini
```

**Use for:** Validating the full pipeline end-to-end before committing to a full run. Not suitable for training.

---

### `make curate-download`

Runs the download stage only — fetches Wikipedia, CodeSearchNet (Python), and Common Crawl raw data.

```bash
make curate-download SIZE=125m
```

**Produces:** `data/raw/wikipedia/`, `data/raw/code/`, `data/raw/common_crawl/`
**Resume behaviour:** Wikipedia and code skip shards already on disk. Common Crawl resumes from `data/raw/common_crawl/cc_progress.json` — delete that file to force a full re-download.

---

### `make curate-filter`

Runs quality filtering on all raw shards in parallel using all available CPU cores (defaults to `cpu_count - 2`). Applies heuristic filters including language detection via fasttext.

```bash
make curate-filter SIZE=125m
make curate-filter SIZE=125m WORKERS=32   # override worker count
```

**Requires:** `data/raw/` populated by `make curate-download`, fasttext model at `data/models/lid.176.ftz`.
**Produces:** `data/filtered/wikipedia/`, `data/filtered/code/`, `data/filtered/common_crawl/`
**Resume behaviour:** Skips shards already present in the filtered output directory.

**Filters applied:**

| Filter | Threshold | Catches | Skipped for code |
|---|---|---|---|
| Min length | 500 chars | Stubs, fragments | |
| Max length | 50k chars | Spam, boilerplate | |
| Mean word length | 3–10 chars | Gibberish, SEO spam | ✗ |
| Symbol ratio | < 8% symbols/words | Symbol-heavy spam | ✗ |
| Bullet ratio | < 90% bullet lines | Pure list content | |
| Ellipsis ratio | < 30% ellipsis lines | Truncated content | |
| Alpha ratio | > 75% alpha chars | Numeric/code spam | ✗ |
| Repeated lines | < 20% duplicates | Boilerplate | |
| Boilerplate patterns | < 2 matches | Cookie notices, T&Cs | ✗ |
| Language (fasttext) | en, score ≥ 0.65 | Non-English content | ✗ |
| Stop words (fallback) | ≥ 3 EN stop words | Non-English (no model) | ✗ |

---

### `make curate-dedup`

Runs deduplication on all filtered shards. Two stages: exact SHA-256 dedup (cross-source), then fuzzy MinHash LSH dedup via datatrove (disk-based, bounded RAM).

```bash
make curate-dedup SIZE=125m
make curate-dedup SIZE=125m WORKERS=32
```

**Requires:** `data/filtered/` populated by `make curate-filter`.
**Produces:** `data/filtered/wikipedia_deduped/`, `data/filtered/code_deduped/`, `data/filtered/common_crawl_deduped/`
**Resume behaviour:** Skips sources where the deduped output directory already exists.

---

### `make curate-blend`

Blends deduped sources to the target token ratio (55% CC / 25% Wikipedia / 20% code) and writes the final `train.jsonl`. Uses parallel staging and a chunked shuffle — no random disk seeks, optimal I/O throughput on any block storage.

```bash
make curate-blend SIZE=125m
make curate-blend SIZE=125m WORKERS=32
```

**Requires:** All `*_deduped/` directories populated by `make curate-dedup`.
**Produces:** `data/curated/train.jsonl`, `data/curated/blend_stats.json`
**Resume behaviour:** Skips if `train.jsonl` already exists — delete it to re-blend.

---

### `make curate-upload`

Uploads `data/curated/` to S3 under a versioned path: `{target}/{date}/curated/`. Each run gets its own dated folder so multiple runs never overwrite each other.

```bash
make curate-upload SIZE=125m
```

**Requires:** `data/curated/train.jsonl` produced by `make curate-blend`.
**Produces:** `s3://your-bucket/slm/data/125m/YYYY-MM-DD/curated/`
**Note:** `make curate` already includes this step — only use `curate-upload` when running stages individually.

---

## Stage 2 — Validation

Applies a second round of quality filtering and perplexity filtering to the blended dataset using KenLM.

---

### `make validate`

Runs the validation pipeline on `data/curated/train.jsonl`. Applies terminal punctuation check (C4), repeated n-gram check (Gopher), language detection (fasttext via datatrove), and perplexity filtering (KenLM, auto-threshold at 90th percentile). Skips perplexity filter for code documents.

```bash
make validate
```

**Requires:** `data/curated/train.jsonl`, KenLM model at `data/models/en.arpa.bin`.
**Produces:** `data/validated/train.jsonl`, `data/validated/validation_stats.json`

---

### `make validate-upload`

Uploads `data/validated/` to S3 under a versioned path: `{target}/{date}/validated/`.

```bash
make validate-upload SIZE=125m
```

**Requires:** `data/validated/train.jsonl` produced by `make validate`.
**Produces:** `s3://your-bucket/slm/data/125m/YYYY-MM-DD/validated/train.jsonl`

---

### `make validate-datatrove`

Alternative validation using datatrove's pipeline instead of the manual implementation. More thorough, recommended for final production runs.

```bash
make validate-datatrove
```

---

## Stage 3 — Tokenizer

Trains a BPE tokenizer on the curated dataset.

---

### `make tokenizer`

Trains a BPE tokenizer with a 32k vocabulary and 16 special tokens (`<|user|>`, `<|assistant|>`, `<|code|>`, `<|endofturn|>`, etc.) on `data/curated/train.jsonl`.

```bash
make tokenizer
```

**Requires:** `data/curated/train.jsonl`
**Produces:** `data/tokenizer/` (vocab, merges, config)

---

### `make tokenizer-test`

Runs roundtrip, fertility, and chat template tests on the trained tokenizer.

```bash
make tokenizer-test
```

**Requires:** `data/tokenizer/` produced by `make tokenizer`.

---

## Stage 4 — Pretraining

Tokenizes the dataset to binary format and runs pretraining from scratch.

---

### `make tokenize`

Tokenizes `data/validated/train.jsonl` to a memory-mapped uint16 binary file. Worker count defaults to `cpu_count - 2` automatically. Verifies the output after writing.

```bash
make tokenize

# Override worker count if needed
python pretrain/data/tokenize_data.py --workers 8 --chunk-size 256 --verify
```

**Requires:** `data/validated/train.jsonl`, `data/tokenizer/`
**Produces:** `data/tokenized/train.bin`

---

### `make pretrain`

Runs pretraining from scratch using `accelerate launch`. Config is derived from `SIZE` by default.

```bash
make pretrain SIZE=125m GPUS=4
make pretrain SIZE=350m GPUS=6
make pretrain SIZE=1b   GPUS=8

# Override config explicitly
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=4
```

**Requires:** `data/tokenized/train.bin`, `data/tokenizer/`, accelerate configured.
**Produces:** `results/slm-$(SIZE)/` checkpoints, W&B run.

---

### `make tokenize-upload`

Uploads `data/tokenized/` to S3 under a versioned path. Run on the CPU curation instance after `make tokenize`. At 1b scale the binary exceeds 50GB — uploading once and downloading on the GPU instance is faster and cheaper than re-tokenizing on expensive GPU hardware.

```bash
make tokenize-upload SIZE=125m
```

**Requires:** `data/tokenized/train.bin` produced by `make tokenize`.
**Produces:** `s3://your-bucket/slm/data/125m/YYYY-MM-DD/tokenized/`

---

### `make tokenize-download`

Downloads the tokenized binary from S3 to `data/tokenized/`. Run on the GPU training instance before `make pretrain`.

```bash
make tokenize-download SIZE=125m DATE=2026-04-12
```

**Requires:** A prior `make tokenize-upload` run, AWS credentials in `.env`.
**Produces:** `data/tokenized/train.bin`

---

### `make tokenizer-upload`

Uploads `DATA_DIR/tokenizer/` to S3. Run on the CPU curation instance after `make tokenizer`.

```bash
make tokenizer-upload
```

**Requires:** `data/tokenizer/` produced by `make tokenizer`.
**Produces:** `s3://your-bucket/slm/data/tokenizer/`

---

### `make tokenizer-download`

Downloads the tokenizer from S3 to `DATA_DIR/tokenizer/`. Run on the GPU instance before SFT or DPO.

```bash
make tokenizer-download
```

**Requires:** A prior `make tokenizer-upload` run.
**Produces:** `DATA_DIR/tokenizer/`

---

### `make pretrain-mini`

Runs a minimal pretraining pass using `pretrain/configs/gpt_mini.yaml` — a 6-layer, 384-hidden model trained for 500 steps. Use this to validate the full training loop before committing to an expensive full pretraining run.

```bash
make pretrain-mini GPUS=1
```

**Requires:** `data/tokenized/train.bin`, `data/tokenizer/`, accelerate configured.
**Produces:** `results/slm-mini/` checkpoint.
**Runtime:** ~5–10 min on a single GPU.
**Note:** The mini model is not useful for inference — it is a pipeline validation tool only.

---

### `make pretrain-resume`

Resumes pretraining from the last checkpoint.

```bash
make pretrain-resume SIZE=125m GPUS=4
```

**Requires:** Existing checkpoint in `results/slm-$(SIZE)/`.

---

## Stage 5 — Supervised Fine-Tuning

---

### `make prepare-sft`

Downloads and prepares both chat and code SFT datasets.

```bash
make prepare-sft
```

**Produces:** `data/sft/chat/`, `data/sft/code/`

---

### `make sft`

Runs chat supervised fine-tuning on the pretrained checkpoint.

```bash
make sft SIZE=125m GPUS=4
make sft CONFIG=finetune/configs/sft_chat_125m.yaml GPUS=4
```

**Requires:** `results/slm-$(SIZE)/final`, `data/sft/chat/`
**Produces:** `results/slm-$(SIZE)-sft/` checkpoints.

---

### `make sft-resume`

Resumes chat SFT from the last checkpoint.

```bash
make sft-resume SIZE=125m GPUS=4
```

---

### `make sft-code`

Runs code supervised fine-tuning on the chat SFT checkpoint. Uses a lower learning rate than chat SFT to reduce catastrophic forgetting.

```bash
make sft-code SIZE=125m GPUS=4
```

**Requires:** `results/slm-$(SIZE)-sft/final`, `data/sft/code/`
**Produces:** `results/slm-$(SIZE)-sft-code/` checkpoints.

---

### `make sft-code-resume`

Resumes code SFT from the last checkpoint.

```bash
make sft-code-resume SIZE=125m GPUS=4
```

---

## Stage 6 — Preference Alignment

---

### `make prepare-dpo`

Downloads and prepares the DPO preference dataset blend.

```bash
make prepare-dpo
```

**Produces:** `data/dpo/`

---

### `make dpo`

Runs DPO alignment on the SFT checkpoint using `trl`'s `DPOTrainer`.

```bash
make dpo SIZE=125m GPUS=2
make dpo CONFIG=alignment/configs/dpo_125m.yaml GPUS=2
```

**Requires:** `results/slm-$(SIZE)-sft-code/final`, `data/dpo/`
**Produces:** `results/slm-$(SIZE)-dpo/` checkpoints.

---

### `make dpo-resume`

Resumes DPO from the last checkpoint.

```bash
make dpo-resume SIZE=125m GPUS=2
```

---

## Stage 7 — Evaluation

---

### `make eval`

Evaluates the final DPO checkpoint on standard benchmarks via `lm-evaluation-harness`: HellaSwag, ARC-Easy, ARC-Challenge, MMLU, TruthfulQA, HumanEval.

```bash
make eval SIZE=125m
```

**Requires:** `results/slm-$(SIZE)-dpo/final`
**Produces:** Benchmark results printed to stdout and saved to `results/slm-$(SIZE)-dpo/eval/`.

---

## Stage 8 — Export

---

### `make export`

Exports all three model variants to HuggingFace Hub.

```bash
make export SIZE=125m
```

**Requires:** All three checkpoints present, `HF_TOKEN` in `.env`.
**Produces:** `tohio/slm-$(SIZE)`, `tohio/slm-$(SIZE)-instruct`, `tohio/slm-$(SIZE)-chat` on Hub.

---

### `make export-base`

Exports the base pretrained checkpoint to `tohio/slm-{size}`.

```bash
make export-base SIZE=125m
```

---

### `make export-instruct`

Exports the instruction-tuned checkpoint to `tohio/slm-{size}-instruct`.

```bash
make export-instruct SIZE=125m
```

---

### `make export-chat`

Exports the DPO-aligned checkpoint to `tohio/slm-{size}-chat`.

```bash
make export-chat SIZE=125m
```

---

## Stage 10 — Serving

---

### `make serve`

Launches a vLLM server using the Hub model. Exposes an OpenAI-compatible REST API.

```bash
make serve SIZE=125m
```

---

### `make serve-local`

Launches a vLLM server using a local checkpoint instead of the Hub model.

```bash
make serve-local SIZE=125m
```

---

## S3 Utilities

---

### `make s3-upload`

Uploads `data/curated/` to S3 at the `curated/` prefix. Unlike `curate-upload`, this is not versioned by date — it overwrites.

```bash
make s3-upload
```

---

### `make s3-download`

Downloads curated data from S3 at the `curated/` prefix to `data/curated/`.

```bash
make s3-download
```

---

### `make s3-list`

Lists all objects in the configured S3 bucket under the SLM prefix.

```bash
make s3-list
```

---

## Clean

---

### `make clean`

Removes Python cache files and log directories. Safe to run at any time.

```bash
make clean
```

---

### `make clean-data`

Removes all data directories. Does not remove `data/models/`. Does not affect S3.

```bash
make clean-data
```

⚠️ **Destructive**

---

### `make clean-results`

Removes all training results. Ensure results are exported or backed up first.

```bash
make clean-results
```

⚠️ **Destructive**

---

### `make clean-logs`

Removes the `logs/` directory.

```bash
make clean-logs
```

---

## Infrastructure

### Data Curation (CPU) — Stages 1–4a

| Target | vCPUs | RAM | Est. runtime |
|---|---|---|---|
| mini (validation) | Any | 4GB+ | ~30–45 min |
| 125m | 16 vCPU | 32GB | ~8–12 hrs |
| 350m | 32 vCPU | 64GB | ~20–28 hrs |
| 1b | 64 vCPU | 256GB | ~30–40 hrs |

Run close to `us-east-1` (AWS) or `us-east1` (GCP) to minimise Common Crawl egress latency. Attach a persistent disk (500GB+) for `DATA_DIR` so data survives spot/preemptible interruptions. [Nebius](https://nebius.com) AMD Epyc Genoa instances (64 vCPU, 256GiB RAM) offer strong price/performance for curation.

### Training (GPU) — Stages 4b–6

| Target | GPUs | VRAM | Est. pretrain runtime |
|---|---|---|---|
| mini (validation) | 1× any GPU | 8GB+ | ~5–10 min |
| 125m | 4× H100 or A100 | 320GB+ | ~12–18 hrs |
| 350m | 8× H100 or A100 | 640GB+ | ~24–36 hrs |
| 1b | 8× H100 or A100 | 640GB+ | ~72–96 hrs |

---

## GPU Instance Setup

### Fresh instance or after preemptible restart

```bash
# 1. Clone and enter repo
git clone https://github.com/tohio/slm.git
cd slm

# 2. Install make if needed
sudo apt install -y make

# 3. Fill in credentials
cp .env.sample .env
vi .env    # WANDB_API_KEY, HF_TOKEN, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET

# 4. Run GPU setup
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m DATE=2026-04-12
source ~/.bashrc

# 5. Validate the full pipeline end to end (~15 min total)
make pretrain-mini  GPUS=1
make prepare-sft
make sft-mini       GPUS=1
make sft-code-mini  GPUS=1
make prepare-dpo
make dpo-mini       GPUS=1
make eval-mini

# 6. Full training pipeline
make accelerate-config-multi GPUS=8
make pretrain    SIZE=125m GPUS=8
make prepare-sft
make sft         SIZE=125m GPUS=8
make sft-code    SIZE=125m GPUS=8
make prepare-dpo
make dpo         SIZE=125m GPUS=8
make eval        SIZE=125m
make export      SIZE=125m
```

### After a preemptible restart

```bash
cd ~/slm
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m
source ~/.bashrc
make pretrain-resume SIZE=125m GPUS=8
```

**Not needed on the GPU instance:**
- `make download-fasttext-model` — only used during curation filtering
- `make download-kenlm-model` — only used during validation
- `make install-kenlm` — only used during validation
- `make curate` — runs on the CPU curation instance
- `make validate` — runs on the CPU curation instance
- `make tokenize` — runs on the CPU curation instance

---

## Full Pipeline Reference

```bash
# ── One-time setup ─────────────────────────────────────────────────────────────
make install                                            # install all dependencies
make download-fasttext-model DATA_DIR=/data/slm/data   # ~1MB, for CC language filtering
make download-kenlm-model    DATA_DIR=/data/slm/data   # ~4GB, for validation perplexity

# ── Data ───────────────────────────────────────────────────────────────────────
make curate SIZE=1b                 # Stage 1: download, filter, dedup, blend, upload
make validate                       # Stage 2: perplexity filter
make validate-upload SIZE=1b        # Stage 2: push validated data to S3

# ── Tokenizer ──────────────────────────────────────────────────────────────────
make tokenizer                      # Stage 3: train BPE tokenizer
make tokenizer-upload               # Stage 3: push tokenizer to S3
make tokenize                       # Stage 4a: tokenize to binary
make tokenize-upload SIZE=1b        # Stage 4a: push tokenized binary to S3

# ── Training (GPU instance) ────────────────────────────────────────────────────
make setup-gpu DATA_DIR=/data/slm/data SIZE=1b DATE=2026-04-15
make accelerate-config-multi GPUS=8
make pretrain-mini GPUS=1           # Stage 4b: validate training loop (~5-10 min)
make pretrain    SIZE=1b GPUS=8     # Stage 4b: pretrain from scratch
make prepare-sft                    # Stage 5a: download SFT datasets
make sft         SIZE=1b GPUS=8     # Stage 5b: chat SFT
make sft-code    SIZE=1b GPUS=8     # Stage 5c: code SFT
make prepare-dpo                    # Stage 6a: download DPO datasets
make dpo         SIZE=1b GPUS=8     # Stage 6b: DPO alignment

# ── Ship ───────────────────────────────────────────────────────────────────────
make eval        SIZE=1b            # Stage 7: benchmark evaluation
make export      SIZE=1b            # Stage 8: push to HuggingFace Hub
make serve       SIZE=1b            # Stage 10: launch vLLM server
```