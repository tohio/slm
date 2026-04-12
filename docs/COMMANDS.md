# SLM Pipeline — Command Reference

Complete reference for all `make` targets. For a high-level overview of the pipeline see the [README](README.md).

---

## Variables

All targets accept these variables as overrides:

| Variable | Default | Description |
|---|---|---|
| `SIZE` | `125m` | Model size target — controls data volume and config selection. One of `125m`, `350m`, `1b`. |
| `GPUS` | `1` | Number of GPUs for `accelerate launch`. Used by `pretrain`, `sft`, `sft-code`, `dpo`. |
| `WORKERS` | _(cpu_count // 2)_ | Parallel workers for the dedup stage. |
| `DATA_DIR` | `data` | Root data directory. Override when using a separate EBS volume. |
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

**Use instead:** `make setup-data-dir` when using a separate EBS volume (recommended).

---

### `make setup-data-dir`

Same as `setup` but configures a custom data directory. Use when data lives on a separate EBS volume so it survives spot interruptions.

```bash
make setup-data-dir DATA_DIR=/data/slm/data
```

**Produces:** Configured instance with `DATA_DIR` set in the environment.

---

### `make install`

Installs Python dependencies from `requirements.txt` using pip.

```bash
make install
```

---

### `make install-uv`

Installs Python dependencies using `uv` (faster alternative to pip). Creates a `.venv` virtualenv first.

```bash
make install-uv
```

---

### `make install-conda`

Creates a conda environment named `slm` and installs dependencies.

```bash
make install-conda
```

---

### `make install-kenlm`

Installs the KenLM Python bindings from source. Required for the validation stage. KenLM is not on PyPI so this cannot be included in `requirements.txt`.

```bash
make install-kenlm
```

**Must run before:** `make validate` or `make download-kenlm-model`.

---

### `make download-fasttext-model`

Downloads the fasttext language identification model (`lid.176.ftz`, ~1MB) to `DATA_DIR/models/`. Required by the Common Crawl download stage for English language filtering.

```bash
make download-fasttext-model
make download-fasttext-model DATA_DIR=/data/slm/data
```

**Produces:** `data/models/lid.176.ftz`
**Must run before:** Any `curate` or `curate-download` target.

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

Launches the interactive `accelerate config` wizard. Run once on a GPU instance before any distributed training target (`pretrain`, `sft`, `sft-code`, `dpo`).

```bash
make accelerate-config
```

---

## Stage 1 — Data Curation

Downloads raw data from three sources (Wikipedia, CodeSearchNet, Common Crawl), applies quality filters, deduplicates, blends to target token ratios, and uploads to S3.

---

### `make curate`

Runs the full curation pipeline end-to-end: download → filter → dedup → blend → upload to S3.

```bash
make curate SIZE=125m WORKERS=16
make curate SIZE=350m WORKERS=32
```

**Requires:** fasttext model downloaded, AWS credentials in `.env`, S3 bucket configured.
**Produces:** `data/curated/train.jsonl`, `data/curated/blend_stats.json`, uploaded to S3.
**Note:** Already includes the S3 upload step — do not follow with `make curate-upload`.

---

### `make curate-mini`

Runs a minimal curation pipeline for pipeline validation. Caps Wikipedia at 5k docs, CodeSearchNet at 10k samples (Python + JS only), and Common Crawl at 2 WARC segments. Total runtime ~30–45 min.

```bash
make curate-mini
```

**Use for:** Validating the full pipeline end-to-end before committing to a full run. Not suitable for training.

---

### `make curate-download`

Runs the download stage only — fetches Wikipedia, CodeSearchNet, and Common Crawl raw data.

```bash
make curate-download SIZE=125m
```

**Produces:** `data/raw/wikipedia/`, `data/raw/code/`, `data/raw/common_crawl/`
**Resume behaviour:** Wikipedia and code skip shards already on disk. Common Crawl resumes from `data/raw/common_crawl/cc_progress.json` — delete that file to force a full re-download.

---

### `make curate-filter`

Runs quality filtering on all raw shards. Applies heuristic filters (FineWeb/Gopher-style) to remove low-quality documents.

```bash
make curate-filter SIZE=125m
```

**Requires:** `data/raw/` populated by `make curate-download`.
**Produces:** `data/filtered/wikipedia/`, `data/filtered/code/`, `data/filtered/common_crawl/`
**Resume behaviour:** Skips shards already present in the filtered output directory.

---

### `make curate-dedup`

Runs deduplication on all filtered shards. Two stages: exact SHA-256 dedup (cross-source), then fuzzy MinHash LSH dedup via datatrove (disk-based, bounded RAM).

```bash
make curate-dedup SIZE=125m WORKERS=16
```

**Requires:** `data/filtered/` populated by `make curate-filter`.
**Produces:** `data/filtered/wikipedia_deduped/`, `data/filtered/code_deduped/`, `data/filtered/common_crawl_deduped/`
**Resume behaviour:** Skips sources where the deduped output directory already exists.

---

### `make curate-blend`

Blends deduped sources to the target token ratio (70% CC / 20% Wikipedia / 10% code) and writes the final `train.jsonl`. Uses streaming to keep peak RAM at O(1).

```bash
make curate-blend SIZE=125m
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

Runs the validation pipeline on `data/curated/train.jsonl`. Filters on terminal punctuation, repeated lines, and perplexity (auto-threshold at 90th percentile).

```bash
make validate
```

**Requires:** `data/curated/train.jsonl`, KenLM model at `data/models/en.arpa.bin`.
**Produces:** `data/validated/train.jsonl`, `data/validated/validation_stats.json`

---

### `make validate-upload`

Uploads `data/validated/` to S3 under a versioned path: `{target}/{date}/validated/`. Implemented in `validation/scripts/upload_validated.py`, which mirrors the curated upload but uses the `validated` S3 path segment so the two artifacts are stored independently.

```bash
make validate-upload SIZE=125m
```

**Requires:** `data/validated/train.jsonl` produced by `make validate`.
**Produces:** `s3://your-bucket/slm/data/125m/YYYY-MM-DD/validated/train.jsonl`
**Note:** Re-uploading on the same day overwrites that day's run. Runs on different days are preserved independently.

---

### `make validate-datatrove`

Alternative validation using datatrove's pipeline instead of the manual implementation. Produces equivalent output with different internals.

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

Tokenizes `data/validated/train.jsonl` to a memory-mapped uint16 binary file using 8 workers. Verifies the output after writing.

```bash
make tokenize
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

### `make pretrain-resume`

Resumes pretraining from the last checkpoint.

```bash
make pretrain-resume SIZE=125m GPUS=4
```

**Requires:** Existing checkpoint in `results/slm-$(SIZE)/`.

---

## Stage 5 — Supervised Fine-Tuning

Fine-tunes the pretrained model on chat and code datasets using `trl`'s `SFTTrainer`.

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

Aligns the fine-tuned model using Direct Preference Optimisation (DPO).

---

### `make prepare-dpo`

Downloads and prepares the DPO preference dataset blend.

```bash
make prepare-dpo
```

**Produces:** `data/dpo/`

---

### `make dpo`

Runs DPO alignment on the SFT checkpoint using `trl`'s `DPOTrainer`. No separate reward model required.

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

Exports the final DPO checkpoint to HuggingFace Hub as `tohio/slm-$(SIZE)`. Generates a model card automatically.

```bash
make export SIZE=125m
```

**Requires:** `results/slm-$(SIZE)-dpo/final`, `HF_TOKEN` in `.env`.
**Produces:** `tohio/slm-$(SIZE)` on HuggingFace Hub.

---

## Stage 10 — Serving

---

### `make serve`

Launches a vLLM server using the Hub model. Exposes an OpenAI-compatible REST API.

```bash
make serve SIZE=125m
```

**Requires:** Model exported to `tohio/slm-$(SIZE)` on HuggingFace Hub, vLLM installed.

---

### `make serve-local`

Launches a vLLM server using a local checkpoint instead of the Hub model. Useful for testing before export.

```bash
make serve-local SIZE=125m
```

**Requires:** `results/slm-$(SIZE)-dpo/final`

---

## S3 Utilities

Low-level S3 operations for manual data management. Prefer the stage-specific upload targets (`curate-upload`, `validate-upload`) for normal pipeline use.

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

Removes Python cache files (`__pycache__`, `*.pyc`) and log directories. Safe to run at any time — does not touch data or results.

```bash
make clean
```

---

### `make clean-data`

Removes all data directories: `data/raw`, `data/filtered`, `data/curated`, `data/validated`, `data/tokenized`, `data/sft`, `data/dpo`, `data/dedup_scratch`.

```bash
make clean-data
```

⚠️ **Destructive** — does not remove `data/models/`. Does not affect S3.

---

### `make clean-results`

Removes all training results: `results/`.

```bash
make clean-results
```

⚠️ **Destructive** — removes all checkpoints. Ensure results are exported or backed up first.

---

### `make clean-logs`

Removes the `logs/` directory.

```bash
make clean-logs
```

---

## Full Pipeline Reference

Correct end-to-end sequence for a fresh run:

```bash
# ── One-time setup ─────────────────────────────────────────────────────────────
make download-fasttext-model DATA_DIR=/data/slm/data   # ~1MB, for CC language filtering
make download-kenlm-model    DATA_DIR=/data/slm/data   # ~4GB, for validation perplexity
make accelerate-config                                  # GPU instance only, run once

# ── Data ───────────────────────────────────────────────────────────────────────
make curate SIZE=125m WORKERS=16    # Stage 1: download, curate, upload to S3
make validate                       # Stage 2: perplexity filter
make validate-upload SIZE=125m      # Stage 2: push validated data to S3

# ── Tokenizer ──────────────────────────────────────────────────────────────────
make tokenizer                      # Stage 3: train BPE tokenizer
make tokenize                       # Stage 4a: tokenize to binary

# ── Training ───────────────────────────────────────────────────────────────────
make pretrain GPUS=4                # Stage 4b: pretrain from scratch
make prepare-sft                    # Stage 5a: download SFT datasets
make sft      GPUS=4                # Stage 5b: chat SFT
make sft-code GPUS=4                # Stage 5c: code SFT
make prepare-dpo                    # Stage 6a: download DPO datasets
make dpo      GPUS=2                # Stage 6b: DPO alignment

# ── Ship ───────────────────────────────────────────────────────────────────────
make eval                           # Stage 7: benchmark evaluation
make export                         # Stage 8: push to HuggingFace Hub
make serve                          # Stage 10: launch vLLM server
```