# SLM Pipeline — Command Reference

Complete reference for all `make` targets. For a high-level overview of the pipeline see the [README](README.md).

---

## Variables

All targets accept these variables as overrides:

| Variable | Default | Description |
|---|---|---|
| `SIZE` | `125m` | Model size target — controls data volume and config selection. One of `mini`, `125m`, `350m`, `1b`. |
| `GPUS` | `1` | Number of GPUs for `accelerate launch`. Used by `pretrain`, `sft`, `sft-code`, `dpo`. |
| `WORKERS` | _(cpu_count - 2)_ | Parallel workers for filter, dedup, and blend stages. Defaults to `cpu_count - 2` automatically — only set this to override. |
| `DATA_DIR` | `data` | Root data directory. Override when using a separate disk volume. |
| `CONFIG` | _(derived from SIZE)_ | Explicit path to a YAML config file. Overrides the SIZE-derived default. |
| `GPU` | _(auto-detect)_ | GPU model for `config-gen-*` (e.g. `h200`, `b200`, `h100`, `a100_80`). When unset, the script uses `nvidia-smi` to detect the GPU. |
| `MODE` | `balanced` | Tuning mode for `config-gen-*` — one of `conservative` (70% VRAM), `balanced` (80%, default), or `aggressive` (90%). Aggressive mode also allows non-power-of-2 micro_batch values. |
| `AGGRESSIVE` | _(unset)_ | Backwards-compat alias for `MODE=aggressive`. Wins over `MODE` if both are set. |

---

## One-Time Setup

These targets are run once per instance before the pipeline starts.

---

### `make setup`

Bootstraps a fresh Ubuntu 24.04 instance for curation. Installs system dependencies, configures the environment, and sets up the default data directory at `repo/data`.

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

## Tests

Tests validate real pipeline outputs at each stage. Run each test target immediately after its corresponding make stage. See [tests/README.md](../tests/README.md) for full documentation of what each test checks.

---

### `make test-curator`

Validates `make curate-mini` outputs — raw shards, filtered quality, deduplication correctness, `train.jsonl` source mix, and `blend_stats.json` accuracy.

```bash
make curate-mini && make test-curator
```

**Requires:** `DATA_DIR` set, `make curate-mini` completed.

---

### `make test-validate`

Validates `make validate` outputs — retention rate, subset correctness, quality of retained docs, and `validation_stats.json` consistency.

```bash
make validate && make test-validate
```

**Requires:** `make test-curator` passing, `make validate` completed.

---

### `make test-tokenizer`

Validates `make tokenizer` outputs — all 16 special token IDs correct, encode/decode roundtrip, no auto BOS/EOS, fertility < 1.5, chat template via `apply_chat_template`.

```bash
make tokenizer && make test-tokenizer
```

**Requires:** `make validate` completed.

---

### `make test-data-pipeline`

Runs all three data pipeline tests in sequence.

```bash
make test-data-pipeline
```

---

### `make test-training`

Validates pretrain outputs — model loads, config matches the size's YAML, loss is finite and below random-init threshold, dataset indexing correct.

Defaults to `SIZE=mini` so the standard pipeline-validation flow (`pretrain-mini` → `test-training`) works with no flag. Pass `SIZE=<size>` to validate a full run.

```bash
make pretrain-mini  GPUS=1 && make test-training            # validates results/slm-mini/final
make test-training  SIZE=125m                                # validates results/slm-125m/final
```

**Requires:** GPU instance, the matching pretrain run completed.

---

### `make test-sft-chat`

Validates chat SFT outputs — SFT data format, chat model loads, tokenizer has chat template, forward pass finite, generation runs.

```bash
make sft-mini GPUS=1 && make test-sft-chat                  # validates results/slm-mini-chat/final
make test-sft-chat SIZE=125m                                # validates results/slm-125m-chat/final
```

**Requires:** `make test-training` passing, the matching SFT run completed.

---

### `make test-sft-code`

Validates code SFT outputs — code model loads, forward pass finite, code special tokens present.

```bash
make sft-code-mini GPUS=1 && make test-sft-code             # validates results/slm-mini-chat-code/final
make test-sft-code SIZE=125m                                # validates results/slm-125m-chat-code/final
```

**Requires:** `make test-sft-chat` passing, the matching code SFT run completed.

---

### `make test-dpo`

Validates DPO outputs — DPO data format (prompt/chosen/rejected), chosen ≠ rejected, model loads, forward pass finite, generation runs.

```bash
make dpo-mini GPUS=1 && make test-dpo                       # validates results/slm-mini-dpo/final
make test-dpo SIZE=125m                                     # validates results/slm-125m-dpo/final
```

**Requires:** `make test-sft-code` passing, the matching DPO run completed.

---

### `make test-gpu-pipeline`

Runs all four GPU pipeline tests in sequence. Respects `SIZE`.

```bash
make test-gpu-pipeline                # all four against mini (default)
make test-gpu-pipeline SIZE=125m      # all four against 125m
```

---

### `make test-model`

Model architecture unit tests — no pipeline outputs needed, runs on CPU anywhere.

```bash
make test-model
```

**Covers:** RMSNorm, SwiGLU, GQA shapes, causal mask, weight tying, parameter count ~25M, save/load roundtrip.

---

### `make test-config-gen`

Config generator unit tests — no pipeline outputs needed, runs on CPU anywhere. Covers the algorithm (token budget invariants, global batch resolution, GPU memory budgeting) for pretrain, SFT, and DPO stages; user override flags; input validation; and YAML rendering.

```bash
make test-config-gen
```

**Covers:** All combinations of size × GPU × num_gpus across all three stages; gradient checkpointing auto-policy; conservative/balanced/aggressive modes; warnings system; output YAML round-trip parsing; recipe preservation (LR, betas, epochs, warmup_ratio are never mutated by the script).

---

### `make test-accel-gen`

Accelerate config generator unit tests — covers DDP and FSDP YAML rendering. No pipeline outputs needed.

```bash
make test-accel-gen
```

**Covers:** DDP `MULTI_GPU` distributed type, FSDP `FULL_SHARD` + `TRANSFORMER_BASED_WRAP` policy, alternative sharding strategies (`SHARD_GRAD_OP`), CPU offload toggle, custom transformer layer class, mixed-precision selection.

---

### `make test-unit`

Runs all unit tests — `test-model`, `test-config-gen`, and `test-accel-gen`. No GPU or pipeline outputs required.

```bash
make test-unit
```

---

## Stage 1 — Data Curation

Downloads raw data from 12 sources (7 non-code top-level + 5 code sub-sources), applies quality filters, deduplicates, blends to target token ratios, and uploads to S3. The mix is defined in `config/data_mix.py` — see [README.md](../README.md#source-mix) for the full per-source breakdown.

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
make curate SIZE=125m WORKERS=62   # override worker count
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

Runs the download stage only — fetches raw data for all 12 sources.

```bash
make curate-download SIZE=125m
```

**Produces:** `data/raw/<source>/` for each source.
**Resume behaviour:** Most sources skip shards already on disk. Common Crawl resumes from `data/raw/common_crawl/cc_progress.json` — delete that file to force a full re-download.

---

### `make curate-filter`

Runs quality filtering on all raw shards in parallel using all available CPU cores (defaults to `cpu_count - 2`). Applies heuristic filters including language detection via fasttext.

```bash
make curate-filter SIZE=125m
make curate-filter SIZE=125m WORKERS=62   # override worker count
```

**Requires:** `data/raw/` populated by `make curate-download`, fasttext model at `data/models/lid.176.ftz`.
**Produces:** `data/filtered/<source>/` for each source.
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
make curate-dedup SIZE=125m WORKERS=62
```

**Requires:** `data/filtered/` populated by `make curate-filter`.
**Produces:** `data/filtered/<source>_deduped/` for each source.
**Resume behaviour:** Skips sources where the deduped output directory already exists.

---

### `make curate-blend`

Blends deduped sources to the target token ratio defined in `config/data_mix.py` and writes the final `train.jsonl` and `val.jsonl`. Parallel staging writes one intermediate file per source, then a single-pass shuffle produces the final output. For corpora that fit in RAM (controlled by the `SHUFFLE_RAM_BUDGET_GB` env var, default 64GB), the shuffle is done in memory; for larger corpora the pipeline falls back to a two-pass chunked disk shuffle with purely sequential I/O.

```bash
make curate-blend SIZE=125m
make curate-blend SIZE=125m WORKERS=62

# Force the chunked disk shuffle (e.g. if other processes need the RAM)
SHUFFLE_RAM_BUDGET_GB=0 make curate-blend SIZE=1b
```

**Requires:** All `*_deduped/` directories populated by `make curate-dedup`.
**Produces:** `data/curated/train.jsonl`, `data/curated/val.jsonl`, `data/curated/blend_stats.json`
**Resume behaviour:** Skips if `train.jsonl` already exists — delete it to re-blend. Per-source staging files (`blend_{source}.jsonl`) are reused on restart if present.

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

Runs the validation pipeline on `data/curated/train.jsonl` and `val.jsonl`. Applies terminal punctuation check (C4), repeated n-gram check (Gopher), language detection (fasttext via datatrove), and perplexity filtering (KenLM, auto-threshold at 90th percentile). Skips perplexity filter for code documents.

```bash
make validate
```

**Requires:** `data/curated/train.jsonl`, KenLM model at `data/models/en.arpa.bin`.
**Produces:** `data/validated/train.jsonl`, `data/validated/val.jsonl`, `data/validated/validation_stats.json`

---

### `make validate-upload`

Uploads `data/validated/` to S3 under a versioned path: `{target}/{date}/validated/`.

```bash
make validate-upload SIZE=125m
```

**Requires:** `data/validated/train.jsonl` produced by `make validate`.
**Produces:** `s3://your-bucket/slm/data/125m/YYYY-MM-DD/validated/`

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

Tokenizes the dataset to binary format, generates a per-GPU training config, and runs pretraining from scratch.

---

### `make tokenize`

Tokenizes `data/validated/train.jsonl` and `val.jsonl` to memory-mapped uint16 binary files. Worker count defaults to `cpu_count - 2` automatically. Verifies the output after writing.

```bash
make tokenize
```

**Requires:** `data/validated/train.jsonl`, `data/tokenizer/`
**Produces:** `data/tokenized/train.bin`, `data/tokenized/val.bin`

---

### `make config-gen-pretrain`

Auto-generates `pretrain/configs/gpt_$(SIZE).yaml` tuned for the current GPU and GPU count. The script picks `micro_batch_size`, `gradient_accumulation_steps`, `max_steps`, `warmup_steps`, and `gradient_checkpointing` to hit the size's reference global batch and token budget while staying within a safe fraction of GPU VRAM (default 80%).

```bash
make config-gen-pretrain SIZE=125m GPUS=1                     # auto-detect GPU
make config-gen-pretrain SIZE=350m GPUS=4 GPU=h200            # explicit GPU
make config-gen-pretrain SIZE=1b GPUS=8 GPU=b200 MODE=aggressive  # 90% VRAM budget
```

**Requires:** Nothing — runs on CPU, no pipeline state needed.
**Produces:** `pretrain/configs/gpt_$(SIZE).yaml` (overwrites any existing file).

---

### `make config-gen-sft`

Auto-generates **both** SFT configs in one shot — `finetune/configs/sft_chat_$(SIZE).yaml` and `finetune/configs/sft_code_$(SIZE).yaml`. The script picks `micro_batch_size`, `gradient_accumulation_steps`, and `gradient_checkpointing` for each. SFT uses `epochs`, not `max_steps`, so the token-budget math doesn't apply.

The chat and code recipes have different LRs (chat > code) but the same memory profile and reference global batch — so both files can be generated from the same hardware decision.

```bash
make config-gen-sft SIZE=125m GPUS=1
make config-gen-sft SIZE=350m GPUS=4 GPU=h200
make config-gen-sft SIZE=1b GPUS=8 MODE=conservative
```

**Requires:** Nothing.
**Produces:** `finetune/configs/sft_chat_$(SIZE).yaml`, `finetune/configs/sft_code_$(SIZE).yaml`.

---

### `make config-gen-dpo`

Auto-generates `alignment/configs/dpo_$(SIZE).yaml`. DPO state is roughly 1.15× SFT (policy + frozen reference model) and activations are ~4× SFT (chosen + rejected pairs through both models), so the script accounts for that automatically.

```bash
make config-gen-dpo SIZE=125m GPUS=1
make config-gen-dpo SIZE=1b GPUS=8 GPU=b200
```

**Requires:** Nothing.
**Produces:** `alignment/configs/dpo_$(SIZE).yaml`.

DPO is sensitive to LR and batch size. The script always emits a `# Heads-up:` warning in the YAML header reminding you that the recipe LR (e.g. `2e-7` for 1b) was tuned for the reference global batch — if your auto-tuned global differs significantly, expect to retune.

---

### `make config-gen`

Convenience target — runs `config-gen-pretrain`, `config-gen-sft`, and `config-gen-dpo` for the same `SIZE` and `GPUS`. Generates all training configs the pipeline needs in one command.

```bash
make config-gen SIZE=125m GPUS=1
make config-gen SIZE=350m GPUS=4 GPU=h200 MODE=aggressive
```

**Run before:** `make pretrain`, `make sft`, `make sft-code`, `make dpo` — for reliable throughput on the GPU you're actually using.

---

### Tuning modes (all `config-gen-*` targets)

| Flag | VRAM budget | Notes |
|---|---|---|
| `MODE=conservative` | 70% | Leaves headroom; useful on preemptible VMs and unfamiliar hardware |
| `MODE=balanced` *(default)* | 80% | Comfortable margin |
| `MODE=aggressive` | 90% | Maximum throughput; allows non-power-of-2 micro_batch |
| `AGGRESSIVE=1` | 90% | Backwards-compat alias for `MODE=aggressive` |

**Generated YAML always includes a `# Heads-up:` section** in the header comment describing things you might want to verify — categories include low/high VRAM utilization, deviation from reference global batch, auto-policy enabling checkpointing, token-budget rounding loss (pretrain), DPO LR sensitivity, FSDP recommendation for 1b on multi-GPU, and a reminder that activation memory estimates are analytical (not measured). Inspect the YAML before training.

**Hand-edited configs are overwritten.** If you've manually tuned a config and want to keep changes, copy it first.

**Supported GPUs (via `GPU=...`):** `h200`, `b200`, `h100`, `h100_sxm`, `a100_80`, `a100_40`, `l40s`, `rtx4090`, `rtx5090`. Without `GPU=...`, the script reads `nvidia-smi` to detect.

---

### `make accel-gen-ddp`

Auto-generates `accelerate_configs/multi_gpu.yaml` with the right `num_processes` for your GPU count. Equivalent to the existing `accelerate-config-multi GPUS=N` target but more explicit — the latter performs a `sed` substitution and is preserved for backwards compat.

```bash
make accel-gen-ddp GPUS=8
```

**Use when:** training 125m or 350m at any GPU count, or 1b on small clusters with high-VRAM GPUs (≤ 4× H200/B200).

**Produces:** `accelerate_configs/multi_gpu.yaml`.

---

### `make accel-gen-fsdp`

Auto-generates `accelerate_configs/fsdp.yaml` with `FULL_SHARD` strategy and `TRANSFORMER_BASED_WRAP` policy targeting the model's transformer block class.

```bash
make accel-gen-fsdp GPUS=8
```

**Use when:** training 1b on ≥ 4 GPUs. FSDP shards weights, gradients, and optimizer state across GPUs, freeing ~10 GB/GPU on optimizer state — frees that memory for larger micro_batches and removes the need for gradient checkpointing.

**Produces:** `accelerate_configs/fsdp.yaml`.

---

### `make pretrain`

Runs pretraining from scratch using `accelerate launch`. Config is derived from `SIZE` by default.

```bash
make config-gen-pretrain SIZE=125m GPUS=4    # generate config for current GPU
make pretrain            SIZE=125m GPUS=4

make pretrain SIZE=350m GPUS=6
make pretrain SIZE=1b   GPUS=8

# Override config explicitly
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=4
```

**Requires:** `data/tokenized/train.bin`, `data/tokenizer/`, `pretrain/configs/gpt_$(SIZE).yaml` (run `make config-gen-pretrain` first), accelerate configured.
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
**Produces:** `data/tokenized/train.bin`, `data/tokenized/val.bin`

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

Runs a minimal pretraining pass using `pretrain/configs/gpt_mini.yaml` — a 6-layer, 384-hidden model (~25M parameters) trained for 5000 steps (~40M tokens). Trains long enough to confirm loss decreases and produce semi-coherent output — use `inference/chat.py` after to verify the model is learning before committing to a full run.

```bash
make pretrain-mini GPUS=1
```

**Requires:** `data/tokenized/train.bin`, `data/tokenizer/`, accelerate configured.
**Produces:** `results/slm-mini/` checkpoint.
**Runtime:** ~30–45 min on a single GPU.
**Note:** Semi-coherent output expected — enough to confirm the model is learning before a full run.

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

**Requires:** `results/slm-$(SIZE)/final`, `data/sft/chat/`, `finetune/configs/sft_chat_$(SIZE).yaml` (run `make config-gen-sft` first)
**Produces:** `results/slm-$(SIZE)-chat/` checkpoints.

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

**Requires:** `results/slm-$(SIZE)-chat/final`, `data/sft/code/`, `finetune/configs/sft_code_$(SIZE).yaml` (run `make config-gen-sft` first)
**Produces:** `results/slm-$(SIZE)-chat-code/` checkpoints.

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

**Requires:** `results/slm-$(SIZE)-chat-code/final`, `data/dpo/`, `alignment/configs/dpo_$(SIZE).yaml` (run `make config-gen-dpo` first)
**Produces:** `results/slm-$(SIZE)-dpo/` checkpoints.

---

### `make dpo-resume`

Resumes DPO from the last checkpoint.

```bash
make dpo-resume SIZE=125m GPUS=2
```

---

## Stage 7 — Evaluation

Each variant has its own eval target so per-variant model cards (written by `make export-*`) carry real benchmark scores. Eval results land under `results/eval/<checkpoint-name>/` — that's where `export.py` reads them from.

---

### `make eval-base`

Evaluates the base pretrained checkpoint on standard benchmarks via `lm-evaluation-harness`: HellaSwag, ARC-Easy, ARC-Challenge, MMLU, TruthfulQA, HumanEval.

```bash
make eval-base SIZE=125m
```

**Requires:** `results/slm-$(SIZE)/final`
**Produces:** `results/eval/slm-$(SIZE)/eval_<timestamp>.json`

---

### `make eval-instruct`

Evaluates the SFT-tuned (chat + code) checkpoint.

```bash
make eval-instruct SIZE=125m
```

**Requires:** `results/slm-$(SIZE)-chat-code/final`
**Produces:** `results/eval/slm-$(SIZE)-chat-code/eval_<timestamp>.json`

---

### `make eval-chat`

Evaluates the DPO-aligned checkpoint.

```bash
make eval-chat SIZE=125m
```

**Requires:** `results/slm-$(SIZE)-dpo/final`
**Produces:** `results/eval/slm-$(SIZE)-dpo/eval_<timestamp>.json`

---

### `make eval`

Alias for `make eval-chat` — evaluates the final aligned variant. The default for callers that want "the eval" without thinking about variants.

```bash
make eval SIZE=125m
```

---

### `make eval-mini`

Quick smoke test — runs HellaSwag with a 50-example limit on the mini DPO checkpoint. ~2 minutes. Use after `dpo-mini` during pipeline validation.

```bash
make eval-mini
```

**Requires:** `results/slm-mini-dpo/final`.

---

## Stage 8 — Export

---

### `make export`

Exports all three model variants to HuggingFace Hub. Each variant's model card includes whatever eval results are present in `results/eval/<checkpoint-name>/` at the time of export — run `make eval-base`, `eval-instruct`, `eval-chat` beforehand to populate them.

```bash
make export SIZE=125m
```

**Requires:** All three checkpoints present, `HF_TOKEN` and `HF_USERNAME` in `.env`.
**Produces:** `<HF_USERNAME>/slm-$(SIZE)`, `<HF_USERNAME>/slm-$(SIZE)-instruct`, `<HF_USERNAME>/slm-$(SIZE)-chat` on Hub.

---

### `make export-base`

Exports the base pretrained checkpoint to `<HF_USERNAME>/slm-{size}`. Reads `results/eval/slm-{size}/eval_<latest>.json` for the model card if present.

```bash
make export-base SIZE=125m
```

---

### `make export-instruct`

Exports the instruction-tuned checkpoint to `<HF_USERNAME>/slm-{size}-instruct`.

```bash
make export-instruct SIZE=125m
```

---

### `make export-chat`

Exports the DPO-aligned checkpoint to `<HF_USERNAME>/slm-{size}-chat`.

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
| 125m | 32 vCPU | 64GB | ~4–6 hrs |
| 350m | 64 vCPU | 128GB | ~10–14 hrs |
| 1b | 64 vCPU | 256GB | ~20–28 hrs |

> **Runtimes are rough reference points — measure your own.**
> The CC download stage dominates total runtime and is sensitive to: network
> peering between your cloud and AWS `us-east-1`, CloudFront throughput at
> your time of day, disk IOPS, and CC's throttling behavior. Cross-cloud
> (Nebius → AWS, GCP → AWS) runs can be 2–3× faster or slower than
> same-region (AWS `us-east-1`) runs. Time a `curate-mini` or
> `curate SIZE=125m` run on your target instance before committing to a
> full 1b run.

Run close to `us-east-1` (AWS) or `us-east1` (GCP) to minimise Common Crawl egress latency. Attach a persistent disk (500GB+) for `DATA_DIR` so data survives spot/preemptible interruptions. [Nebius](https://nebius.com) AMD Epyc Genoa instances (64 vCPU, 256GiB RAM) offer strong price/performance for curation.

### Training (GPU) — Stages 4b–6

| Target | GPUs | VRAM | Est. pretrain runtime |
|---|---|---|---|
| mini (validation) | 1× any GPU | 8GB+ | ~5–10 min |
| 125m | 1× H200 | 141GB | ~3–4 hrs (with `make config-gen`) |
| 125m | 4× H100 or A100 | 320GB+ | ~12–18 hrs |
| 350m | 8× H100 or A100 | 640GB+ | ~24–36 hrs |
| 1b | 8× H100 or A100 | 640GB+ | ~72–96 hrs |

> **Runtimes are rough reference points — measure your own.**
> Numbers assume bf16 training in a single-node data-parallel setup with the
> auto-generated `make config-gen` configs. GPU generation, interconnect
> bandwidth (NVLink vs PCIe vs cross-node), and whether activation
> checkpointing is enabled all have large effects. Use
> `make pretrain-mini GPUS=1` to measure step time on your hardware, then
> extrapolate by the ratio of your full-run `max_steps` to the mini run's
> `max_steps`.

---

## GPU Instance Setup

### Fresh instance or after preemptible restart

```bash
# 1. Clone and enter repo
git clone https://github.com/tohio/slm.git /data/slm
cd /data/slm

# 2. Install make if needed
sudo apt install -y make

# 3. Fill in credentials
cp .env.sample .env
vi .env    # WANDB_API_KEY, HF_TOKEN, HF_USERNAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET

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

# 6. Generate configs tuned for current GPU, then run full pipeline
make accelerate-config-multi GPUS=8
make config-gen      SIZE=125m GPUS=8       # convenience: auto-tune pretrain + sft + dpo configs
make pretrain        SIZE=125m GPUS=8
make eval-base       SIZE=125m
make export-base     SIZE=125m
make prepare-sft
make sft             SIZE=125m GPUS=8
make sft-code        SIZE=125m GPUS=8
make eval-instruct   SIZE=125m
make export-instruct SIZE=125m
make prepare-dpo
make dpo             SIZE=125m GPUS=8
make eval-chat       SIZE=125m              # also: make eval
make export-chat     SIZE=125m
```

### After a preemptible restart

```bash
cd /data/slm
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

# ── Validate curation pipeline ─────────────────────────────────────────────────
make curate-mini                    # validate pipeline end-to-end on tiny data (~30–45 min)
make test-data-pipeline             # verify outputs are correct

# ── Full curation ──────────────────────────────────────────────────────────────
make curate SIZE=1b                 # Stage 1: download, filter, dedup, blend, upload
make validate                       # Stage 2: perplexity filter
make validate-upload SIZE=1b        # Stage 2: push validated data to S3

# ── Tokenizer ──────────────────────────────────────────────────────────────────
make tokenizer                      # Stage 3: train BPE tokenizer
make tokenizer-upload               # Stage 3: push tokenizer to S3
make tokenize                       # Stage 4a: tokenize to binary
make tokenize-upload SIZE=1b        # Stage 4a: push tokenized binary to S3

# ── GPU instance setup ─────────────────────────────────────────────────────────
make setup-gpu DATA_DIR=/data/slm/data SIZE=1b DATE=2026-04-15

# ── Validate training pipeline ─────────────────────────────────────────────────
# GPU pipeline test targets default to SIZE=mini, no flag needed.
make accelerate-config-single
make pretrain-mini  GPUS=1 && make test-training
make prepare-sft
make sft-mini       GPUS=1 && make test-sft-chat
make sft-code-mini  GPUS=1 && make test-sft-code
make prepare-dpo
make dpo-mini       GPUS=1 && make test-dpo
make eval-mini

# ── Full training ──────────────────────────────────────────────────────────────
# 1b on multi-GPU benefits from FSDP (saves ~10 GB/GPU on optimizer state)
make accel-gen-fsdp GPUS=8           # accelerate config: FSDP for 1b
make config-gen      SIZE=1b GPUS=8   # auto-tune all training configs (pretrain + sft + dpo)
make pretrain        SIZE=1b GPUS=8   # Stage 4b: pretrain from scratch
make eval-base       SIZE=1b          # Stage 7:  evaluate base
make export-base     SIZE=1b          # Stage 8:  push base to Hub
make prepare-sft                      # Stage 5a: download SFT datasets
make sft             SIZE=1b GPUS=8   # Stage 5b: chat SFT
make sft-code        SIZE=1b GPUS=8   # Stage 5c: code SFT
make eval-instruct   SIZE=1b          # Stage 7:  evaluate instruct
make export-instruct SIZE=1b          # Stage 8:  push instruct to Hub
make prepare-dpo                      # Stage 6a: download DPO datasets
make dpo             SIZE=1b GPUS=8   # Stage 6b: DPO alignment
make eval-chat       SIZE=1b          # Stage 7:  evaluate chat (also: make eval)
make export-chat     SIZE=1b          # Stage 8:  push chat to Hub

# ── Ship ───────────────────────────────────────────────────────────────────────
make serve           SIZE=1b          # Stage 10: launch vLLM server
```