# pretrain

Pretraining pipeline for SLM. Tokenizes the validated dataset into memory-mapped binary files and trains `SLMForCausalLM` from scratch using HuggingFace `Trainer`.

---

## Pipeline

```
data/validated/train.jsonl  ─┐
                             ├──►  pretrain/data/tokenize_data.py  ─►  data/tokenized/train.bin  (+ train.json)
data/validated/val.jsonl    ─┘                                         data/tokenized/val.bin    (+ val.json)
                                                                                │
                                                                                ▼
                                                                  pretrain/train.py
                                                                                │
                                                                                ▼
                                                                  results/slm-125m/final/
```

The train / val split is produced upstream by the curator's blend stage, not at runtime. After the blend shuffle, the last 0.5% of documents are routed to `val.jsonl` and the rest to `train.jsonl`. Because the shuffle makes order uniformly random, val is a clean random sample from the same distribution as train.

---

## Getting Started

**Prerequisites**

```bash
# Validated datasets — both splits produced by the curator
ls $DATA_DIR/validated/train.jsonl
ls $DATA_DIR/validated/val.jsonl

# Tokenizer must be trained and complete — both files required
ls $DATA_DIR/tokenizer/tokenizer_config.json   # contains chat_template
ls $DATA_DIR/tokenizer/slm_tokenizer.json      # raw BPE tokenizer
```

**Step 1 — Tokenize**

```bash
make tokenize

# Upload to S3 for use on GPU instance
make tokenize-upload SIZE=125m
```

**Step 2 — Download on GPU instance**

```bash
make tokenizer-download
make tokenize-download SIZE=125m DATE=YYYY-MM-DD
```

**Step 3 — Pretrain**

```bash
# Mini validation run — confirm the training loop works before committing
make pretrain-mini GPUS=1

# Single GPU
make pretrain SIZE=125m GPUS=1

# Multi-GPU
make accelerate-config-multi GPUS=4
make pretrain SIZE=125m GPUS=4

# Resume from last checkpoint
make pretrain-resume SIZE=125m GPUS=4
```

---

## Multi-GPU Config Scaling

> **Important:** The configs in `pretrain/configs/` are written assuming **1 GPU**. Before running multi-GPU training, update `gradient_accumulation_steps` and `max_steps` to keep the global batch size and token budget constant.

The invariant to preserve:
```
global_batch_tokens = micro_batch_size × gradient_accumulation_steps × num_gpus × seq_len
```

Each config includes a scaling comment with the exact values for 1, 4, and 8 GPUs.

Example for 125m:

| GPUs | gradient_accumulation_steps | max_steps |
|---|---|---|
| 1 | 8 | 152,000 |
| 4 | 2 | 38,000 |
| 8 | 1 | 19,000 |

Example for 1b:

| GPUs | gradient_accumulation_steps | max_steps |
|---|---|---|
| 1 | 64 | 57,000 |
| 4 | 16 | 14,250 |
| 8 | 8 | 7,125 |

---

## Configs

Token targets match `TARGET_CONFIGS` in `curator/scripts/curate.py`.

| Config | Model | Layers | Hidden | Steps (1 GPU) | Global batch | Target tokens |
|---|---|---|---|---|---|---|
| `gpt_mini.yaml` | `slm-mini` | 6 | 384 | 5k | 8 | validation only |
| `gpt_125m.yaml` | `slm-125m` | 12 | 768 | 152k | 32 | 5B (2 epochs) |
| `gpt_350m.yaml` | `slm-350m` | 24 | 1024 | 230k | 64 | 15B (2 epochs) |
| `gpt_1b.yaml` | `slm-1b` | 32 | 2048 | 57k | 128 | 30B (1 epoch) |

Global batch size = `micro_batch_size × gradient_accumulation_steps × num_gpus`.

---

## Files

```
pretrain/
├── configs/
│   ├── gpt_mini.yaml        training config for mini validation run
│   ├── gpt_125m.yaml        training config for 125M
│   ├── gpt_350m.yaml        training config for 350M
│   └── gpt_1b.yaml          training config for 1B
├── data/
│   ├── tokenize_data.py     JSONL → memory-mapped uint16 binary (train + val)
│   ├── upload_tokenized.py  S3 upload/download for tokenized binaries
│   └── dataset.py           PyTorch Dataset wrapping each .bin file
└── train.py                 HF Trainer pretraining entry point
```

---

## Tokenizer Validation

`train.py` validates the tokenizer directory before training starts and fails
immediately if anything is missing or corrupt. This prevents discovering a broken
checkpoint after hours of training. Each required file is `json.load()`ed to
catch truncation or other corruption — not just file-existence.

Required files:
- `tokenizer_config.json` — HuggingFace config including baked-in `chat_template`.
  Required by `train_sft.py` to load the tokenizer via `from_pretrained()`.
- `slm_tokenizer.json` — Raw BPE tokenizer used by `tokenize_data.py`.

If either is missing or corrupt: `make tokenizer && make tokenizer-upload && make tokenizer-download`

---

## Data Format

The tokenized datasets are flat binary files of uint16 token IDs:

```
[doc1_tok1, ..., doc1_tokN, EOS, doc2_tok1, ..., doc2_tokM, EOS, ...]
```

Documents are concatenated with EOS as separator — no padding, no wasted tokens. Each training example is a fixed-length window sliced from this array.

`tokenize_data.py` asserts the tokenizer's vocab fits in uint16 (< 65,536) before running — a larger vocab would silently overflow every token ID above 65,535 and corrupt the training data. If you ever grow the vocab past that limit, switch the binary format to uint32 in both `tokenize_data.py` and `dataset.py`.

`dataset.py` validates that each `.bin` file's metadata declares `dtype: uint16` and that `n_tokens` matches what's actually on disk. A dtype mismatch or truncated binary fails fast with a clear error rather than silently returning garbage token IDs.

---

## Val split

The val split is created by the curator, not at training time.

1. **Blend stage** — after shuffling all staging sources, the last `val_fraction` (default 0.5%) of documents are written to `data/curated/val.jsonl`. The rest go to `data/curated/train.jsonl`. Because the shuffle produces uniformly random order, val is an unbiased sample from the same distribution as train.
2. **Tokenize stage** — `tokenize_data.py` processes both files, producing `data/tokenized/train.bin` and `data/tokenized/val.bin` with matching `.json` metadata.
3. **Train stage** — `dataset.load_train_val()` wraps each `.bin` with a `PretrainingDataset` instance. No runtime splitting.

This replaces the earlier approach of splitting the single tokenized binary at runtime, which had two bugs: stale split files silently used on re-run, and the tail-of-the-stream val slice was not a uniform random sample when the shuffle had any ordering bias.

---

## Checkpoints

Checkpoints are saved every `save_steps` steps to `results/<model_name>/`:

```
results/slm-125m/
├── checkpoint-500/          HF checkpoint (weights + optimizer)
├── checkpoint-1000/
├── ...
└── final/                   final model after training completes
    ├── config.json
    ├── model.safetensors
    ├── tokenizer/            full tokenizer including tokenizer_config.json
    └── ...
```

The `final/` directory is a complete, self-contained HF model directory — load with `AutoModelForCausalLM.from_pretrained("results/slm-125m/final")`.

When resuming with `--resume`, `train.py` scans the output directory for `checkpoint-*` subdirectories and logs the latest one it picks up before training starts — so you can tell at a glance which state was loaded.

---

## Monitoring

Training metrics are logged to Weights & Biases automatically. Key metrics to watch:

| Metric | What to look for |
|---|---|
| `train/loss` | Should decrease smoothly. Spikes are normal, persistent plateaus are not. |
| `eval/loss` | Should track training loss. Widening gap = overfitting. |
| `train/learning_rate` | Cosine decay — should ramp up then decay smoothly. |
| `train/grad_norm` | Should stay below `gradient_clip_val=1.0`. Persistent high norms = instability. |

`train.py` runs a baseline eval before training starts, so the `eval/loss` curve in W&B begins at step 0 with the random-init loss. Without this, the first eval point would be hundreds of steps in and you'd lose the "loss went from X at step 0 to Y at step N" comparison.

Expected validation loss at convergence: **2.5–3.5** for `slm-125m` at 5B tokens.

---

## Multi-GPU Training

Multi-GPU training is configured via `accelerate_configs/` and launched through `make`. No code changes are needed — accelerate handles distributed training transparently.

```bash
# Configure accelerate for single GPU (mini validation)
make accelerate-config-single

# Configure accelerate for multi-GPU (full training)
make accelerate-config-multi GPUS=4

# Launch training
make pretrain SIZE=125m GPUS=4
make pretrain SIZE=350m GPUS=8
make pretrain SIZE=1b   GPUS=8

# Override config directly
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=4
```

---

## Hardware Estimates

| Model | GPU | Precision | Global batch | Est. tokens/sec | Est. time |
|---|---|---|---|---|---|
| `slm-125m` | 1× H200 | bf16 | 32 | _TBD — pending 125m run_ | _TBD_ |
| `slm-125m` | 8× H200 | bf16 | 32 | _TBD_ | _TBD_ |
| `slm-350m` | 8× H200 | bf16 | 64 | _TBD — pending 350m run_ | _TBD_ |
| `slm-1b` | 8× H200 | bf16 | 128 | _TBD — pending 1b run_ | _TBD_ |

Actual throughput depends on GPU generation, network topology (multi-node), CPU/GPU data-loading balance, and the specific cloud region. Measure your own throughput with `make pretrain-mini` before committing to a full run.

---

## Key Design Decisions

**Why memory-mapped binary?** Tokenizing on the fly during training adds significant CPU overhead and reduces GPU utilization. Pre-tokenizing once and loading with `np.memmap` gives near-instant data loading with constant memory usage regardless of dataset size.

**Why uint16?** Our 32k vocab fits in uint16 (max 65,535), halving the storage and memory bandwidth compared to int32. The `train.bin` for 5B tokens is ~10GB in uint16 vs ~20GB in int32. `tokenize_data.py` asserts this fits at startup so vocab growth past the limit fails loudly instead of silently corrupting the training data.

**Why val split produced by the curator, not at training time?** Two correctness bugs with a runtime split: (1) split files could become stale if `train.bin` was regenerated without noticing, silently using old val data; (2) the tail-of-stream slice is only an unbiased sample if the shuffle has perfectly uniform ordering — the disk-chunked shuffle used at 1b scale doesn't quite have that property. Splitting at blend time, after shuffle, eliminates both issues by construction. Val and train are tokenized independently and the dataset code is a thin wrapper over each binary.

**Why cosine LR schedule?** Cosine annealing smoothly decays the learning rate to a small minimum, giving the model time to converge without abrupt changes. Used by GPT-3, LLaMA, and most modern pretraining runs.

**Why 2 epochs at 125m and 350m, but 1 epoch at 1b?** It's about how the token budget compares to the supply of each source. At 125m (5B tokens) and 350m (15B tokens) the total budget is small enough that 2 epochs fits comfortably within every source's supply — repeating the data improves downstream performance with negligible overfitting risk at this scale. At 1b (30B tokens / 1 epoch) every source is already close to its supply ceiling, so no repetition is needed; adding a second epoch would force either per-source overflow or a lower-quality over-epoching loop. Modern small-model training (Llama, Phi, Qwen) follows the same pattern at scale — fresh tokens outperform repeated ones once the budget is big enough.

**Why gradient checkpointing for 1B only?** At 125M and 350M, activations fit comfortably in H200 memory. At 1B with sequence length 4096, activation memory becomes the bottleneck. Gradient checkpointing trades ~30% compute for ~60% memory reduction.

**Why baseline eval before training?** So the W&B eval curve starts at step 0 with random-init loss. Without this, the first eval point is `eval_steps` steps in and you can't see how much the model has actually learned. The baseline is skipped on `--resume` since the original run already has one.

**Why validate the tokenizer (and parse the JSON) before training?** A missing or truncated `tokenizer_config.json` won't affect pretraining loss but makes the saved checkpoint unusable downstream — `train_sft.py` requires it to load the chat template. A corrupt file that passes a naive file-existence check crashes much later. Parsing the JSON up front — not just checking presence — turns a late failure into an early one.

**Why pre-count documents before tokenizing?** `tokenize_data.py` reads the input file once to count documents, then again to tokenize. The first pass is O(lines) and costs ~30 seconds at 1b scale; the payoff is an accurate ETA in the tqdm progress bar during the much longer tokenization run. Worth trading a small one-time cost for actionable progress reporting on multi-hour jobs.