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
                                                                  config_gen/config_gen.py
                                                                  (auto-tunes config for current GPU)
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

**Step 3 — Generate per-GPU config**

```bash
# Auto-detect the GPU and write pretrain/configs/gpt_125m.yaml
make config-gen-pretrain SIZE=125m GPUS=1

# Or with an explicit GPU
make config-gen-pretrain SIZE=350m GPUS=4 GPU=h200
make config-gen-pretrain SIZE=1b   GPUS=8 GPU=b200 MODE=aggressive
```

The generated YAML has a header comment listing the inputs used and predicted peak VRAM. Inspect the file before training if you want to verify the decisions. Re-running `config-gen` overwrites the file — copy it first if you've hand-edited.

**Step 4 — Pretrain**

```bash
# Mini validation run — confirm the training loop works before committing
make pretrain-mini GPUS=1

# Single GPU
make pretrain SIZE=125m GPUS=1

# Multi-GPU
make accelerate-config-multi GPUS=4
make config-gen-pretrain SIZE=125m GPUS=4    # re-tune for the GPU count
make pretrain            SIZE=125m GPUS=4

# Resume from last checkpoint
make pretrain-resume SIZE=125m GPUS=4
```

---

## Multi-GPU Config Scaling

The configs in `pretrain/configs/` are written assuming **1 GPU**. For multi-GPU training, scale `gradient_accumulation_steps` and `max_steps` to keep the global batch size and token budget constant:

```
global_batch_tokens = micro_batch_size × gradient_accumulation_steps × num_gpus × seq_len
```

When you change `num_gpus`, scale `gradient_accumulation_steps` inversely:

```
gradient_accumulation_steps_new = gradient_accumulation_steps_old × old_gpus / new_gpus
max_steps_new                   = max_steps_old × old_gpus / new_gpus
```

Per-size reference tables:

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

### Auto-generate the config (recommended)

`make config-gen-pretrain` does the math above automatically and also picks `micro_batch_size` and `gradient_checkpointing` based on the GPU model and count. The script reads the GPU model (auto-detected via `nvidia-smi`, or override with `GPU=...`), computes everything, and writes `pretrain/configs/gpt_$(SIZE).yaml`.

```bash
make config-gen-pretrain SIZE=125m GPUS=1                     # auto-detect GPU
make config-gen-pretrain SIZE=350m GPUS=4 GPU=h200            # explicit GPU
make config-gen-pretrain SIZE=1b   GPUS=8 GPU=b200 MODE=aggressive  # 90% VRAM budget

make pretrain SIZE=125m GPUS=1
```

The generated YAML has a header comment listing the inputs used and predicted peak VRAM. Inspect the file before training if you want to verify the decisions. Re-running overwrites the file — copy it first if you've hand-edited.

What `config-gen-pretrain` decides automatically across the matrix:

| Run | micro × accum × gpus | global | ckpt | max_steps |
|---|---|---|---|---|
| 125m on H200 × 1 | 32 × 1 × 1 | 32 | off | 152,587 |
| 125m on H200 × 8 | 4 × 1 × 8 | 32 | off | 152,587 |
| 350m on H200 × 1 | 128 × 1 × 1 | 128 | off | 114,440 |
| 350m on H200 × 4 | 32 × 1 × 4 | 128 | off | 114,440 |
| 1b on H200 × 1 | 128 × 1 × 1 | 128 | on | 57,220 |
| 1b on H200 × 8 | 16 × 1 × 8 | 128 | off | 57,220 |
| 1b on B200 × 1 | 32 × 4 × 1 | 128 | off | 57,220 |

### Tuning modes

Three modes control how aggressively the script packs the GPU:

| Mode | VRAM budget | Notes |
|---|---|---|
| `conservative` | 70% | Leaves headroom; safer on preemptible VMs and unfamiliar hardware |
| `balanced` *(default)* | 80% | Comfortable margin |
| `aggressive` | 90% | Maximum throughput; tolerates non-power-of-2 micro_batch |

```bash
make config-gen-pretrain SIZE=1b GPUS=1 MODE=aggressive
make config-gen-pretrain SIZE=1b GPUS=1 MODE=conservative
```

### Override knobs

```bash
# Force gradient checkpointing on or off
.venv/bin/python -m config_gen.config_gen --stage pretrain --gpu h200 --size 1b --gpus 1 \
    --no-ckpt -o pretrain/configs/gpt_1b.yaml

# Override target global batch
.venv/bin/python -m config_gen.config_gen --stage pretrain --gpu h200 --size 350m --gpus 4 \
    --target-global-batch 256 -o pretrain/configs/gpt_350m.yaml
```

Supported GPUs: `h200`, `b200`, `h100`, `h100_sxm`, `a100_80`, `a100_40`, `l40s`, `rtx4090`, `rtx5090`. To add a new one, edit `GPU_SPECS` in `config_gen/config_gen.py` and re-run.

To extend to a new model size: add a `SIZE_PROFILES` entry in `config_gen/config_gen.py` with the architecture, target tokens, reference global batch, and measured per-sequence activation memory.

---

## Configs

Token targets match `TARGET_CONFIGS` in `curator/scripts/curate.py`. Pretrain config values shown below are auto-generated by `make config-gen`.

| Config | Model | Layers | Hidden | Reference global | Corpus × Epochs |
|---|---|---|---|---|---|
| `gpt_mini.yaml` | `slm-mini` | 6 | 384 | 8 | validation only |
| `gpt_125m.yaml` | `slm-125m` | 12 | 768 | 32 | 5B × 2 |
| `gpt_350m.yaml` | `slm-350m` | 24 | 1024 | 128 | 15B × 2 |
| `gpt_1b.yaml` | `slm-1b` | 32 | 2048 | 128 | 30B × 1 |

Reference global = global batch in sequences. Multiplied by `seq_len` it gives tokens-per-step.

`gpt_mini.yaml` is hand-written and not regenerated — the mini run is fixed by definition (it's the smoke test for the training loop). All other configs are intended to be regenerated by `make config-gen` whenever you change GPU.

---

## Files

```
pretrain/
├── configs/
│   ├── gpt_mini.yaml        training config for mini validation run (hand-written)
│   ├── gpt_125m.yaml        training config for 125M (auto-generated)
│   ├── gpt_350m.yaml        training config for 350M (auto-generated)
│   └── gpt_1b.yaml          training config for 1B (auto-generated)
├── data/
│   ├── tokenize_data.py     JSONL → memory-mapped uint16 binary (train + val)
│   ├── upload_tokenized.py  S3 upload/download for tokenized binaries
│   └── dataset.py           PyTorch Dataset wrapping each .bin file
└── train.py                 HF Trainer pretraining entry point
```

---

## Throughput Optimizations

`train.py` enables several PyTorch performance flags at startup. These run unconditionally on GPU (no-op on CPU) and are not exposed as config options because they are pure wins on H100/H200/B200 with no quality downside:

- **TF32** for `cuda.matmul` and `cudnn` — large speedup on Hopper/Ampere/Blackwell over FP32 fallback at no measurable quality cost.
- **FlashAttention SDPA** and **memory-efficient SDPA** — explicitly enabled. The model's GQA module dispatches attention through `F.scaled_dot_product_attention`, which routes to FlashAttention when shapes and dtypes allow.
- **Math SDPA** — explicitly disabled. The math fallback is dramatically slower; turning it off forces an error on the rare path where neither fast kernel can be used, instead of silently degrading.

`build_training_args` also opts into:

- **`adamw_torch_fused`** — fused CUDA AdamW kernel; ~5-10% step-time win on Hopper+ for free. Falls back to plain `adamw_torch` on CPU.
- **`torch.compile`** (default on, opt-out via `torch_compile: false`) — graph-compiles the forward pass on the first step. Adds ~1-2 minutes to step 1, then runs ~1.3-1.5x faster every step after. Static shapes (fixed `seq_len`, fixed `micro_batch`) make this safe; if you hit a kernel issue while debugging, set `torch_compile: false` in the YAML.

A startup log line prints the active settings:

```
Throughput knobs: micro_batch=32, grad_accum=1, bf16=True,
                  optim=adamw_torch_fused, compile=True, grad_ckpt=False
```

Use this to confirm the settings reached `TrainingArguments` without trawling W&B.

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

Expected validation loss at convergence: **2.5–3.5** for `slm-125m` after 2 epochs over the 5B-token corpus

In a separate shell during the first ~200 steps of a fresh run, watch GPU utilization to confirm the auto-generated config is actually saturating the device:

```bash
nvidia-smi dmon -s pucm -d 2
```

Healthy signs: GPU-Util ≥ 95% sustained, memory usage near the estimate in the YAML header comment, power close to TDP. If util is dipping or memory is far below the estimate, see the troubleshooting notes in `config_gen/config_gen.py`.

---

## Multi-GPU Training

Multi-GPU training is configured via `accelerate_configs/` and launched through `make`. No code changes are needed — accelerate handles distributed training transparently.

```bash
# Configure accelerate for single GPU (mini validation)
make accelerate-config-single

# Configure accelerate for multi-GPU (full training)
make accelerate-config-multi GPUS=4

# Re-tune config for the GPU count, then launch
make config-gen-pretrain SIZE=125m GPUS=4
make pretrain            SIZE=125m GPUS=4

make config-gen-pretrain SIZE=350m GPUS=8
make pretrain            SIZE=350m GPUS=8

# 1b on multi-GPU benefits from FSDP — frees ~10 GB/GPU on optimizer state
make accel-gen-fsdp GPUS=8
make config-gen-pretrain SIZE=1b GPUS=8
make pretrain            SIZE=1b GPUS=8

# Override config directly (skips config-gen — use only if hand-tuning)
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=4
```

`config-gen-pretrain` keeps the global batch (and token budget) constant across GPU counts, so the model and recipe are identical at 1× H200 vs 8× H200 — only wall clock changes. No need to manually rescale `gradient_accumulation_steps` and `max_steps`.

---

## Hardware Estimates

| Model | GPU | Precision | Global batch | Est. tokens/sec | Est. time |
|---|---|---|---|---|---|
| `slm-125m` | 1× H200 | bf16 | 32 | ~1M (target) | ~3-4 hrs (target, with `make config-gen`) |
| `slm-125m` | 8× H200 | bf16 | 32 | _TBD_ | _TBD_ |
| `slm-350m` | 8× H200 | bf16 | 128 | _TBD — pending 350m run_ | _TBD_ |
| `slm-1b` | 8× H200 | bf16 | 128 | _TBD — pending 1b run_ | _TBD_ |

Actual throughput depends on GPU generation, network topology (multi-node), CPU/GPU data-loading balance, and the specific cloud region. Measure your own throughput with `make pretrain-mini` before committing to a full run.

---

## Key Design Decisions

**Why memory-mapped binary?** Tokenizing on the fly during training adds significant CPU overhead and reduces GPU utilization. Pre-tokenizing once and loading with `np.memmap` gives near-instant data loading with constant memory usage regardless of dataset size.

**Why uint16?** Our 32k vocab fits in uint16 (max 65,535), halving the storage and memory bandwidth compared to int32.The `train.bin` for the 5B-token corpus is ~10GB in uint16 vs ~20GB in int32. `tokenize_data.py` asserts this fits at startup so vocab growth past the limit fails loudly instead of silently corrupting the training data.

**Why val split produced by the curator, not at training time?** Two correctness bugs with a runtime split: (1) split files could become stale if `train.bin` was regenerated without noticing, silently using old val data; (2) the tail-of-stream slice is only an unbiased sample if the shuffle has perfectly uniform ordering — the disk-chunked shuffle used at 1b scale doesn't quite have that property. Splitting at blend time, after shuffle, eliminates both issues by construction. Val and train are tokenized independently and the dataset code is a thin wrapper over each binary.

**Why cosine LR schedule?** Cosine annealing smoothly decays the learning rate to a small minimum, giving the model time to converge without abrupt changes. Used by GPT-3, LLaMA, and most modern pretraining runs.

**Why 2 epochs at 125m and 350m, but 1 epoch at 1b?** t's about how the corpus size compares to the supply of each source. At 125m (5B corpus tokens) and 350m (15B corpus tokens) the corpus is small enough that 2 epochs fits comfortably within every source's supply — repeating the data improves downstream performance with negligible overfitting risk at this scale. At 1b (30B corpus tokens, 1 epoch) every source is already close to its supply ceiling, so no repetition is needed; adding a second epoch would force either per-source overflow or a lower-quality over-epoching loop. Modern small-model training (Llama, Phi, Qwen) follows the same pattern at scale — fresh tokens outperform repeated ones once the budget is big enough.

**Why auto-generate pretrain configs instead of committing them?** A single hand-written `gpt_125m.yaml` cannot be right for both 1× H200 and 8× A100 at the same time — `micro_batch_size` that fits in 141GB underuses 8× larger memory pools, and `gradient_accumulation_steps` written for 1 GPU need to scale with GPU count to preserve the global batch. The previous approach was a comment in the YAML listing 1/4/8-GPU values that the user had to manually swap before running. That works exactly until someone forgets, at which point the run silently trains on the wrong number of tokens. `config_gen/config_gen.py` makes the math automatic and keyed off the actual hardware. The recipe (LR, schedule, betas) is intentionally NOT in the script's scope — those are model-size decisions, not hardware decisions, and stay constant across GPUs.

**Why does `config-gen` sometimes enable gradient checkpointing on H200?** The auto-policy turns checkpointing on when activations don't fit without it (forced) OR when checkpointing unlocks ≥ 4× larger micro-batch and the no-ckpt config can't reach the target global batch in a single optimizer step. The second case shows up at 1b on H200 × 1: without ckpt, max micro is ~31, forcing 4-step accumulation; with ckpt, all 128 sequences fit in one step. Eliminating the accumulation overhead can outweigh the recompute cost. If you want to A/B it for your specific GPU and batch size, run `make config-gen ... AGGRESSIVE=1` or call `config_gen/config_gen.py --no-ckpt` directly.

**Why baseline eval before training?** So the W&B eval curve starts at step 0 with random-init loss. Without this, the first eval point is `eval_steps` steps in and you can't see how much the model has actually learned. The baseline is skipped on `--resume` since the original run already has one.

**Why validate the tokenizer (and parse the JSON) before training?** A missing or truncated `tokenizer_config.json` won't affect pretraining loss but makes the saved checkpoint unusable downstream — `train_sft.py` requires it to load the chat template. A corrupt file that passes a naive file-existence check crashes much later. Parsing the JSON up front — not just checking presence — turns a late failure into an early one.

**Why pre-count documents before tokenizing?** `tokenize_data.py` reads the input file once to count documents, then again to tokenize. The first pass is O(lines) and costs ~30 seconds at 1b scale; the payoff is an accurate ETA in the tqdm progress bar during the much longer tokenization run. Worth trading a small one-time cost for actionable progress reporting on multi-hour jobs.