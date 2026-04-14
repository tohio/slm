# pretrain

Pretraining pipeline for SLM. Tokenizes the validated dataset into a memory-mapped binary file and trains `SLMForCausalLM` from scratch using HuggingFace `Trainer`.

---

## Pipeline

```
data/validated/train.jsonl
        │
        ▼
pretrain/data/tokenize.py   →   data/tokenized/train.bin  (uint16, flat token array)
        │
        ▼
pretrain/train.py           →   results/slm-125m/final/   (HF checkpoint)
```

---

## Getting Started

**Prerequisites**

```bash
# Validated dataset must exist at DATA_DIR
ls $DATA_DIR/validated/train.jsonl

# Tokenizer must be trained
ls $DATA_DIR/tokenizer/tokenizer.json
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

| GPUs | gradient_accumulation_steps | max_steps | Global batch (125m) |
|---|---|---|---|
| 1 | 8 | 150,000 | 32 seqs |
| 4 | 2 | 37,500 | 32 seqs |
| 8 | 1 | 18,750 | 32 seqs |

Each config file includes a scaling comment with the exact values for 1, 4, and 8 GPUs. Copy the appropriate values before launching.

---

## Configs

| Config | Model | Layers | Hidden | Steps (1 GPU) | Global batch | Target tokens |
|---|---|---|---|---|---|---|
| `gpt_mini.yaml` | `slm-mini` | 6 | 384 | 500 | 8 | validation only |
| `gpt_125m.yaml` | `slm-125m` | 12 | 768 | 150k | 32 | ~9.8B |
| `gpt_350m.yaml` | `slm-350m` | 24 | 1024 | 500k | 64 | ~65B |
| `gpt_1b.yaml` | `slm-1b` | 32 | 2048 | 1.5M | 128 | ~786B |

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
│   ├── tokenize_data.py     JSONL → memory-mapped uint16 binary
│   ├── upload_tokenized.py  S3 upload/download for tokenized binary
│   └── dataset.py           PyTorch Dataset wrapping the .bin file
└── train.py                 HF Trainer pretraining entry point
```

---

## Data Format

The tokenized dataset is a flat binary file of uint16 token IDs:

```
[doc1_tok1, ..., doc1_tokN, EOS, doc2_tok1, ..., doc2_tokM, EOS, ...]
```

Documents are concatenated with EOS as separator — no padding, no wasted tokens. Each training example is a fixed-length window sliced from this array.

---

## Checkpoints

Checkpoints are saved every `save_steps` steps to `results/<model_name>/`:

```
results/slm-125m/
├── checkpoint-1000/         HF checkpoint (weights + optimizer)
├── checkpoint-2000/
├── ...
└── final/                   final model after training completes
    ├── config.json
    ├── model.safetensors
    ├── tokenizer/
    └── ...
```

The `final/` directory is a complete, self-contained HF model directory — load with `AutoModelForCausalLM.from_pretrained("results/slm-125m/final")`.

---

## Monitoring

Training metrics are logged to Weights & Biases automatically. Key metrics to watch:

| Metric | What to look for |
|---|---|
| `train/loss` | Should decrease smoothly. Spikes are normal, persistent plateaus are not. |
| `eval/loss` | Should track training loss. Widening gap = overfitting. |
| `train/learning_rate` | Cosine decay — should ramp up then decay smoothly. |
| `train/grad_norm` | Should stay below `gradient_clip_val=1.0`. Persistent high norms = instability. |

Expected validation perplexity at convergence: **20–40** for `slm-125m` at 9.8B tokens.

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

Accelerate configs live in `accelerate_configs/single_gpu.yaml` and `accelerate_configs/multi_gpu.yaml`. Both are committed to the repo — no interactive wizard needed on each new instance.

---

## Hardware Estimates

| Model | GPU | Precision | Global batch | Est. tokens/sec | Est. time |
|---|---|---|---|---|---|
| `slm-125m` | 1× H200 | bf16 | 32 | ~350k | ~8 hrs (9.8B tokens) |
| `slm-125m` | 8× H200 | bf16 | 32 | ~2.4M | ~70 min (9.8B tokens) |
| `slm-350m` | 8× H200 | bf16 | 64 | ~800k | ~23 hrs (65B tokens) |
| `slm-1b` | 8× H200 | bf16 | 128 | ~300k | ~7 days (786B tokens) |

---

## Key Design Decisions

**Why memory-mapped binary?** Tokenizing on the fly during training adds significant CPU overhead and reduces GPU utilization. Pre-tokenizing once and loading with `np.memmap` gives near-instant data loading with constant memory usage regardless of dataset size.

**Why uint16?** Our 32k vocab fits in uint16 (max 65,535), halving the storage and memory bandwidth compared to int32. The `train.bin` for 9.8B tokens is ~19GB in uint16 vs ~38GB in int32.

**Why cosine LR schedule?** Cosine annealing smoothly decays the learning rate to a small minimum, giving the model time to converge without abrupt changes. Used by GPT-3, LLaMA, and most modern pretraining runs.

**Why global batch 32/64/128?** Larger batches provide more stable gradient estimates and allow higher learning rates, improving convergence. Gradient accumulation steps simulate larger batches on single GPUs without requiring more GPU memory.

**Why gradient checkpointing for 1B only?** At 125M and 350M, activations fit comfortably in H200 memory. At 1B with sequence length 4096, activation memory becomes the bottleneck. Gradient checkpointing trades ~30% compute for ~60% memory reduction.