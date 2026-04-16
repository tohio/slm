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
# Validated dataset must exist
ls $DATA_DIR/validated/train.jsonl

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

---

## Configs

Token targets match `TARGET_CONFIGS` in `curator/scripts/curate.py`.

| Config | Model | Layers | Hidden | Steps (1 GPU) | Global batch | Target tokens |
|---|---|---|---|---|---|---|
| `gpt_mini.yaml` | `slm-mini` | 6 | 384 | 500 | 8 | validation only |
| `gpt_125m.yaml` | `slm-125m` | 12 | 768 | 152k | 32 | 5B (2 epochs) |
| `gpt_350m.yaml` | `slm-350m` | 24 | 1024 | 230k | 64 | 15B (2 epochs) |
| `gpt_1b.yaml` | `slm-1b` | 32 | 2048 | 114k | 128 | 30B (2 epochs) |

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

## Tokenizer Validation

`train.py` validates the tokenizer directory before training starts and fails
immediately if anything is missing. This prevents discovering a broken checkpoint
after hours of training.

Required files:
- `tokenizer_config.json` — HuggingFace config including baked-in `chat_template`.
  Required by `train_sft.py` to load the tokenizer via `from_pretrained()`.
- `slm_tokenizer.json` — Raw BPE tokenizer used by `tokenize_data.py`.

If either is missing: `make tokenizer && make tokenizer-upload && make tokenizer-download`

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

---

## Monitoring

Training metrics are logged to Weights & Biases automatically. Key metrics to watch:

| Metric | What to look for |
|---|---|
| `train/loss` | Should decrease smoothly. Spikes are normal, persistent plateaus are not. |
| `eval/loss` | Should track training loss. Widening gap = overfitting. |
| `train/learning_rate` | Cosine decay — should ramp up then decay smoothly. |
| `train/grad_norm` | Should stay below `gradient_clip_val=1.0`. Persistent high norms = instability. |

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
| `slm-125m` | 1× H200 | bf16 | 32 | ~350k | ~4 hrs (5B tokens) |
| `slm-125m` | 8× H200 | bf16 | 32 | ~2.4M | ~35 min (5B tokens) |
| `slm-350m` | 8× H200 | bf16 | 64 | ~800k | ~5 hrs (15B tokens) |
| `slm-1b` | 8× H200 | bf16 | 128 | ~300k | ~28 hrs (30B tokens) |

---

## Key Design Decisions

**Why memory-mapped binary?** Tokenizing on the fly during training adds significant CPU overhead and reduces GPU utilization. Pre-tokenizing once and loading with `np.memmap` gives near-instant data loading with constant memory usage regardless of dataset size.

**Why uint16?** Our 32k vocab fits in uint16 (max 65,535), halving the storage and memory bandwidth compared to int32. The `train.bin` for 5B tokens is ~10GB in uint16 vs ~20GB in int32.

**Why cosine LR schedule?** Cosine annealing smoothly decays the learning rate to a small minimum, giving the model time to converge without abrupt changes. Used by GPT-3, LLaMA, and most modern pretraining runs.

**Why 2 epochs?** Modern small model research consistently shows that training beyond Chinchilla optimal — repeating the dataset — improves downstream task performance at inference time, with only marginal overfitting risk at this scale. Two epochs doubles the compute budget without requiring more data.

**Why gradient checkpointing for 1B only?** At 125M and 350M, activations fit comfortably in H200 memory. At 1B with sequence length 4096, activation memory becomes the bottleneck. Gradient checkpointing trades ~30% compute for ~60% memory reduction.

**Why validate the tokenizer before training?** A missing `tokenizer_config.json` won't affect pretraining loss but makes the saved checkpoint unusable downstream — `train_sft.py` requires it to load the chat template. Failing early at the start of pretraining is far cheaper than discovering the issue after training completes.