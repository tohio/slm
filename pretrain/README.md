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
pip install -r requirements.txt
# Validated dataset must exist
ls data/validated/train.jsonl
# Tokenizer must be trained
ls data/tokenizer/slm_tokenizer.json
```

**Step 1 — Tokenize**

```bash
# Default: data/validated/train.jsonl → data/tokenized/train.bin
make tokenize

# Or directly
python pretrain/data/tokenize.py --workers 8 --verify
```

**Step 2 — Pretrain**

```bash
# Single GPU — 125M
make pretrain

# Single GPU — 350M or 1B
make pretrain SIZE=350m
make pretrain SIZE=1b

# Multi-GPU via accelerate
accelerate launch pretrain/train.py --config pretrain/configs/gpt_125m.yaml

# Resume from checkpoint
python pretrain/train.py --config pretrain/configs/gpt_125m.yaml --resume
```

---

## Configs

| Config | Model | Layers | Hidden | Steps | Global batch | Target tokens |
|---|---|---|---|---|---|---|
| `gpt_125m.yaml` | `slm-125m` | 12 | 768 | 150k | 32 | ~3B |
| `gpt_350m.yaml` | `slm-350m` | 24 | 1024 | 500k | 64 | ~10B |
| `gpt_1b.yaml` | `slm-1b` | 32 | 2048 | 1.5M | 128 | ~25B |

Global batch size = `micro_batch_size × gradient_accumulation_steps × num_gpus`.

---

## Files

```
pretrain/
├── configs/
│   ├── gpt_125m.yaml        training config for 125M
│   ├── gpt_350m.yaml        training config for 350M
│   └── gpt_1b.yaml          training config for 1B
├── data/
│   ├── tokenize.py          JSONL → memory-mapped uint16 binary
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
| `train/tokens_per_second` | GPU throughput — should be stable after warmup. |

Expected validation perplexity at convergence: **20–40** for `slm-125m` at 3B tokens.

---

## Multi-GPU Training

Uses HuggingFace `accelerate` — no code changes needed, just launch differently:

```bash
# Configure accelerate (run once)
accelerate config

# Launch on all available GPUs
accelerate launch pretrain/train.py --config pretrain/configs/gpt_125m.yaml

# Launch on specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
    pretrain/train.py --config pretrain/configs/gpt_350m.yaml
```

---

## Hardware Estimates

| Model | GPU | Precision | Batch | Est. tokens/sec | Est. time (3B tokens) |
|---|---|---|---|---|---|
| `slm-125m` | 1× H100 80GB | bf16 | 32 | ~250k | ~3.5 hrs |
| `slm-125m` | 1× A100 40GB | bf16 | 32 | ~120k | ~7 hrs |
| `slm-350m` | 4× H100 80GB | bf16 | 64 | ~400k | ~7 hrs (10B) |
| `slm-1b` | 4× H100 80GB | bf16 | 128 | ~200k | ~35 hrs (25B) |

---

## Key Design Decisions

**Why memory-mapped binary?** Tokenizing on the fly during training adds significant CPU overhead and reduces GPU utilization. Pre-tokenizing once and loading with `np.memmap` gives near-instant data loading with constant memory usage regardless of dataset size.

**Why uint16?** Our 32k vocab fits in uint16 (max 65,535), halving the storage and memory bandwidth compared to int32. The `train.bin` for 3B tokens is ~6GB in uint16 vs ~12GB in int32.

**Why cosine LR schedule?** Cosine annealing smoothly decays the learning rate to a small minimum, giving the model time to converge without abrupt changes. Used by GPT-3, LLaMA, and most modern pretraining runs.

**Why global batch 32/64/128?** Larger batches provide more stable gradient estimates and allow higher learning rates, improving convergence. Gradient accumulation steps simulate larger batches on single GPUs.

**Why gradient checkpointing for 1B only?** At 125M and 350M, activations fit comfortably in H100 memory. At 1B with sequence length 4096, activation memory becomes the bottleneck. Gradient checkpointing trades ~30% compute for ~60% memory reduction.