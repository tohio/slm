# finetune

Supervised Fine-Tuning (SFT) pipeline for SLM. Two sequential stages — chat then code — across all three model sizes, using HuggingFace `trl SFTTrainer`.

---

## Pipeline

```
results/slm-{size}/final          (pretrained base)
        │
        ▼
Stage 1: Chat SFT (OpenHermes-2.5)
        │
        ▼
results/slm-{size}-chat/final
        │
        ▼
Stage 2: Code SFT (Magicoder-OSS-Instruct)
        │
        ▼
results/slm-{size}-chat-code/final
```

---

## Datasets

| Stage | Dataset | Size | Purpose |
|---|---|---|---|
| Chat | `teknium/OpenHermes-2.5` | ~1M examples | General instruction following |
| Code | `ise-uiuc/Magicoder-OSS-Instruct-75K` | ~75k examples | Code generation and understanding |

---

## Getting Started

**Step 1 — Prepare data**

```bash
make prepare-sft

# Or directly
python finetune/data/prepare_sft.py --stage both
python finetune/data/prepare_sft.py --stage chat
python finetune/data/prepare_sft.py --stage code
```

Per-stage defaults: chat uses `val_fraction=0.02`, code uses `0.05`. Override both with `--val-fraction`.

**Step 2 — Chat SFT**

```bash
# 125M
python finetune/train_sft.py --config finetune/configs/sft_chat_125m.yaml

# 350M
python finetune/train_sft.py --config finetune/configs/sft_chat_350m.yaml

# 1B
python finetune/train_sft.py --config finetune/configs/sft_chat_1b.yaml

# Multi-GPU
accelerate launch finetune/train_sft.py --config finetune/configs/sft_chat_125m.yaml

# Resume
python finetune/train_sft.py --config finetune/configs/sft_chat_125m.yaml --resume
```

**Step 3 — Code SFT**

```bash
python finetune/train_sft.py --config finetune/configs/sft_code_125m.yaml
python finetune/train_sft.py --config finetune/configs/sft_code_350m.yaml
python finetune/train_sft.py --config finetune/configs/sft_code_1b.yaml
```

---

## Files

```
finetune/
├── configs/
│   ├── sft_chat_125m.yaml    chat SFT — 125M, LR=1e-5
│   ├── sft_chat_350m.yaml    chat SFT — 350M, LR=8e-6
│   ├── sft_chat_1b.yaml      chat SFT — 1B,   LR=5e-6
│   ├── sft_chat_mini.yaml    chat SFT — mini pipeline smoke test
│   ├── sft_code_125m.yaml    code SFT — 125M, LR=5e-6
│   ├── sft_code_350m.yaml    code SFT — 350M, LR=3e-6
│   ├── sft_code_1b.yaml      code SFT — 1B,   LR=2e-6
│   └── sft_code_mini.yaml    code SFT — mini pipeline smoke test
├── data/
│   └── prepare_sft.py        download and format both datasets
└── train_sft.py              trl SFTTrainer entry point
```

---

## Config Summary

| Config | Base model | LR | Epochs | Micro batch | Grad accum | Seq len | Grad ckpt |
|---|---|---|---|---|---|---|---|
| `sft_chat_125m` | `slm-125m/final` | 1e-5 | 2 | 4 | 4 | 2048 | No |
| `sft_chat_350m` | `slm-350m/final` | 8e-6 | 2 | 2 | 8 | 2048 | No |
| `sft_chat_1b` | `slm-1b/final` | 5e-6 | 2 | 1 | 16 | 4096 | Yes |
| `sft_code_125m` | `slm-125m-chat/final` | 5e-6 | 2 | 4 | 4 | 2048 | No |
| `sft_code_350m` | `slm-350m-chat/final` | 3e-6 | 2 | 2 | 8 | 2048 | No |
| `sft_code_1b` | `slm-1b-chat/final` | 2e-6 | 2 | 1 | 16 | 4096 | Yes |

All configs use effective global batch size of 128 and `warmup_ratio=0.03` on a cosine schedule. Grad-accum values shown are for 1 GPU; each config includes scaling comments for 4/8 GPU.

---

## Chat Template

All data is formatted into the SLM chat template before training:

```
<|system|>You are a helpful assistant.<|endofturn|>
<|user|>What is the capital of France?<|endofturn|>
<|assistant|>The capital of France is Paris.<|endofturn|>
```

The template is applied automatically by `SFTTrainer` via `tokenizer.apply_chat_template()` — `prepare_sft.py` only emits structured `conversations` records, never pre-formatted text. This is required for `assistant_only_loss=True`, which depends on the `{% generation %}` tags baked into the tokenizer's chat template to mask prompt tokens from the loss.

---

## Checkpoints

```
results/
├── slm-125m-chat/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── final/                best checkpoint (lowest eval loss)
├── slm-125m-chat-code/
│   └── final/
├── slm-350m-chat/
│   └── final/
...
```

`final/` contains the **lowest-eval-loss** checkpoint. This is enforced in `train_sft.py` by `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"`. Because HF Trainer always keeps the best checkpoint in addition to the N most recent (`save_total_limit=3`), disk usage is up to 4 checkpoints per run during training.

---

## Key Design Decisions

**Why sequential SFT?** Training code SFT on top of the chat checkpoint preserves instruction-following capability learned in stage 1. The lower LR in code SFT further reduces catastrophic forgetting.

**Why 2 epochs?** With ~1M OpenHermes examples and ~75k Magicoder examples, 2 epochs is enough for strong capability gain on GPT-4-generated data without triggering memorization or forgetting of pretraining knowledge — especially at 1B where 3 epochs is consistently over-trained in our runs.

**Why `warmup_ratio=0.03`?** A ratio scales cleanly with both dataset size and epoch count; the previous fixed `warmup_steps: 100` was ~0.3% of one epoch at 125M and effectively no warmup at 1B, which caused LR to hit peak before the optimizer stabilized.

**Why OpenHermes-2.5?** One of the highest-quality open instruction datasets — generated by GPT-4 with careful filtering. 1M diverse examples covering reasoning, coding, creative writing, and general knowledge.

**Why Magicoder?** Generates coding problems inspired by real open-source code then produces solutions. More diverse and higher quality than CodeAlpaca. 75k examples is sufficient for strong code SFT at all three model scales.

**Why LR decreases with model size?** Larger models are more sensitive to fine-tuning — a high LR causes catastrophic forgetting of pretraining knowledge. The conservative LRs for 350M and 1B maintain stability.

**Why gradient checkpointing only for 1B?** At 125M and 350M, activations fit comfortably in H100 memory. At 1B with seq_len=4096, activation memory becomes the bottleneck.