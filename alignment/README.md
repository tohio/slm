# alignment

Direct Preference Optimization (DPO) pipeline for SLM. Aligns the SFT model to human preferences using a blended dataset of three complementary preference sources.

---

## Pipeline

```
results/slm-{size}-chat-code/final   (SFT model)
        │
        ▼
DPO training (hh-rlhf + orca + argilla)
        │
        ▼
results/slm-{size}-dpo/final         (aligned model)
```

---

## Datasets

| Dataset | Size | Signal |
|---|---|---|
| `Anthropic/hh-rlhf` | ~170k pairs | Human preference — helpfulness + harmlessness |
| `Intel/orca_dpo_pairs` | ~30k pairs | Synthetic — GPT-4 vs GPT-3.5 on reasoning tasks |
| `argilla/dpo-mix-7k` | ~7k pairs | Curated high quality mix |

---

## Getting Started

**Step 1 — Prepare data**

```bash
make prepare-dpo

# Or directly
python alignment/data/prepare_dpo.py

# Single source
python alignment/data/prepare_dpo.py --source hh-rlhf
python alignment/data/prepare_dpo.py --source orca
python alignment/data/prepare_dpo.py --source argilla
```

**Step 2 — DPO training**

```bash
# 125M
python alignment/train_dpo.py --config alignment/configs/dpo_125m.yaml

# 350M
python alignment/train_dpo.py --config alignment/configs/dpo_350m.yaml

# 1B
python alignment/train_dpo.py --config alignment/configs/dpo_1b.yaml

# Multi-GPU
accelerate launch alignment/train_dpo.py --config alignment/configs/dpo_125m.yaml

# Resume
python alignment/train_dpo.py --config alignment/configs/dpo_125m.yaml --resume
```

---

## Files

```
alignment/
├── configs/
│   ├── dpo_125m.yaml     DPO config — 125M, LR=5e-7, beta=0.1
│   ├── dpo_350m.yaml     DPO config — 350M, LR=3e-7, beta=0.1
│   └── dpo_1b.yaml       DPO config — 1B,   LR=2e-7, beta=0.1
├── data/
│   └── prepare_dpo.py    download and blend preference datasets
└── train_dpo.py          trl DPOTrainer entry point
```

---

## Config Summary

| Config | Base model | LR | Beta | Micro batch | Grad accum | Epochs |
|---|---|---|---|---|---|---|
| `dpo_125m` | `slm-125m-chat-code/final` | 5e-7 | 0.1 | 2 | 8 | 1 |
| `dpo_350m` | `slm-350m-chat-code/final` | 3e-7 | 0.1 | 1 | 16 | 1 |
| `dpo_1b` | `slm-1b-chat-code/final` | 2e-7 | 0.1 | 1 | 16 | 1 |

---

## DPO Data Format

Each record in `data/dpo/train.jsonl`:

```json
{
    "prompt":   "<|system|>...<|user|>...<|endofturn|><|assistant|>",
    "chosen":   "The preferred assistant response",
    "rejected": "The non-preferred assistant response",
    "source":   "hh-rlhf | orca | argilla"
}
```

---

## Checkpoints

```
results/
├── slm-125m-dpo/
│   ├── checkpoint-200/
│   └── final/
├── slm-350m-dpo/
│   └── final/
└── slm-1b-dpo/
    └── final/
```

---

## Key Design Decisions

**Why DPO over PPO?** At small model scale, PPO's actor-critic setup requires simultaneously loading the policy, reference, reward, and value models — at least 4× the memory of a single model. DPO achieves comparable alignment with a single model and no reward model, making it tractable on a single GPU.

**Why beta=0.1?** Beta controls how far the model is allowed to deviate from the SFT reference. Lower beta = more alignment, less fluency. Higher beta = less alignment, better fluency. 0.1 is the standard starting point used across most DPO papers and production runs.

**Why LR in the 1e-7 range?** DPO is extremely sensitive to learning rate — too high and the model collapses, too low and the alignment signal doesn't propagate. The 1e-7 range is significantly lower than SFT and has been validated across multiple open DPO runs.

**Why 1 epoch?** DPO over-optimizes quickly — after one epoch the reward margin typically saturates and further training degrades model quality. Most production DPO runs use 1–2 epochs.

**Why blend three sources?** Each dataset provides a different alignment signal. hh-rlhf provides broad human preference coverage. Orca provides high-quality reasoning preference signal. Argilla provides a carefully curated quality baseline. The blend reduces dataset-specific biases.

**Why start from chat-code?** DPO learns preference signal on top of the SFT distribution. Starting from a well-trained SFT model with both chat and code capability ensures the alignment generalizes across both domains.