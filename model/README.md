# Model

## Purpose

`model/` implements the train-time SLM decoder architecture. It owns model
construction, causal language-model loss, attention masks, RoPE, grouped-query
attention, KV-cache behavior, and Hugging Face generation integration. Training
configuration, data loading, and native export live in their respective stages.

## Contents

```text
model/
├── attention.py   RoPE and grouped-query self-attention
├── block.py       pre-normalized decoder block
├── config.py      SLMConfig and predefined model profiles
├── mlp.py         SwiGLU feed-forward network
├── model.py       SLMModel and SLMForCausalLM
└── norm.py        RMSNorm
```

## How It Fits In

Pretraining constructs `SLMForCausalLM` from the selected YAML profile. SFT and
DPO reload the resulting checkpoints through the same class. Evaluation and
local inference register `SLMConfig` with `AutoModelForCausalLM`. The export
stage maps the compatible weights and configuration into a native
`LlamaForCausalLM` artifact.

## Architecture

| Component | Implementation |
|---|---|
| Objective | decoder-only causal language modeling |
| Decoder | pre-norm attention and MLP residual blocks |
| Position encoding | RoPE, computed in float32 and shared across layers |
| Attention | PyTorch SDPA with grouped-query attention |
| Normalization | RMSNorm |
| Feed-forward network | bias-free SwiGLU |
| Embeddings | tied token embedding and LM-head weights |
| Generation | Transformers `Cache` support and legacy-cache conversion |
| Dropout | configurable attention dropout; zero in the supplied profiles |

The attention layer accepts only the SDPA implementation. CUDA runtime setup
leaves Flash, memory-efficient, cuDNN, and math SDPA kernels enabled so PyTorch
can select the valid implementation for each shape and mask.

## Configured Profiles

Counts are unique trainable parameters; tied embeddings are counted once.

| Size | Parameters | Layers | Hidden | Intermediate | Q heads | KV heads | Context |
|---|---:|---:|---:|---:|---:|---:|---:|
| `smoke` | 21.7M | 6 | 384 | 1,024 | 6 | 2 | 1,024 |
| `mini` | 69.9M | 17 | 512 | 1,536 | 8 | 4 | 2,048 |
| `125m` | 125.3M | 16 | 768 | 2,048 | 12 | 4 | 2,048 |
| `350m` | 351.3M | 27 | 1,024 | 2,816 | 16 | 8 | 2,048 |
| `1b` | 1.012B | 21 | 2,048 | 5,632 | 32 | 8 | 4,096 |

`pretrain/configs/` is the training source of truth for complete model
profiles. `SLM_SMOKE`, `SLM_MINI`, `SLM_125M`, `SLM_350M`, and `SLM_1B` in
`config.py` provide matching programmatic defaults.

## API

Construct a model directly:

```python
from model import SLMConfig, SLMForCausalLM

config = SLMConfig(
    vocab_size=32_000,
    hidden_size=768,
    intermediate_size=2_048,
    num_hidden_layers=16,
    num_attention_heads=12,
    num_key_value_heads=4,
    max_position_embeddings=2_048,
    rope_theta=500_000.0,
)
model = SLMForCausalLM(config)
```

Load a local SLM checkpoint:

```python
from transformers import AutoConfig, AutoModelForCausalLM
from model import SLMConfig, SLMForCausalLM

AutoConfig.register("slm", SLMConfig)
AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

model = AutoModelForCausalLM.from_pretrained(
    "results/runs/125m/pretrain/final"
)
```

## Constraints

- `hidden_size` must divide evenly by `num_attention_heads`.
- `num_attention_heads` must divide evenly by `num_key_value_heads`.
- Attention head dimensions must be even for RoPE.
- Context scaling is not implemented; `rope_scaling` must remain `None`.
- Gradient checkpointing disables the generation cache while training.
- `SLMModel` is the internal `nn.Module`; use `SLMForCausalLM` for training,
  saving, loading, and generation.

## Tests

```bash
make test-model
```

The model suite covers construction, parameter counts, causal and padding
masks, cached decoding, generation parity, validation errors, and checkpoint
round trips.
