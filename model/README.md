# model

Custom decoder-only transformer architecture registered as a HuggingFace `PreTrainedModel`. Designed from first principles — every architectural decision is explicit and documented.

---

## Architecture

| Component | Choice | File |
|---|---|---|
| Positional encoding | RoPE | `attention.py` |
| Normalization | RMSNorm | `norm.py` |
| Activation | SwiGLU | `mlp.py` |
| Attention | GQA | `attention.py` |
| Bias | None | all |
| Embeddings | Tied | `model.py` |

---

## Model Sizes

| Model | Layers | Hidden | Intermediate | Q heads | KV heads | Context | Parameters |
|---|---|---|---|---|---|---|---|
| `slm-125m` | 12 | 768 | 2048 | 12 | 4 | 2048 | ~125M |
| `slm-350m` | 24 | 1024 | 2816 | 16 | 8 | 2048 | ~350M |
| `slm-1b` | 32 | 2048 | 5632 | 32 | 8 | 4096 | ~1B |

---

## Files

```
model/
├── config.py       SLMConfig(PretrainedConfig) — hyperparameters and predefined configs
├── norm.py         RMSNorm — root mean square normalization
├── attention.py    GroupedQueryAttention + RotaryEmbedding — GQA with RoPE
├── mlp.py          SwiGLUMLP — gated feed-forward network
├── block.py        SLMDecoderBlock — pre-norm transformer block
└── model.py        SLMModel + SLMForCausalLM — full model registered with HuggingFace
```

---

## Usage

**Instantiate from a predefined config:**

```python
from model import SLMForCausalLM, SLM_125M

model = SLMForCausalLM(SLM_125M)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Register with HuggingFace AutoModel:**

```python
from transformers import AutoConfig, AutoModelForCausalLM
from model import SLMConfig, SLMForCausalLM

AutoConfig.register("slm", SLMConfig)
AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

# Now loadable from Hub
model = AutoModelForCausalLM.from_pretrained("tohio/slm-125m")
```

**Save and load:**

```python
model.save_pretrained("checkpoints/slm-125m")
model = SLMForCausalLM.from_pretrained("checkpoints/slm-125m")
```

**Forward pass:**

```python
import torch

input_ids = torch.randint(0, 32000, (1, 128))
output = model(input_ids, labels=input_ids)
print(f"Loss: {output.loss.item():.4f}")
```

**Generate text:**

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
)
```

---

## Design Decisions

**RoPE over learned absolute embeddings** — RoPE encodes position information directly into the attention dot product via rotation, preserving relative position awareness without adding parameters. Generalizes better to sequence lengths not seen during training.

**RMSNorm over LayerNorm** — removes the mean subtraction step, keeping only the RMS scaling. Computationally cheaper and empirically equivalent in quality. Upcasts to float32 internally for numerical stability during bfloat16 training.

**SwiGLU over GELU** — the gating mechanism in SwiGLU provides an additional learned control over information flow through the FFN. Uses three projections (`gate_proj`, `up_proj`, `down_proj`) rather than two, but intermediate size is scaled down by `8/3` so total parameter count matches a standard 4x FFN.

**GQA over MHA** — KV cache is the primary memory bottleneck at inference. GQA reduces KV heads (4 for 125M, 8 for 350M/1B) while keeping full query heads. At 125M this gives a 3x reduction in KV memory with negligible quality loss. Directly improves throughput in vLLM.

**No bias** — bias terms add parameters without meaningfully improving performance in large transformer models. Removing them simplifies the model and reduces memory slightly.

**Tied embeddings** — the LM head weight matrix is shared with the input embedding matrix. At small model scale this regularizes the model, reduces parameters by `vocab_size × hidden_size`, and has been shown to improve perplexity.

**Pre-norm** — RMSNorm is applied before each sub-layer rather than after. Pre-norm significantly improves training stability in deep networks by keeping gradient magnitudes consistent across layers, making learning rate tuning more predictable.

**`F.scaled_dot_product_attention`** — PyTorch's built-in SDPA dispatches to FlashAttention 2 automatically when available (requires `torch >= 2.0` and a compatible GPU). No extra dependency or custom CUDA kernel needed.