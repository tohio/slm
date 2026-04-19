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

The `intermediate_size` column is computed from `hidden_size` via the LLaMA formula: `round(8/3 × hidden_size)` rounded up to the nearest multiple of 256. The `8/3` factor compensates for SwiGLU's three projections so the total parameter count matches a standard FFN with 4× expansion. If you pass `intermediate_size=None` (the default), `SLMConfig` computes it automatically; passing an explicit value overrides.

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

## Why two classes?

`model.py` defines two classes — `SLMModel` and `SLMForCausalLM` — and the split isn't obvious from the code alone.

**`SLMModel`** is the backbone: embeddings, decoder layers, final norm. It takes token IDs and returns hidden-state vectors. No head. No loss.

**`SLMForCausalLM`** wraps `SLMModel` and adds a linear LM head projecting hidden states to vocabulary logits. It also adds the language-modelling loss and `generate()` compatibility.

```python
class SLMForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        self.model = SLMModel(config)   # the backbone
        self.lm_head = nn.Linear(...)   # the head
```

This matches the HuggingFace convention used by Llama, Mistral, Qwen, and essentially every modern causal LM in the `transformers` library. The pattern exists so one backbone can serve multiple task heads. Hypothetical variants would share the `SLMModel` weights:

- `SLMForCausalLM` — linear to vocab (what we use)
- `SLMForSequenceClassification` — linear to N classes, pooled
- `SLMForTokenClassification` — linear per-token (NER, tagging)
- `SLMForQuestionAnswering` — two linears for answer span start/end

This project only trains `SLMForCausalLM`, but the split is still worth preserving because:

1. **Tools in the ecosystem assume it.** `trl`, `lm-evaluation-harness`, and `vLLM` expect a `.model` attribute on the causal-LM class and a `base_model_prefix` of `"model"` in state dicts. Flattening the architecture into one class would break these integrations.
2. **`from_pretrained` uses the split to load cross-head.** If you ever add a classification head for fine-tuning, `SLMForSequenceClassification.from_pretrained(causal_lm_checkpoint)` will load the backbone weights correctly and only re-initialise the new head. That's free functionality from the HF machinery.
3. **It keeps the LM-specific concerns separate from the architecture.** Loss computation, generation logic, and the vocabulary projection all live in `SLMForCausalLM`. The transformer itself stays task-agnostic.

If you see `self.model.embed_tokens` or `self.model.layers` in training or inference code, that's the outer `SLMForCausalLM` reaching into its wrapped `SLMModel` — not a weird double-nesting, just the convention.

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