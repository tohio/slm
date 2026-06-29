# model

Decoder-only Transformer implementation for SLM.

---

## Responsibility

`model/` owns the architecture used for training, export, inference, and serving.

---

## Architecture

| Component | Choice |
|---|---|
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Attention | GQA |
| Bias | None |
| Embeddings | Tied |

---

## Model sizes

| Size | Layers | Hidden | Q heads | KV heads | Context |
|---|---:|---:|---:|---:|---:|
| `125m` | 12 | 768 | 12 | 4 | 2048 |
| `350m` | 24 | 1024 | 16 | 8 | 2048 |
| `1b` | 32 | 2048 | 32 | 8 | 4096 |

---

## Files

```text
model/
├── config.py
├── attention.py
├── mlp.py
├── norm.py
├── block.py
├── model.py
└── README.md
```

---

## Key classes

- `SLMConfig` defines architecture and tokenizer/model metadata.
- `SLMForCausalLM` implements the causal language model.
- Blocks use Pre-LN style normalization with residual connections.
- Attention uses grouped-query attention.
- MLP uses SwiGLU.

---

## Hub loading

Exported models bundle these source files for `trust_remote_code=True`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tohio/slm-125m-chat"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

---

## Tests

```bash
make test-model
```

Model tests cover architecture construction, masks, cache behavior, and parameterization.
