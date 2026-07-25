# model

## Purpose

Decoder-only Transformer implementation for SLM.

- `model/config.py` — `SLMConfig`
- `model/model.py` — `SLMModel` and `SLMForCausalLM`
- `model/block.py` — decoder block
- `model/attention.py` — GQA attention and RoPE
- `model/mlp.py` — SwiGLU MLP
- `model/norm.py` — RMSNorm

## How It Fits In

Training stages use this implementation directly. `export/` converts final
checkpoints to an equivalent native Transformers package for distribution; see
[Architecture](../docs/ARCHITECTURE.md).

## Architecture

| Component | Choice |
|---|---|
| Model type | dense decoder-only causal LM |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Block style | pre-norm with residual connections |
| MLP | SwiGLU |
| Attention | grouped-query attention |
| Bias | none |
| Embeddings | tied input/output embeddings |
| Cache | generation KV cache support |

## Configured sizes

Counts below are unique trainable parameters; the LM head shares the token
embedding matrix.

| Size | Parameters | Layers | Hidden | Q heads | KV heads | Context |
|---|---:|---:|---:|---:|---:|---:|
| `mini` | 21.7M | 6 | 384 | 6 | 2 | 1024 |
| `125m` | 125.3M | 16 | 768 | 12 | 4 | 2048 |
| `350m` | 351.3M | 27 | 1024 | 16 | 8 | 2048 |
| `1b` | 1.012B | 21 | 2048 | 32 | 8 | 4096 |

Size-specific training configs live in `pretrain/configs/`.

## Key classes

- `SLMConfig` extends `transformers.PretrainedConfig`.
- `SLMModel` is the internal decoder stack and is a plain `nn.Module`.
- `SLMForCausalLM` is the Hugging Face `PreTrainedModel` wrapper used for training, export, inference, and serving.

Use `AutoModelForCausalLM`, not `AutoModel`, for exported checkpoints.

## Attention

`attention.py` implements grouped-query attention with RoPE.

RoPE caches are rebuilt lazily in float32 and are not persisted in checkpoints. This avoids stale or low-precision RoPE buffers when loading or casting models.

## Export

Training checkpoints use the in-repository `SLMConfig` and
`SLMForCausalLM`. The export stage converts them to the equivalent native
Transformers Llama configuration and state-dict contract. Published models
therefore load without repository code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tohio/slm-125m-chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

The exported `config.json` must advertise:

```json
{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"]
}
```

It must not contain `auto_map`, and the Hub repository must not contain
architecture Python files.

## Tests

```bash
make test-model
```

Model tests cover architecture construction, masks, cache behavior, and parameterization.
