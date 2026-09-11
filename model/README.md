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
| Attention | FlashAttention-3 or explicit PyTorch SDPA, with grouped-query attention |
| Normalization | RMSNorm |
| Feed-forward network | bias-free SwiGLU |
| Embeddings | tied token embedding and LM-head weights |
| Generation | Transformers `Cache` support and legacy-cache conversion |
| Dropout | configurable attention dropout; zero in the supplied profiles |

Pretraining selects `attn_implementation="flash_attention_3"`. SLM requires
CUDA BF16 inputs and zero attention dropout for this backend. Transformers'
existing FA3 integration handles 2D padding masks, explicit positions, and
dynamic KV-cache attention; FA3 consumes the original grouped KV heads. A
missing extension or unsupported input raises an error; SLM does not retry with
FA2, SDPA, or a downloaded Hub kernel. Setup builds the pinned FA3 source after
installing PyTorch; see [GPU setup](../infra/README.md).

Direct `SLMConfig()` construction keeps SDPA as its default for CPU/FP32
diagnostics. To select it explicitly, pass `attn_implementation="sdpa"` to
`SLMConfig` or `SLMForCausalLM.from_pretrained`. Use SDPA for custom 4D masks,
static caches, or nonzero attention dropout. Its existing fused-backend
restriction for CUDA BF16/FP16 training is retained. The stored KV cache stays
grouped; only SDPA may temporarily expand KV heads for backend eligibility.

MLP gate/up and attention Q/K/V each use one combined linear projection.
Weights are concatenated during forward and outputs are split by their original
widths (Mini: MLP 1536/1536, QKV 512/256/256). The original Parameters, names,
initialization, optimizer ownership, and native Llama export remain intact;
there is no checkpoint migration. Weight concatenation and its backward cost
must be included when measuring a speedup.

Generated and checked-in pretraining recipes use Inductor's
`max-autotune-no-cudagraphs` mode. `training.torch_compile_mode: default` is the
comparison setting. Autotuning adds startup work and does not guarantee faster
complete updates. BF16 loss, gradient/update agreement, compilation, and A100
throughput still require validation on the training host.

RoPE inverse frequencies use a canonical CPU FP32 calculation, followed by a
copy to the buffer's destination device. Initialization, checkpoint-buffer
recovery, and dtype/device moves use the same calculation; moving to CUDA must
not recompute these constants with different rounding. The buffer stays FP32
and non-persistent, including after BF16/FP16 conversion. This matches the
CPU-initialized native Llama reference used by the implementation check without
copying values from that reference or changing its tolerances. The forward
path is unchanged: no CPU frequency calculation or transfer is added per step.

Pretraining loss-only calls can enable Liger fused linear cross-entropy. In that
path SLM passes final hidden states and the tied LM-head weight directly to the
fused loss, so the full `[batch, sequence, vocab]` logits tensor is not
materialized. Training and evaluation preserve Transformers' next-token shift,
`ignore_index=-100`, and `num_items_in_batch` normalization. Generation, SFT/DPO,
implementation checks that request outputs, and ordinary callers continue to use
the standard logits-producing path.

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

Model tests require the pinned Transformers 5.14.1 training contract:

```bash
make setup-train  # omit when the training environment is already prepared
make test-model
```

The model suite covers construction, parameter counts, causal and padding
masks, cached decoding, generation parity, validation errors, and checkpoint
round trips.

## Derived RoPE state during checkpoint loading

`RotaryEmbedding.inv_freq` is not learned state and remains non-persistent.
Native low-memory checkpoint loading can materialize such a buffer without a
checkpoint tensor to fill it. `SLMForCausalLM.from_pretrained` therefore
reconstructs it from `rope_theta` and `head_dim` after native loading completes.
The RoPE module also reconstructs it after state-dict loads and device/dtype
conversion, retaining FP32 frequencies. This is architecture-level behavior,
not a Mini or generation workaround. No broad `init_weights()` call is made on
the loaded model; learned embeddings and other learned tensors are unchanged.

Run `tests/model/test_rope_loading.py` on the pinned training stack and reproduce
the real Mini validation baseline with the matching original tokenizer/data.
See [the data/evaluation guide](../docs/PRETRAINING_DATA.md) for the distinction
between that validation check and an unseen test evaluation.
