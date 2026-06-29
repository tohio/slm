# export

Export trained SLM variants to Hugging Face Hub with bundled remote-code files, tokenizer assets, model cards, and eval metadata.

---

## Responsibility

`export/` owns:

- variant checkpoint validation
- Hub repo naming
- model-card generation
- remote-code bundling for custom `SLMConfig` / `SLMForCausalLM`
- tokenizer packaging

---

## Variants

| Variant | Checkpoint | Hub repo |
|---|---|---|
| base | `results/runs/<size>/pretrain/final` | `tohio/slm-<size>` |
| instruct | `results/runs/<size>/sft_instruct/final` | `tohio/slm-<size>-instruct` |
| chat | `results/runs/<size>/dpo_chat/final` | `tohio/slm-<size>-chat` |
| code | `results/runs/<size>/sft_code/final` | `tohio/slm-<size>-code` |

---

## Commands

```bash
make export-base     SIZE=125m
make export-instruct SIZE=125m
make export-chat     SIZE=125m
make export-code     SIZE=125m
make export          SIZE=125m
```

Direct invocation:

```bash
python export/export.py --size 125m --variant base
python export/export.py --size 125m --variant instruct
python export/export.py --size 125m --variant chat
python export/export.py --size 125m --variant code
```

---

## Prerequisites

Set Hub credentials:

```bash
HF_USERNAME=tohio
HF_TOKEN=...
```

Run eval before export so model cards can include variant-specific scores:

```bash
make eval-base     SIZE=125m
make eval-instruct SIZE=125m
make eval-chat     SIZE=125m
make eval-code     SIZE=125m
```

---

## Model card inputs

Export reads:

```text
config/data_mix.py
data/runs/<size>/metadata/blend_stats.json
data/runs/<size>/curated/blend_stats.json
results/runs/<size>/eval/
```

The metadata path is preferred. The curated path is a legacy fallback.

---

## What gets pushed

Each Hub repo contains:

- model weights
- tokenizer files
- config files
- bundled `model/` source files required for `trust_remote_code=True`
- generated model card
- eval metadata when available

---

## Loading exported models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tohio/slm-125m-chat"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```
