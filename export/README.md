# export

Hugging Face export for SLM variants.

Export packages checkpoint weights, tokenizer files, custom model source files, config metadata, and a model card. It does not run evaluation and model cards do not include benchmark tables.

---

## Owns

- `export/export.py` — export entry point
- variant-to-checkpoint mapping
- Hub repo naming
- tokenizer root packaging
- remote-code bundling
- model-card generation

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

Make targets:

```bash
make export-base SIZE=125m
make export-instruct SIZE=125m
make export-chat SIZE=125m
make export-code SIZE=125m
make export SIZE=125m
```

Direct calls:

```bash
python export/export.py --size 125m --variant base
python export/export.py --size 125m --variant instruct
python export/export.py --size 125m --variant chat
python export/export.py --size 125m --variant code
python export/export.py --size 125m --variant chat --dry-run
```

---

## Environment

Required for Hub push:

```bash
HF_USERNAME=tohio
HF_TOKEN=...
```

`.env` is loaded by `export/export.py`.

---

## What export writes

Before pushing, export updates the checkpoint directory with:

```text
README.md
tokenizer.json
tokenizer_config.json
special_tokens_map.json
config.py
model.py
block.py
attention.py
mlp.py
norm.py
```

Tokenizer files are saved/copied to the checkpoint root so standard Hub loading works without `subfolder="tokenizer"`.

Remote-code files are copied to the checkpoint root and `config.json` is updated with:

```json
{
  "AutoConfig": "config.SLMConfig",
  "AutoModelForCausalLM": "model.SLMForCausalLM"
}
```

---

## Model card policy

Model cards include:

- model family and variant
- architecture summary
- pretraining data-mix table
- fine-tuning/alignment summary
- usage example
- limitations

Model cards do not include evaluation or benchmark tables.

---

## Data-mix metadata

Export loads realized curation stats from:

```text
data/runs/<size>/metadata/blend_stats.json
```

Legacy fallback:

```text
data/runs/<size>/curated/blend_stats.json
```

If no blend stats are available, export falls back to the design mix from `config/data_mix.py`.

---

## Validation

Export performs a short generation hygiene check before pushing. This is packaging validation, not benchmark evaluation.

The check catches obviously broken checkpoints such as:

- missing tokenizer files
- missing architecture files
- missing `auto_map`
- empty generation
- highly repetitive generation

Use `--dry-run` to validate packaging without pushing to the Hub.

---

## Loading exported models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tohio/slm-125m-chat"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```
