# export

Exports trained SLM checkpoints to the HuggingFace Hub. Registers the custom architecture with AutoConfig and AutoModelForCausalLM, generates a fully populated model card, and pushes weights, tokenizer, and config in a single commit.

Three variants are exported per model size, each to a separate Hub repository:

| Variant | Checkpoint | Hub repo | Description |
|---|---|---|---|
| `base` | `results/slm-{size}/final` | `$HF_USERNAME/slm-{size}` | Pretrained only |
| `instruct` | `results/slm-{size}-chat-code/final` | `$HF_USERNAME/slm-{size}-instruct` | Chat + response-control + code SFT |
| `chat` | `results/slm-{size}-dpo/final` | `$HF_USERNAME/slm-{size}-chat` | SFT + DPO aligned |

---

## Prerequisites

```bash
# Set HuggingFace credentials in .env
HF_TOKEN=hf_...
HF_USERNAME=<your-hub-username>    # required — export aborts if unset

# Run evaluation before export — each variant's checkpoint is evaluated
# separately, and the model card for each variant embeds its own scores.
python eval/eval.py --model results/slm-125m/final
python eval/eval.py --model results/slm-125m-chat-code/final
python eval/eval.py --model results/slm-125m-dpo/final
# Or: make eval-all SIZE=125m
```

`HF_USERNAME` has no default — export fails loudly if it's unset, to prevent a fork from accidentally writing links that point at someone else's Hub namespace.

---

## Usage

```bash
# Export all three variants
make export SIZE=125m

# Export individual variants
make export-base     SIZE=125m
make export-instruct SIZE=125m
make export-chat     SIZE=125m

# Dry run — validate and preview the model card without pushing
python export/export.py --size 125m --variant chat --dry-run

# Private repositories
python export/export.py --size 125m --variant chat --private

# Override checkpoint path
python export/export.py --size 125m --variant chat --model path/to/checkpoint
```

---

## What Gets Pushed

Pushed in a single commit per variant:

- `model.safetensors` — model weights in safetensors format
- `config.json` — SLMConfig with architecture details
- `tokenizer.json`, `tokenizer_config.json` — trained BPE tokenizer including the baked-in chat template
- `README.md` — auto-generated model card (written to the checkpoint directory first, then included in the same commit as the weights)

---

## Model Card

Generated automatically at export time. Every variant gets:

- **Architecture table** — component choices and rationale
- **Training section**
  - Pretraining corpus table: all 13 sources (8 non-code top-level + 5 code sub-sources) with target percentages from `config/data_mix.py` and realized percentages from `data/curated/blend_stats.json` if available. Realized columns appear when blend_stats.json is present and matches the export size; otherwise the table falls back to design-target-only with a caveat noting that the realized mix may differ. Both views are sourced from `config/data_mix.py` as the single source of truth shared with the curator and notebooks
  - Fine-tuning tables for the `instruct` and `chat` variants, listing chat SFT, response-control SFT, code SFT, and (for chat) DPO datasets
- **Parameter count** — actual value from the loaded checkpoint
- **Token targets** — sourced from `config/data_mix.py` (5B/15B/30B for 125m/350m/1b)
- **Benchmark results** — populated from the most recent `eval.py` run for **this variant's checkpoint**. `base` shows base-model scores, `instruct` shows SFT scores, `chat` shows post-DPO scores
- **Hardware** — training hardware used
- **Limitations** — scale, hallucination, safety, language, and code coverage
- **Usage example** — copy-paste ready code using `apply_chat_template`

If a variant has no eval results yet, its benchmark table is rendered with a placeholder ("_Benchmark results will be added after evaluation._") rather than a stale table. A warning is logged during export.

### Realized vs target mix

When `data/curated/blend_stats.json` is present and its `target` field matches the `--size` being exported, the model card includes a "Realized" column showing what actually shipped. Supply-bound sources can under-fill their target; the deficit routes to FineWeb, lifting its realized share above target. Showing both columns gives readers an honest view of what was trained on.

When blend_stats.json is missing (e.g., exporting from a training instance that doesn't have the curated data on local disk), the table falls back to design-target-only with a caveat. To get realized numbers in the model card, ensure blend_stats.json is downloaded from S3 alongside the validated/tokenized data, or run export from the curate instance.

---

## Pre-push Validation

Before any Hub push, `export.py` runs a short generation and asserts the model produced at least one non-stop token. This catches checkpoints that silently break (NaN weights, broken tied-weight restore, corrupted save) and would otherwise be published to the Hub. The check takes a few seconds and uses the real tokenizer's special-token IDs resolved via `inference.utils`.

A failing validation aborts the export with a clear error — nothing reaches the Hub.

---

## Chat Template

The exported tokenizer includes the baked-in Jinja2 chat template from `tokenizer/train_tokenizer.py`. `export.py` loads the tokenizer via `PreTrainedTokenizerFast.from_pretrained()` — it never reconstructs or overwrites the template. The template on the Hub is always identical to the one the model was trained with.

Export will raise an error if the tokenizer has no `chat_template`, preventing a model with a broken chat format from being pushed.

---

## After Export

Models are loadable anywhere with the standard HuggingFace API:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("<user>/slm-125m-chat")
tokenizer = AutoTokenizer.from_pretrained("<user>/slm-125m-chat")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a transformer?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
)
output = model.generate(inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```