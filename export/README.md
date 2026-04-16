# export

Exports trained SLM checkpoints to the HuggingFace Hub. Registers the custom
architecture with AutoConfig and AutoModelForCausalLM, generates a fully
populated model card, and pushes weights, tokenizer, and config.

Three variants are exported per model size, each to a separate Hub repository:

| Variant | Checkpoint | Hub repo | Description |
|---|---|---|---|
| `base` | `results/slm-{size}/final` | `tohio/slm-{size}` | Pretrained only |
| `instruct` | `results/slm-{size}-chat-code/final` | `tohio/slm-{size}-instruct` | Chat + code SFT |
| `chat` | `results/slm-{size}-dpo/final` | `tohio/slm-{size}-chat` | SFT + DPO aligned |

---

## Prerequisites

```bash
# Set HuggingFace credentials in .env
HF_TOKEN=hf_...
HF_USERNAME=tohio

# Run evaluation before export — eval results are embedded in the model card
make eval SIZE=125m
```

---

## Usage

```bash
# Export all three variants
make export SIZE=125m

# Export individual variants
make export-base     SIZE=125m
make export-instruct SIZE=125m
make export-chat     SIZE=125m

# Dry run — validate without pushing
python export/export.py --size 125m --variant chat --dry-run

# Private repositories
python export/export.py --size 125m --variant chat --private

# Override checkpoint path
python export/export.py --size 125m --variant chat --model path/to/checkpoint
```

---

## What Gets Pushed

- `model.safetensors` — model weights in safetensors format
- `config.json` — SLMConfig with architecture details
- `tokenizer.json` — trained BPE tokenizer
- `tokenizer_config.json` — tokenizer metadata including chat template
- `README.md` — auto-generated model card

---

## Model Card

The model card is generated automatically at export time and includes:

- **Architecture table** — component choices and rationale
- **Training table** — dataset names, Hub links, and sizes per stage (55% CC / 25% Wikipedia / 20% Python)
- **Parameter count** — actual value from the loaded checkpoint
- **Token targets** — 5B (125m), 15B (350m), 30B (1b)
- **Benchmark results** — populated from the most recent `make eval` run (chat variant only)
- **Hardware** — training hardware used
- **Limitations** — scale, hallucination, safety, language, and code coverage
- **Usage example** — copy-paste ready code with `apply_chat_template`

If `make eval` has not been run before `make export-chat`, the benchmark table
will contain a placeholder. Run `make eval SIZE={size}` first.

---

## Chat Template

The exported tokenizer includes the baked-in Jinja2 chat template from
`tokenizer/train_tokenizer.py`. `export.py` loads the tokenizer via
`PreTrainedTokenizerFast.from_pretrained()` — it never reconstructs or
overwrites the template. The template on the Hub is always identical to
the one the model was trained with.

Export will raise an error if the tokenizer has no `chat_template`, preventing
a model with a broken chat format from being pushed to the Hub.

---

## After Export

Models are loadable anywhere with the standard HuggingFace API:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tohio/slm-125m-chat")
tokenizer = AutoTokenizer.from_pretrained("tohio/slm-125m-chat")

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

---

## Hub Repositories

| Size | Base | Instruct | Chat |
|---|---|---|---|
| 125M | [tohio/slm-125m](https://huggingface.co/tohio/slm-125m) | [tohio/slm-125m-instruct](https://huggingface.co/tohio/slm-125m-instruct) | [tohio/slm-125m-chat](https://huggingface.co/tohio/slm-125m-chat) |
| 350M | [tohio/slm-350m](https://huggingface.co/tohio/slm-350m) | [tohio/slm-350m-instruct](https://huggingface.co/tohio/slm-350m-instruct) | [tohio/slm-350m-chat](https://huggingface.co/tohio/slm-350m-chat) |
| 1B | [tohio/slm-1b](https://huggingface.co/tohio/slm-1b) | [tohio/slm-1b-instruct](https://huggingface.co/tohio/slm-1b-instruct) | [tohio/slm-1b-chat](https://huggingface.co/tohio/slm-1b-chat) |