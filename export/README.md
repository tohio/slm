# export

Exports trained SLM checkpoints to the HuggingFace Hub. Registers the custom architecture with AutoConfig and AutoModelForCausalLM, generates a model card, and pushes weights, tokenizer, and config.

---

## Usage

```bash
# Export 125M DPO model
make export SIZE=125m

# Or directly
python export/export.py --model results/slm-125m-dpo/final --size 125m

# Dry run — validate without pushing
python export/export.py --model results/slm-125m-dpo/final --size 125m --dry-run

# Private repository
python export/export.py --model results/slm-125m-dpo/final --size 125m --private
```

---

## Prerequisites

```bash
# Set HuggingFace credentials in .env
HF_TOKEN=hf_...
HF_USERNAME=tohio

# Login (handled automatically by export.py)
huggingface-cli login
```

---

## What Gets Pushed

- `model.safetensors` — model weights in safetensors format
- `config.json` — SLMConfig with architecture details
- `tokenizer.json` — trained BPE tokenizer
- `tokenizer_config.json` — tokenizer metadata including chat template
- `README.md` — auto-generated model card

---

## After Export

Models are loadable anywhere with the standard HuggingFace API:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tohio/slm-125m")
tokenizer = AutoTokenizer.from_pretrained("tohio/slm-125m")

# Chat template built in
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a transformer?"},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Hub Repositories

| Model | Hub path |
|---|---|
| `slm-125m` | `tohio/slm-125m` |
| `slm-350m` | `tohio/slm-350m` |
| `slm-1b` | `tohio/slm-1b` |