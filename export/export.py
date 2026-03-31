"""
export/export.py
-----------------
Export a trained SLM checkpoint to the HuggingFace Hub.

Registers the custom SLMConfig and SLMForCausalLM with AutoConfig and
AutoModelForCausalLM so the model can be loaded anywhere with:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("tohio/slm-125m")

Pushes:
    - Model weights (model.safetensors)
    - Config (config.json)
    - Tokenizer files
    - Model card (README.md)

Usage:
    python export/export.py --model results/slm-125m-dpo/final --size 125m
    python export/export.py --model results/slm-350m-dpo/final --size 350m
    python export/export.py --model results/slm-1b-dpo/final --size 1b

    # Dry run — validate without pushing
    python export/export.py --model results/slm-125m-dpo/final --size 125m --dry-run
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HF_USERNAME = os.environ.get("HF_USERNAME", "tohio")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ── Model card ────────────────────────────────────────────────────────────────

def generate_model_card(size: str, hub_name: str) -> str:
    """Generate a model card README for the HuggingFace Hub."""
    size_upper = size.upper()
    return f"""---
license: mit
language:
  - en
tags:
  - causal-lm
  - decoder-only
  - custom-architecture
  - rope
  - gqa
  - swiglu
base_model: trained from scratch
---

# {hub_name}

A {size_upper} decoder-only language model trained from scratch. Part of the SLM model family.

## Architecture

| Component | Choice |
|---|---|
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Attention | GQA |
| Vocab size | 32,000 |

## Training

- **Pretraining:** {_token_target(size)} tokens on Wikipedia EN + CodeSearchNet + Common Crawl
- **SFT:** OpenHermes-2.5 (chat) + Magicoder-OSS-Instruct (code)
- **Alignment:** DPO on Anthropic/hh-rlhf + Intel/orca_dpo_pairs + argilla/dpo-mix-7k

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{HF_USERNAME}/{hub_name}")
tokenizer = AutoTokenizer.from_pretrained("{HF_USERNAME}/{hub_name}")

messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Explain what a transformer is."}},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Related

- [slm](https://github.com/tohio/slm) — training pipeline
- [ai-infra](https://github.com/tohio/ai-infra) — production serving
"""


def _token_target(size: str) -> str:
    targets = {"125m": "3B", "350m": "10B", "1b": "25B"}
    return targets.get(size, "N/A")


# ── Export ────────────────────────────────────────────────────────────────────

def export(
    model_path: Path,
    size: str,
    dry_run: bool = False,
    private: bool = False,
) -> None:
    """
    Export a model checkpoint to the HuggingFace Hub.

    Args:
        model_path: Path to the model checkpoint directory.
        size: Model size string (125m, 350m, 1b).
        dry_run: If True, validate without pushing to Hub.
        private: If True, create a private repository.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
    from huggingface_hub import HfApi, login
    from model import SLMConfig, SLMForCausalLM

    hub_name = f"slm-{size}"
    repo_id = f"{HF_USERNAME}/{hub_name}"

    log.info(f"=== SLM Export ===")
    log.info(f"Model:    {model_path}")
    log.info(f"Hub:      {repo_id}")
    log.info(f"Dry run:  {dry_run}")

    # Register custom model with AutoConfig
    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    # Load model
    log.info("Loading model...")
    config = SLMConfig.from_pretrained(str(model_path))
    model = SLMForCausalLM.from_pretrained(str(model_path))
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Load tokenizer
    tokenizer_path = model_path / "tokenizer"
    if not tokenizer_path.exists():
        tokenizer_path = Path("data/tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token = "<PAD>"
    tokenizer.eos_token = "<EOS>"
    tokenizer.bos_token = "<BOS>"

    # Add chat template
    tokenizer.chat_template = _chat_template()

    if dry_run:
        log.info("Dry run — skipping Hub push")
        log.info(f"Would push to: https://huggingface.co/{repo_id}")
        _validate_model(model, tokenizer, config)
        return

    # Login
    if not HF_TOKEN:
        log.error("HF_TOKEN not set in .env")
        sys.exit(1)
    login(token=HF_TOKEN)

    # Generate model card
    model_card = generate_model_card(size, hub_name)
    card_path = model_path / "README.md"
    with open(card_path, "w") as f:
        f.write(model_card)

    # Push to Hub
    log.info(f"Pushing to {repo_id}...")
    model.push_to_hub(
        repo_id,
        token=HF_TOKEN,
        private=private,
        safe_serialization=True,
    )
    tokenizer.push_to_hub(
        repo_id,
        token=HF_TOKEN,
        private=private,
    )
    config.push_to_hub(
        repo_id,
        token=HF_TOKEN,
        private=private,
    )

    # Push model card
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=HF_TOKEN,
    )

    log.info(f"Export complete: https://huggingface.co/{repo_id}")


def _validate_model(model, tokenizer, config) -> None:
    """Quick sanity check before pushing."""
    import torch

    log.info("Validating model...")
    model.eval()

    prompt = "<|system|>You are a helpful assistant.<|endofturn|><|user|>Hello!<|endofturn|><|assistant|>"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=20,
            pad_token_id=0,
            eos_token_id=3,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    log.info(f"Validation output: {decoded[:100]}")
    log.info("✓ Model validation passed")


def _chat_template() -> str:
    """Jinja2 chat template for the SLM chat format."""
    return (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|system|>{{ message['content'] }}<|endofturn|>"
        "{% elif message['role'] == 'user' %}"
        "<|user|>{{ message['content'] }}<|endofturn|>"
        "{% elif message['role'] == 'assistant' %}"
        "<|assistant|>{{ message['content'] }}<|endofturn|>"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export SLM to HuggingFace Hub")
    parser.add_argument("--model", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--size", type=str, required=True, choices=["125m", "350m", "1b"])
    parser.add_argument("--dry-run", action="store_true", help="Validate without pushing")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    args = parser.parse_args()

    if not args.model.exists():
        log.error(f"Model not found: {args.model}")
        sys.exit(1)

    export(
        model_path=args.model,
        size=args.size,
        dry_run=args.dry_run,
        private=args.private,
    )


if __name__ == "__main__":
    main()