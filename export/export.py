"""
export/export.py
-----------------
Export a trained SLM checkpoint to the HuggingFace Hub.

Registers the custom SLMConfig and SLMForCausalLM with AutoConfig and
AutoModelForCausalLM so the model can be loaded anywhere with:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("tohio/slm-125m")

Three variants are exported per model size:

    Variant     Checkpoint                        Hub repo
    --------    ---------                         --------
    base        results/slm-{size}/final          tohio/slm-{size}
    instruct    results/slm-{size}-chat-code/final tohio/slm-{size}-instruct
    chat        results/slm-{size}-dpo/final       tohio/slm-{size}-chat

Pushes:
    - Model weights (model.safetensors)
    - Config (config.json)
    - Tokenizer files
    - Model card (README.md)

Usage:
    python export/export.py --size 125m --variant base
    python export/export.py --size 125m --variant instruct
    python export/export.py --size 125m --variant chat

    # Dry run — validate without pushing
    python export/export.py --size 125m --variant chat --dry-run

    # Export all three variants
    for variant in base instruct chat; do
        python export/export.py --size 125m --variant $variant
    done
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
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))

# ── Variant definitions ────────────────────────────────────────────────────────
#
# Maps (size, variant) → (checkpoint_dir, hub_suffix, description)
#
#   base      — pretrained only, no fine-tuning
#   instruct  — chat SFT + code SFT (instruction following)
#   chat      — instruct + DPO alignment (preferred for conversation)

VARIANTS = {
    "base": {
        "checkpoint": lambda size: RESULTS_DIR / f"slm-{size}" / "final",
        "hub_suffix": "",                  # tohio/slm-125m
        "description": "base pretrained model",
        "pipeline_tag": "text-generation",
    },
    "instruct": {
        "checkpoint": lambda size: RESULTS_DIR / f"slm-{size}-chat-code" / "final",
        "hub_suffix": "-instruct",         # tohio/slm-125m-instruct
        "description": "instruction-tuned via chat SFT + code SFT",
        "pipeline_tag": "text-generation",
    },
    "chat": {
        "checkpoint": lambda size: RESULTS_DIR / f"slm-{size}-dpo" / "final",
        "hub_suffix": "-chat",             # tohio/slm-125m-chat
        "description": "chat-aligned via SFT + DPO preference learning",
        "pipeline_tag": "conversational",
    },
}


# ── Model card ────────────────────────────────────────────────────────────────

def generate_model_card(size: str, variant: str, hub_name: str) -> str:
    """Generate a model card README for the HuggingFace Hub."""
    size_upper = size.upper()
    variant_cfg = VARIANTS[variant]
    description = variant_cfg["description"]
    pipeline_tag = variant_cfg["pipeline_tag"]
    token_target = _token_target(size)

    variant_section = {
        "base": f"""\
This is the **base** variant — pretrained on {token_target} tokens with no fine-tuning.
Use `{HF_USERNAME}/slm-{size}-instruct` for instruction following or
`{HF_USERNAME}/slm-{size}-chat` for aligned conversation.
""",
        "instruct": f"""\
This is the **instruct** variant — the base model fine-tuned on chat and code instruction datasets.
Use `{HF_USERNAME}/slm-{size}-chat` for the DPO-aligned version preferred for conversation.
Use `{HF_USERNAME}/slm-{size}` for the raw base model.
""",
        "chat": f"""\
This is the **chat** variant — the instruct model further aligned via DPO preference learning.
This is the recommended variant for conversational use.
Use `{HF_USERNAME}/slm-{size}-instruct` for the SFT-only version.
Use `{HF_USERNAME}/slm-{size}` for the raw base model.
""",
    }[variant]

    training_section = {
        "base": f"- **Pretraining:** {token_target} tokens on Wikipedia EN + CodeSearchNet + Common Crawl (70/20/10)",
        "instruct": f"""\
- **Pretraining:** {token_target} tokens on Wikipedia EN + CodeSearchNet + Common Crawl (70/20/10)
- **Chat SFT:** OpenHermes-2.5
- **Code SFT:** Magicoder-OSS-Instruct""",
        "chat": f"""\
- **Pretraining:** {token_target} tokens on Wikipedia EN + CodeSearchNet + Common Crawl (70/20/10)
- **Chat SFT:** OpenHermes-2.5
- **Code SFT:** Magicoder-OSS-Instruct
- **DPO alignment:** Anthropic/hh-rlhf + Intel/orca_dpo_pairs + argilla/dpo-mix-7k""",
    }[variant]

    return f"""---
license: mit
language:
  - en
pipeline_tag: {pipeline_tag}
tags:
  - causal-lm
  - decoder-only
  - custom-architecture
  - rope
  - gqa
  - swiglu
  - {variant}
base_model: {"trained from scratch" if variant == "base" else f"{HF_USERNAME}/slm-{size}"}
---

# {hub_name}

A {size_upper} decoder-only language model ({description}). Part of the SLM model family
built entirely from scratch — from raw web data through to a production-serving aligned model.

{variant_section}

## Model Family

| Variant | Hub | Description |
|---|---|---|
| Base | [{HF_USERNAME}/slm-{size}](https://huggingface.co/{HF_USERNAME}/slm-{size}) | Pretrained only |
| Instruct | [{HF_USERNAME}/slm-{size}-instruct](https://huggingface.co/{HF_USERNAME}/slm-{size}-instruct) | Chat + code SFT |
| Chat | [{HF_USERNAME}/slm-{size}-chat](https://huggingface.co/{HF_USERNAME}/slm-{size}-chat) | SFT + DPO aligned |

## Architecture

| Component | Choice | Rationale |
|---|---|---|
| Positional encoding | RoPE | Better length generalisation |
| Normalization | RMSNorm | Faster than LayerNorm |
| Activation | SwiGLU | Better gradient flow |
| Attention | GQA | Reduced KV memory at inference |
| Vocab size | 32,000 | Custom BPE tokenizer |
| Parameters | {_param_count(size)} | |

## Training

{training_section}

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

- [slm](https://github.com/tohio/slm) — full training pipeline
- [ai-infra](https://github.com/tohio/ai-infra) — production Kubernetes serving
"""


def _token_target(size: str) -> str:
    return {"125m": "3B", "350m": "10B", "1b": "25B"}.get(size, "N/A")


def _param_count(size: str) -> str:
    return {"125m": "125M", "350m": "350M", "1b": "1B"}.get(size, "N/A")


# ── Export ────────────────────────────────────────────────────────────────────

def export(
    size: str,
    variant: str,
    model_path: Path | None = None,
    dry_run: bool = False,
    private: bool = False,
) -> None:
    """
    Export a model checkpoint to the HuggingFace Hub.

    Args:
        size:       Model size string (125m, 350m, 1b).
        variant:    Model variant (base, instruct, chat).
        model_path: Override checkpoint path. Defaults to VARIANTS mapping.
        dry_run:    If True, validate without pushing to Hub.
        private:    If True, create a private repository.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
    from huggingface_hub import HfApi, login
    from model import SLMConfig, SLMForCausalLM

    variant_cfg = VARIANTS[variant]
    checkpoint = model_path or variant_cfg["checkpoint"](size)
    hub_suffix = variant_cfg["hub_suffix"]
    hub_name = f"slm-{size}{hub_suffix}"
    repo_id = f"{HF_USERNAME}/{hub_name}"

    log.info(f"=== SLM Export ===")
    log.info(f"Size:     {size}")
    log.info(f"Variant:  {variant}")
    log.info(f"Checkpoint: {checkpoint}")
    log.info(f"Hub:      {repo_id}")
    log.info(f"Dry run:  {dry_run}")

    if not checkpoint.exists():
        log.error(f"Checkpoint not found: {checkpoint}")
        log.error(f"Run the full training pipeline first: make pretrain sft sft-code dpo SIZE={size}")
        sys.exit(1)

    # Register custom model with AutoConfig
    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    # Load model
    log.info("Loading model...")
    config = SLMConfig.from_pretrained(str(checkpoint))
    model = SLMForCausalLM.from_pretrained(str(checkpoint))
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Load tokenizer
    tokenizer_path = checkpoint / "tokenizer"
    if not tokenizer_path.exists():
        tokenizer_path = Path("data/tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token  = "<PAD>"
    tokenizer.eos_token  = "<EOS>"
    tokenizer.bos_token  = "<BOS>"
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

    # Generate and write model card
    model_card = generate_model_card(size, variant, hub_name)
    card_path = checkpoint / "README.md"
    with open(card_path, "w") as f:
        f.write(model_card)

    # Push to Hub
    log.info(f"Pushing to {repo_id}...")
    model.push_to_hub(repo_id, token=HF_TOKEN, private=private, safe_serialization=True)
    tokenizer.push_to_hub(repo_id, token=HF_TOKEN, private=private)
    config.push_to_hub(repo_id, token=HF_TOKEN, private=private)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=HF_TOKEN,
    )

    log.info(f"Export complete: https://huggingface.co/{repo_id}")


def _validate_model(model, tokenizer, config) -> None:
    """Quick sanity check — generate a short sequence before pushing."""
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
    parser = argparse.ArgumentParser(
        description="Export SLM to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export/export.py --size 125m --variant base
  python export/export.py --size 125m --variant instruct
  python export/export.py --size 125m --variant chat
  python export/export.py --size 125m --variant chat --dry-run

  # Export all three variants
  for variant in base instruct chat; do
      python export/export.py --size 125m --variant $variant
  done
        """,
    )
    parser.add_argument(
        "--size",
        type=str,
        required=True,
        choices=["125m", "350m", "1b"],
        help="Model size",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=list(VARIANTS.keys()),
        help="Model variant: base (pretrain only), instruct (SFT), chat (SFT + DPO)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override checkpoint path (defaults to variant mapping)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without pushing to Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private Hub repository",
    )
    args = parser.parse_args()

    export(
        size=args.size,
        variant=args.variant,
        model_path=args.model,
        dry_run=args.dry_run,
        private=args.private,
    )


if __name__ == "__main__":
    main()