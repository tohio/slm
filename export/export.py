"""
export/export.py
-----------------
Export a trained SLM checkpoint to the HuggingFace Hub.

Registers the custom SLMConfig and SLMForCausalLM with AutoConfig and
AutoModelForCausalLM so the model can be loaded anywhere with:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("tohio/slm-125m")

Three variants are exported per model size:

    Variant     Checkpoint                         Hub repo
    --------    ---------                          --------
    base        results/slm-{size}/final           tohio/slm-{size}
    instruct    results/slm-{size}-chat-code/final tohio/slm-{size}-instruct
    chat        results/slm-{size}-dpo/final        tohio/slm-{size}-chat

Pushes:
    - Model weights (model.safetensors)
    - Config (config.json)
    - Tokenizer files (including chat_template from train_tokenizer.py)
    - Model card (README.md) — populated with actual parameter count,
      eval benchmark results, training details, and limitations

Usage:
    python export/export.py --size 125m --variant base
    python export/export.py --size 125m --variant instruct
    python export/export.py --size 125m --variant chat
    python export/export.py --size 125m --variant chat --dry-run
"""

import argparse
import json
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

# Import CHAT_TEMPLATE from train_tokenizer — single source of truth.
# Do not duplicate or redefine the template here. The exported tokenizer
# must use exactly the same template the model was trained with.
from tokenizer.train_tokenizer import CHAT_TEMPLATE

# ── Variant definitions ────────────────────────────────────────────────────────

VARIANTS = {
    "base": {
        "checkpoint": lambda size: RESULTS_DIR / f"slm-{size}" / "final",
        "hub_suffix":   "",           # tohio/slm-125m
        "description":  "base pretrained model",
        "pipeline_tag": "text-generation",
    },
    "instruct": {
        "checkpoint": lambda size: RESULTS_DIR / f"slm-{size}-chat-code" / "final",
        "hub_suffix":   "-instruct",  # tohio/slm-125m-instruct
        "description":  "instruction-tuned via chat SFT + code SFT",
        "pipeline_tag": "text-generation",
    },
    "chat": {
        "checkpoint": lambda size: RESULTS_DIR / f"slm-{size}-dpo" / "final",
        "hub_suffix":   "-chat",      # tohio/slm-125m-chat
        "description":  "chat-aligned via SFT + DPO preference learning",
        "pipeline_tag": "text-generation",
    },
}

# ── Benchmark metadata ─────────────────────────────────────────────────────────

BENCHMARK_META = {
    "hellaswag":     {"name": "HellaSwag",     "metric": "acc_norm", "few_shot": 10},
    "arc_easy":      {"name": "ARC-Easy",      "metric": "acc_norm", "few_shot": 25},
    "arc_challenge": {"name": "ARC-Challenge", "metric": "acc_norm", "few_shot": 25},
    "mmlu":          {"name": "MMLU",          "metric": "acc",      "few_shot": 5},
    "truthfulqa":    {"name": "TruthfulQA",    "metric": "acc",      "few_shot": 0},
    "humaneval":     {"name": "HumanEval",     "metric": "pass@1",   "few_shot": 0},
}


# ── Eval results ───────────────────────────────────────────────────────────────

def load_eval_results(size: str) -> dict:
    """
    Load the most recent eval results for the given model size.

    Eval results are written by make eval to:
        results/eval/slm-{size}-dpo/eval_<timestamp>.json

    Returns a flat dict of {task_key: score} or empty dict if not found.
    """
    eval_dir = RESULTS_DIR / "eval" / f"slm-{size}-dpo"
    if not eval_dir.exists():
        return {}

    result_files = sorted(eval_dir.glob("eval_*.json"))
    if not result_files:
        return {}

    latest = result_files[-1]
    try:
        with open(latest) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    raw_results = data.get("results", {})
    scores = {}
    for task_key, meta in BENCHMARK_META.items():
        task_name = {
            "hellaswag":     "hellaswag",
            "arc_easy":      "arc_easy",
            "arc_challenge": "arc_challenge",
            "mmlu":          "mmlu",
            "truthfulqa":    "truthfulqa_mc2",
            "humaneval":     "humaneval",
        }[task_key]
        if task_name in raw_results:
            metric = meta["metric"]
            score = (
                raw_results[task_name].get(metric)
                or raw_results[task_name].get(f"{metric},none")
                or raw_results[task_name].get(f"{metric},create_test")
            )
            if isinstance(score, (int, float)):
                scores[task_key] = score

    return scores


def _format_eval_table(scores: dict) -> str:
    """Format eval scores as a markdown table."""
    if not scores:
        return "_Benchmark results will be added after evaluation._"

    lines = [
        "| Benchmark | Few-shot | Metric | Score |",
        "|---|---|---|---|",
    ]
    for task_key, score in scores.items():
        meta = BENCHMARK_META[task_key]
        lines.append(
            f"| {meta['name']} | {meta['few_shot']}-shot | {meta['metric']} | {score:.4f} |"
        )
    return "\n".join(lines)


# ── Model card ────────────────────────────────────────────────────────────────

def generate_model_card(
    size: str,
    variant: str,
    hub_name: str,
    n_params: int,
    eval_scores: dict,
) -> str:
    """
    Generate a fully populated model card for the HuggingFace Hub.

    Args:
        size:        Model size string (125m, 350m, 1b).
        variant:     Model variant (base, instruct, chat).
        hub_name:    HuggingFace repo name (e.g. slm-125m-chat).
        n_params:    Actual parameter count from the loaded model.
        eval_scores: Benchmark scores from the most recent eval run.
                     Only populated for the chat variant (post-DPO).
    """
    size_upper    = size.upper()
    variant_cfg   = VARIANTS[variant]
    description   = variant_cfg["description"]
    pipeline_tag  = variant_cfg["pipeline_tag"]
    token_target  = _token_target(size)
    param_str     = f"{n_params / 1e6:.1f}M ({n_params:,} parameters)"

    if variant == "base":
        base_model_yaml = ""
    else:
        base_model_yaml = f"base_model: {HF_USERNAME}/slm-{size}"

    variant_section = {
        "base": f"""\
This is the **base** variant — pretrained on {token_target} tokens with no fine-tuning.
It is suitable for research and as a starting point for further fine-tuning.
Use [`{HF_USERNAME}/slm-{size}-instruct`](https://huggingface.co/{HF_USERNAME}/slm-{size}-instruct) for instruction following or
[`{HF_USERNAME}/slm-{size}-chat`](https://huggingface.co/{HF_USERNAME}/slm-{size}-chat) for aligned conversation.
""",
        "instruct": f"""\
This is the **instruct** variant — the base model supervised fine-tuned on chat and code instruction datasets.
It follows instructions reliably and can generate Python code.
Use [`{HF_USERNAME}/slm-{size}-chat`](https://huggingface.co/{HF_USERNAME}/slm-{size}-chat) for the DPO-aligned version preferred for open-ended conversation.
Use [`{HF_USERNAME}/slm-{size}`](https://huggingface.co/{HF_USERNAME}/slm-{size}) for the raw base model.
""",
        "chat": f"""\
This is the **chat** variant — the instruct model further aligned via Direct Preference Optimization (DPO).
This is the recommended variant for conversational and assistant use cases.
Use [`{HF_USERNAME}/slm-{size}-instruct`](https://huggingface.co/{HF_USERNAME}/slm-{size}-instruct) for the SFT-only version.
Use [`{HF_USERNAME}/slm-{size}`](https://huggingface.co/{HF_USERNAME}/slm-{size}) for the raw base model.
""",
    }[variant]

    # Source mix updated to 55/25/20 CC/Wikipedia/Python
    training_section = {
        "base": f"""\
| Stage | Dataset | Size |
|---|---|---|
| Pretraining | Wikipedia EN (25%) + [CodeSearchNet Python](https://huggingface.co/datasets/code_search_net) (20%) + [Common Crawl](https://commoncrawl.org) (55%) | {token_target} tokens |
""",
        "instruct": f"""\
| Stage | Dataset | Size |
|---|---|---|
| Pretraining | Wikipedia EN (25%) + [CodeSearchNet Python](https://huggingface.co/datasets/code_search_net) (20%) + [Common Crawl](https://commoncrawl.org) (55%) | {token_target} tokens |
| Chat SFT | [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~1M examples |
| Code SFT | [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | ~75K examples |
""",
        "chat": f"""\
| Stage | Dataset | Size |
|---|---|---|
| Pretraining | Wikipedia EN (25%) + [CodeSearchNet Python](https://huggingface.co/datasets/code_search_net) (20%) + [Common Crawl](https://commoncrawl.org) (55%) | {token_target} tokens |
| Chat SFT | [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~1M examples |
| Code SFT | [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | ~75K examples |
| DPO alignment | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) + [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) + [argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k) | ~80K pairs |
""",
    }[variant]

    eval_section = ""
    if variant == "chat":
        eval_section = f"""
## Evaluation

Evaluated using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

{_format_eval_table(eval_scores)}
"""

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
{base_model_yaml}
---

# {hub_name}

A {size_upper} decoder-only language model ({description}). Part of the SLM model family —
built entirely from scratch, from raw web data through to a production-ready aligned model.

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
| Positional encoding | RoPE | Better length generalisation, relative position awareness |
| Normalization | RMSNorm | Faster than LayerNorm, modern standard |
| Activation | SwiGLU | Better gradient flow, used by LLaMA and Mistral |
| Attention | GQA | Reduces KV cache memory at inference |
| Bias | None | Simpler, modern standard |
| Embeddings | Tied | Reduces parameters, effective at small scale |
| Vocab size | 32,000 | Custom BPE tokenizer trained on the pretraining corpus |
| Parameters | {param_str} | |

## Training

{training_section}

**Hardware:** NVIDIA H200 (pretraining on 8× H200, fine-tuning on 1× H200)
{eval_section}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{HF_USERNAME}/{hub_name}")
tokenizer = AutoTokenizer.from_pretrained("{HF_USERNAME}/{hub_name}")

messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Explain what a transformer is."}},
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
)
output = model.generate(inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Limitations

- **Scale:** At {size_upper} parameters this model is significantly smaller than frontier models. It will underperform on complex reasoning, long-context tasks, and domains not well-represented in the pretraining data.
- **Hallucination:** Like all language models, this model can generate plausible-sounding but factually incorrect content. Outputs should not be used as a source of truth without independent verification.
- **Safety:** DPO alignment provides basic harmlessness training but does not guarantee safe outputs in all contexts. This model has not undergone red-teaming or adversarial safety evaluation.
- **Languages:** Training data is predominantly English. Performance on other languages will be significantly degraded.
- **Code:** Code generation is Python-only, reflecting the pretraining and SFT data distribution.

## Related

- [slm](https://github.com/tohio/slm) — full training pipeline (data curation through serving)
- [ai-infra](https://github.com/tohio/ai-infra) — production Kubernetes serving via vLLM
"""


def _token_target(size: str) -> str:
    """Token targets matching TARGET_CONFIGS in curator/scripts/curate.py."""
    return {"125m": "5B", "350m": "15B", "1b": "30B"}.get(size, "N/A")


# ── Tokenizer loader ──────────────────────────────────────────────────────────

def load_tokenizer(tokenizer_path: Path):
    """
    Load the HuggingFace tokenizer saved by train_tokenizer.py.

    Loads directly via PreTrainedTokenizerFast.from_pretrained() which
    reads the saved tokenizer_config.json — including the baked-in
    chat_template. Do not reconstruct from tokenizer.json and re-set
    the template manually, as that would overwrite the saved template
    with whatever is in this file at export time rather than what was
    used during training.
    """
    from transformers import PreTrainedTokenizerFast

    if not (tokenizer_path / "tokenizer_config.json").exists():
        raise FileNotFoundError(
            f"HuggingFace tokenizer not found at {tokenizer_path}. "
            f"Run: python tokenizer/train_tokenizer.py"
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    # Verify the chat template was loaded correctly — fail loudly if missing
    # rather than silently exporting a model with a broken chat template.
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer at {tokenizer_path} has no chat_template. "
            f"Retrain the tokenizer: python tokenizer/train_tokenizer.py"
        )

    return tokenizer


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
    from transformers import AutoConfig, AutoModelForCausalLM
    from huggingface_hub import HfApi, login
    from model import SLMConfig, SLMForCausalLM

    variant_cfg = VARIANTS[variant]
    checkpoint  = model_path or variant_cfg["checkpoint"](size)
    hub_suffix  = variant_cfg["hub_suffix"]
    hub_name    = f"slm-{size}{hub_suffix}"
    repo_id     = f"{HF_USERNAME}/{hub_name}"

    log.info(f"=== SLM Export ===")
    log.info(f"Size:       {size}")
    log.info(f"Variant:    {variant}")
    log.info(f"Checkpoint: {checkpoint}")
    log.info(f"Hub:        {repo_id}")
    log.info(f"Dry run:    {dry_run}")

    if not checkpoint.exists():
        log.error(f"Checkpoint not found: {checkpoint}")
        log.error(f"Run the full training pipeline first: make pretrain sft sft-code dpo SIZE={size}")
        sys.exit(1)

    # Register custom model
    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    # Load model
    log.info("Loading model...")
    config   = SLMConfig.from_pretrained(str(checkpoint))
    model    = SLMForCausalLM.from_pretrained(str(checkpoint))
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Load tokenizer — from_pretrained reads the saved tokenizer_config.json
    # including the baked-in chat_template from train_tokenizer.py
    tokenizer_path = checkpoint / "tokenizer"
    if not (tokenizer_path / "tokenizer_config.json").exists():
        tokenizer_path = Path(os.environ.get("DATA_DIR", "data")) / "tokenizer"
    tokenizer = load_tokenizer(tokenizer_path)
    log.info(f"Tokenizer loaded from {tokenizer_path}")

    # Load eval results
    eval_scores = load_eval_results(size) if variant == "chat" else {}
    if variant == "chat":
        if eval_scores:
            log.info(f"Eval results loaded: {list(eval_scores.keys())}")
        else:
            log.warning("No eval results found — benchmark table will be empty in model card")
            log.warning(f"Run: make eval SIZE={size} before exporting")

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
    model_card = generate_model_card(size, variant, hub_name, n_params, eval_scores)
    card_path  = checkpoint / "README.md"
    with open(card_path, "w") as f:
        f.write(model_card)
    log.info(f"Model card written ({len(model_card):,} chars)")

    # Push to Hub
    log.info(f"Pushing to {repo_id}...")
    model.push_to_hub(repo_id, token=HF_TOKEN, private=private)
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

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=20,
            pad_token_id=0,
            eos_token_id=[3, 7],
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    log.info(f"Validation output: {decoded[:100]}")
    log.info("✓ Model validation passed")


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
        """,
    )
    parser.add_argument("--size",    type=str,  required=True, choices=["125m", "350m", "1b"])
    parser.add_argument("--variant", type=str,  required=True, choices=list(VARIANTS.keys()),
                        help="base (pretrain only) | instruct (SFT) | chat (SFT + DPO)")
    parser.add_argument("--model",   type=Path, default=None,
                        help="Override checkpoint path (defaults to variant mapping)")
    parser.add_argument("--dry-run", action="store_true", help="Validate without pushing to Hub")
    parser.add_argument("--private", action="store_true", help="Create private Hub repository")
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