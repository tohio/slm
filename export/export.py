"""
export/export.py
-----------------
Export a trained SLM checkpoint to the HuggingFace Hub.

Registers the custom SLMConfig and SLMForCausalLM with AutoConfig and
AutoModelForCausalLM so the model can be loaded anywhere with:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "<username>/slm-125m", trust_remote_code=True,
    )

Three variants are exported per model size:

    Variant     Checkpoint                                    Hub repo
    --------    ----------                                    --------
    base        results/slm-{size}/final                      <user>/slm-{size}
    instruct    results/slm-{size}-chat-code/final            <user>/slm-{size}-instruct
    chat        results/slm-{size}-dpo/final                  <user>/slm-{size}-chat

Data mix and token targets are imported from config/data_mix.py — the
single source of truth shared with curator, notebooks, and tests.

Eval results come from the most recent JSON written by eval/eval.py for
the matching checkpoint. Each variant is evaluated against its own
checkpoint directory, so the Hub tables reflect real scores for that
specific variant.

Remote-code bundling:
    SLMConfig declares auto_map pointing at a `slm_arch` subpackage. The
    push below copies the entire local `model/` folder into the checkpoint
    staging dir as `slm_arch/` before uploading, so that loaders calling
    trust_remote_code=True find SLMConfig / SLMModel / SLMForCausalLM on
    the Hub without needing this repo installed locally.

    We use `slm_arch` as the bundled package name rather than `model`
    because `model` is too generic to collide-safely on sys.path when
    transformers dynamically imports remote code.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.eval import metric_score                                 # noqa: E402
from config import DATA_MIX, dataset_link, token_target_display    # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HF_USERNAME = os.environ.get("HF_USERNAME")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))

REPO_ROOT   = Path(__file__).resolve().parents[1]
MODEL_PKG_DIR = REPO_ROOT / "model"

# Name of the subpackage bundled into Hub repos. Must match the prefix
# in SLMConfig.auto_map. If you change this, change both places.
BUNDLED_PKG_NAME = "slm_arch"

# Source files copied from model/ into the Hub repo's slm_arch/ subpackage.
# Listed explicitly rather than globbed so that auxiliary files (tests,
# scratch files, __pycache__) don't accidentally get pushed to the Hub.
BUNDLED_SOURCE_FILES = [
    "config.py",
    "model.py",
    "block.py",
    "attention.py",
    "mlp.py",
    "norm.py",
]


VARIANTS: dict[str, dict] = {
    "base": {
        "checkpoint":    lambda size: RESULTS_DIR / f"slm-{size}" / "final",
        "eval_dir":      lambda size: RESULTS_DIR / "eval" / f"slm-{size}",
        "hub_suffix":    "",
        "description":   "base pretrained model",
        "pipeline_tag":  "text-generation",
    },
    "instruct": {
        "checkpoint":    lambda size: RESULTS_DIR / f"slm-{size}-chat-code" / "final",
        "eval_dir":      lambda size: RESULTS_DIR / "eval" / f"slm-{size}-chat-code",
        "hub_suffix":    "-instruct",
        "description":   "instruction-tuned via chat SFT + code SFT",
        "pipeline_tag":  "text-generation",
    },
    "chat": {
        "checkpoint":    lambda size: RESULTS_DIR / f"slm-{size}-dpo" / "final",
        "eval_dir":      lambda size: RESULTS_DIR / "eval" / f"slm-{size}-dpo",
        "hub_suffix":    "-chat",
        "description":   "chat-aligned via SFT + DPO preference learning",
        "pipeline_tag":  "text-generation",
    },
}


BENCHMARK_META = {
    "hellaswag":     {"name": "HellaSwag",     "task": "hellaswag",      "metric": "acc_norm", "few_shot": 10},
    "arc_easy":      {"name": "ARC-Easy",      "task": "arc_easy",       "metric": "acc_norm", "few_shot": 25},
    "arc_challenge": {"name": "ARC-Challenge", "task": "arc_challenge",  "metric": "acc_norm", "few_shot": 25},
    "mmlu":          {"name": "MMLU",          "task": "mmlu",           "metric": "acc",      "few_shot": 5},
    "truthfulqa":    {"name": "TruthfulQA",    "task": "truthfulqa_mc2", "metric": "acc",      "few_shot": 0},
    "humaneval":     {"name": "HumanEval",     "task": "humaneval",      "metric": "pass@1",   "few_shot": 0},
}


def load_eval_results(variant: str, size: str) -> dict[str, float]:
    """Load the most recent eval results for this variant's checkpoint."""
    eval_dir = VARIANTS[variant]["eval_dir"](size)
    if not eval_dir.exists():
        return {}

    result_files = sorted(eval_dir.glob("eval_*.json"))
    if not result_files:
        return {}

    latest = result_files[-1]
    try:
        with open(latest) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to read eval results from {latest}: {e}")
        return {}

    raw_results = data.get("results", {})
    raw_groups  = data.get("groups", {})

    scores: dict[str, float] = {}
    for task_key, meta in BENCHMARK_META.items():
        task_name   = meta["task"]
        metric      = meta["metric"]
        task_result = raw_results.get(task_name) or raw_groups.get(task_name)
        if task_result is None:
            continue
        score = metric_score(task_result, metric)
        if isinstance(score, (int, float)):
            scores[task_key] = float(score)

    return scores


def _format_eval_table(scores: dict[str, float]) -> str:
    if not scores:
        return "_Benchmark results will be added after evaluation._"

    lines = [
        "| Benchmark | Few-shot | Metric | Score |",
        "|---|---|---|---|",
    ]
    for task_key in BENCHMARK_META:
        if task_key not in scores:
            continue
        meta = BENCHMARK_META[task_key]
        lines.append(
            f"| {meta['name']} | {meta['few_shot']}-shot | {meta['metric']} | {scores[task_key]:.4f} |"
        )
    return "\n".join(lines)


def _format_data_mix_table() -> str:
    """Render DATA_MIX from config/data_mix.py as a markdown table."""
    lines = [
        "| Source | Share | Link |",
        "|---|---|---|",
    ]
    for name, entry in DATA_MIX.items():
        lines.append(f"| `{name}` | {entry['pct']:.1f}% | {dataset_link(entry)} |")
    return "\n".join(lines)


def generate_model_card(
    size: str,
    variant: str,
    hub_name: str,
    n_params: int,
    eval_scores: dict[str, float],
) -> str:
    size_upper    = size.upper()
    variant_cfg   = VARIANTS[variant]
    description   = variant_cfg["description"]
    pipeline_tag  = variant_cfg["pipeline_tag"]
    token_tgt     = token_target_display(size)
    param_str     = f"{n_params / 1e6:.1f}M ({n_params:,} parameters)"

    base_model_yaml = "" if variant == "base" else f"base_model: {HF_USERNAME}/slm-{size}"

    variant_section = {
        "base": f"""\
This is the **base** variant — pretrained on {token_tgt} tokens with no fine-tuning.
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

    pretrain_table = _format_data_mix_table()

    training_section = {
        "base": f"""\
**Pretraining corpus** — {token_tgt} tokens blended across the following sources:

{pretrain_table}
""",
        "instruct": f"""\
**Pretraining corpus** — {token_tgt} tokens blended across the following sources:

{pretrain_table}

**Fine-tuning**

| Stage | Dataset | Size |
|---|---|---|
| Chat SFT | [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~1M examples |
| Code SFT | [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | ~75K examples |
""",
        "chat": f"""\
**Pretraining corpus** — {token_tgt} tokens blended across the following sources:

{pretrain_table}

**Fine-tuning and alignment**

| Stage | Dataset | Size |
|---|---|---|
| Chat SFT | [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~1M examples |
| Code SFT | [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | ~75K examples |
| DPO alignment | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) + [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) + [argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k) | ~60K pairs after filtering |
""",
    }[variant]

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

model = AutoModelForCausalLM.from_pretrained(
    "{HF_USERNAME}/{hub_name}",
    trust_remote_code=True,
)
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

`trust_remote_code=True` loads the custom SLM architecture bundled in the `slm_arch/` subpackage of this repo — no local install of the `tohio/slm` codebase required.

## Limitations

- **Scale:** At {size_upper} parameters this model is significantly smaller than frontier models. It will underperform on complex reasoning, long-context tasks, and domains not well-represented in the pretraining data.
- **Hallucination:** Like all language models, this model can generate plausible-sounding but factually incorrect content. Outputs should not be used as a source of truth without independent verification.
- **Safety:** DPO alignment provides basic harmlessness training but does not guarantee safe outputs in all contexts. This model has not undergone red-teaming or adversarial safety evaluation.
- **Languages:** Training data is predominantly English. Performance on other languages will be significantly degraded.
- **Code:** Code generation is primarily Python-oriented, reflecting the code sub-mix distribution used in pretraining and SFT.

## Related

- [slm](https://github.com/tohio/slm) — full training pipeline (data curation through serving)
- [ai-infra](https://github.com/tohio/ai-infra) — production Kubernetes serving via vLLM
"""


def load_tokenizer(tokenizer_path: Path):
    """Load tokenizer via PreTrainedTokenizerFast — never reconstruct."""
    from transformers import PreTrainedTokenizerFast

    if not (tokenizer_path / "tokenizer_config.json").exists():
        raise FileNotFoundError(
            f"HuggingFace tokenizer not found at {tokenizer_path}. "
            f"Run: python tokenizer/train_tokenizer.py"
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer at {tokenizer_path} has no chat_template. "
            f"Retrain the tokenizer: python tokenizer/train_tokenizer.py"
        )

    return tokenizer


def _bundle_remote_code(checkpoint: Path) -> None:
    """
    Copy the model source into the checkpoint dir as `slm_arch/` so it
    gets pushed to the Hub in the single-commit upload, making the repo
    loadable with trust_remote_code=True on any machine.

    The bundled subpackage is a minimal re-export — deliberately different
    from the local model/__init__.py, which points users at AutoConfig
    registration. Inside remote code we must NOT call AutoConfig.register
    at import time, because transformers' trust_remote_code loader handles
    registration itself via auto_map and a double-register would collide.

    Raises if source files are missing — a broken bundle is worse than a
    missing one because it would be silently wrong at load time on the Hub.
    """
    dest = checkpoint / BUNDLED_PKG_NAME

    # Remove any stale bundle from a previous export — the checkpoint dir
    # is reused across runs, so old files would otherwise accumulate.
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    missing = []
    for filename in BUNDLED_SOURCE_FILES:
        src = MODEL_PKG_DIR / filename
        if not src.is_file():
            missing.append(str(src))
            continue
        shutil.copy2(src, dest / filename)
    if missing:
        raise FileNotFoundError(
            f"Cannot bundle remote code — missing source files: {missing}. "
            f"Expected in {MODEL_PKG_DIR}. "
            f"This would push a broken Hub repo; aborting."
        )

    # Minimal __init__.py — re-exports only, no AutoConfig.register calls.
    (dest / "__init__.py").write_text(
        '"""\n'
        f'{BUNDLED_PKG_NAME}/\n'
        '-' * len(BUNDLED_PKG_NAME) + '-\n'
        'Bundled copy of the SLM model source, shipped inside the Hub repo\n'
        'so that AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)\n'
        'can load the model without a local install of tohio/slm.\n'
        '\n'
        'Do not edit this file by hand — it is written by export/export.py.\n'
        'The source of truth lives at model/ in the tohio/slm repo.\n'
        '"""\n'
        '\n'
        'from .config import SLMConfig\n'
        'from .model import SLMForCausalLM, SLMModel\n'
        '\n'
        '__all__ = ["SLMConfig", "SLMModel", "SLMForCausalLM"]\n'
    )

    log.info(
        f"Bundled remote code: {len(BUNDLED_SOURCE_FILES)} files + __init__.py -> "
        f"{dest.relative_to(checkpoint)}/"
    )


def export(
    size: str,
    variant: str,
    model_path: Path | None = None,
    dry_run: bool = False,
    private: bool = False,
) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM
    from huggingface_hub import login
    from model import SLMConfig, SLMForCausalLM

    if not HF_USERNAME:
        log.error(
            "HF_USERNAME not set in the environment. "
            "Add HF_USERNAME=<your-hub-username> to .env before running export."
        )
        sys.exit(1)

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
        log.error(
            f"Run the training pipeline first. For chat variant: "
            f"make pretrain sft sft-code dpo SIZE={size}"
        )
        sys.exit(1)

    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    log.info("Loading model...")
    config   = SLMConfig.from_pretrained(str(checkpoint))
    model    = SLMForCausalLM.from_pretrained(str(checkpoint))
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    tokenizer_path = checkpoint / "tokenizer"
    if not (tokenizer_path / "tokenizer_config.json").exists():
        tokenizer_path = Path(os.environ.get("DATA_DIR", "data")) / "tokenizer"
    tokenizer = load_tokenizer(tokenizer_path)
    log.info(f"Tokenizer loaded from {tokenizer_path}")

    eval_scores = load_eval_results(variant, size)
    if eval_scores:
        log.info(f"Eval results loaded: {list(eval_scores.keys())}")
    else:
        log.warning(
            f"No eval results found for variant={variant}, size={size}. "
            f"Benchmark table will be empty. Run: "
            f"python eval/eval.py --model {variant_cfg['checkpoint'](size)}"
        )

    if dry_run:
        log.info("Dry run — skipping Hub push")
        log.info(f"Would push to: https://huggingface.co/{repo_id}")
        _validate_model(model, tokenizer, config)
        # Exercise the bundling path in dry-run too — catches missing source
        # files before a real push is attempted.
        _bundle_remote_code(checkpoint)
        card = generate_model_card(size, variant, hub_name, n_params, eval_scores)
        log.info(f"Model card preview ({len(card):,} chars, first 400):")
        log.info(card[:400].replace("\n", "\n  "))
        return

    if not HF_TOKEN:
        log.error("HF_TOKEN not set in .env")
        sys.exit(1)
    login(token=HF_TOKEN)

    _validate_model(model, tokenizer, config)

    # Bundle the custom architecture into the checkpoint dir before the
    # single-commit push, so the Hub repo contains slm_arch/ alongside the
    # weights and config. auto_map in config.json then resolves correctly
    # under trust_remote_code=True loads.
    _bundle_remote_code(checkpoint)

    model_card = generate_model_card(size, variant, hub_name, n_params, eval_scores)
    card_path  = checkpoint / "README.md"
    with open(card_path, "w") as f:
        f.write(model_card)
    log.info(f"Model card written to {card_path} ({len(model_card):,} chars)")

    # Single-commit push of the entire checkpoint dir — weights, config
    # (with auto_map), tokenizer, README.md, and the bundled slm_arch/.
    # Using push_to_hub on the model would omit the bundled subpackage and
    # README, so we upload the folder directly.
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    log.info(f"Pushing {checkpoint} to {repo_id} (single commit)...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(checkpoint),
        commit_message=f"Export {hub_name} ({n_params / 1e6:.1f}M params)",
        # Don't upload training-only artefacts even if they happen to be
        # in the checkpoint dir.
        ignore_patterns=[
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "training_args.bin",
            "rng_state*.pth",
            "global_step*",
            "*.log",
            "__pycache__",
            "*.pyc",
        ],
    )

    log.info(f"Export complete: https://huggingface.co/{repo_id}")


def _validate_model(model, tokenizer, config) -> None:
    """Generate a short sequence; assert non-empty output. Aborts export on failure."""
    import torch
    from inference.utils import resolve_special_token_ids

    log.info("Validating model...")
    model.eval()
    special_ids = resolve_special_token_ids(tokenizer)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    attention_mask = torch.ones_like(input_ids)
    input_length   = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            pad_token_id=special_ids.pad,
            eos_token_id=special_ids.eos_list,
        )

    new_tokens = output[0][input_length:].tolist()
    for stop_id in special_ids.eos_list:
        if stop_id in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(stop_id)]

    if len(new_tokens) == 0:
        raise RuntimeError(
            "Validation failed: model produced only stop tokens. "
            "This suggests the checkpoint is broken (e.g. NaN weights, "
            "wrong tied-weight restore, corrupted save). Aborting export."
        )

    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    log.info(f"Validation output ({len(new_tokens)} tokens): {decoded[:100]}")
    log.info("✓ Model validation passed")


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
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=list(VARIANTS.keys()),
        help="base (pretrain only) | instruct (SFT) | chat (SFT + DPO)",
    )
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