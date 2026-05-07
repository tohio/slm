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
single source of truth for design intent. The model card additionally
loads data/curated/blend_stats.json (if present and matching --size) to
render the realized per-source breakdown alongside the design targets,
so the published card reflects what actually shipped — not just what
was planned. Falls back to design-only with a caveat note if blend_stats
is missing or scale-mismatched.

Eval results come from the most recent JSON written by eval/eval.py for
the matching checkpoint. Each variant is evaluated against its own
checkpoint directory, so the Hub tables reflect real scores for that
specific variant.

Remote-code bundling:
    SLMConfig declares auto_map pointing at config.SLMConfig and
    model.SLMForCausalLM (flat layout, no subpackage). The architecture
    .py files (config.py, model.py, attention.py, block.py, mlp.py,
    norm.py) are written into the checkpoint root by
    SLMForCausalLM.save_pretrained() during training, so by export time
    they're already present alongside the weights. The export step
    validates their presence rather than re-bundling.

    The flat layout works because transformers' dynamic module loader
    namespaces remote code under transformers_modules.<sanitized_name>.*,
    so the bare module names `config` / `model` don't leak into the host's
    sys.path.
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

from eval.eval import metric_score                                 # noqa: E402
from config import (                                                # noqa: E402
    DATA_MIX, CODE_SUBMIX, dataset_link, corpus_tokens_display,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HF_USERNAME = os.environ.get("HF_USERNAME")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
DATA_DIR    = Path(os.environ.get("DATA_DIR", "data"))

# blend_stats.json is written by curator/scripts/curate.py at the end of the
# blend stage. Reading from data/curated/ matches the curator's output
# location regardless of how DATA_DIR is set.
BLEND_STATS_PATH = DATA_DIR / "curated" / "blend_stats.json"

REPO_ROOT   = Path(__file__).resolve().parents[1]
MODEL_PKG_DIR = REPO_ROOT / "model"

# Architecture source files expected at the checkpoint root. Listed
# explicitly so _validate_bundled_files can detect a checkpoint that
# wasn't saved through the SLMForCausalLM.save_pretrained() override.
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


def _load_blend_stats(size: str) -> dict | None:
    """
    Load data/curated/blend_stats.json if present and matching this size.

    Returns the parsed dict, or None if:
      - file doesn't exist (curator hasn't run, or blend output not on this host)
      - file is unreadable / malformed
      - file's `target` field doesn't match the size we're exporting
        (avoids shipping wrong numbers when blend_stats is from a different scale)
    """
    if not BLEND_STATS_PATH.exists():
        log.info(
            f"blend_stats.json not found at {BLEND_STATS_PATH} — "
            f"model card will use design targets only."
        )
        return None

    try:
        with open(BLEND_STATS_PATH) as f:
            stats = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to read {BLEND_STATS_PATH}: {e}")
        return None

    blend_target = stats.get("target")
    if blend_target != size:
        log.warning(
            f"blend_stats.json target={blend_target!r} does not match "
            f"--size {size!r}. Falling back to design targets only — "
            f"re-curate at the matching scale to publish realized numbers."
        )
        return None

    return stats


def _format_data_mix_table(size: str) -> str:
    """
    Render the pretraining data mix as a markdown table.

    If data/curated/blend_stats.json exists and matches `size`, the table
    shows both target % and realized % per source, so the model card
    reflects what actually shipped in the published corpus. Otherwise it
    falls back to a design-target-only view with a caveat note that the
    realized mix may differ.

    Top-level vs code sub-sources:
        DATA_MIX has a logical "code" bucket at 10% — the actual code
        sources live in CODE_SUBMIX. blend_stats.json's source_mix dict
        contains the 5 expanded code sub-sources (no "code" entry). To
        render correctly we expand "code" into its sub-sources here when
        rendering, with realized% pulled from blend_stats per-source.
    """
    stats = _load_blend_stats(size)

    if stats is None:
        # Design-only fallback: render DATA_MIX percentages without
        # realized numbers, plus a caveat that reality may have drifted.
        lines = [
            "| Source | Target Share | Link |",
            "|---|---|---|",
        ]
        for name, entry in DATA_MIX.items():
            lines.append(f"| `{name}` | {entry['pct']:.1f}% | {dataset_link(entry)} |")
        lines.append("")
        lines.append(
            "> _Realized mix may differ from target — supply-bound sources "
            "(pes2o, jupyter at this scale) route their deficit to FineWeb_."
        )
        return "\n".join(lines)

    # Realized + target view. Compute per-source realized share from
    # the char totals in blend_stats.source_mix.
    source_mix = stats.get("source_mix", {})
    total_chars = sum(v.get("chars", 0) for v in source_mix.values())
    if total_chars == 0:
        # Defensive: shouldn't happen for a valid blend, but if chars sum
        # to zero we can't compute percentages — fall back to design-only
        # rather than print all zeros.
        log.warning("blend_stats.source_mix has zero total chars — using design targets only")
        return _format_data_mix_table_design_only()

    lines = [
        "| Source | Target Share | Realized Share | Link |",
        "|---|---|---|---|",
    ]

    # Top-level non-code sources from DATA_MIX, in declaration order.
    for name, entry in DATA_MIX.items():
        if name == "code":
            # Expand code into its sub-sources below, not as a single line.
            continue
        realized_chars = source_mix.get(name, {}).get("chars", 0)
        realized_pct = (realized_chars / total_chars) * 100
        lines.append(
            f"| `{name}` | {entry['pct']:.1f}% | {realized_pct:.2f}% | "
            f"{dataset_link(entry)} |"
        )

    # Code sub-sources, each as its own row. Their target % is
    # CODE_SUBMIX[name].pct of the 10% code share.
    code_top_pct = DATA_MIX["code"]["pct"]
    for name, entry in CODE_SUBMIX.items():
        target_pct_of_total = (entry["pct"] / 100.0) * code_top_pct
        realized_chars = source_mix.get(name, {}).get("chars", 0)
        realized_pct = (realized_chars / total_chars) * 100
        lines.append(
            f"| `{name}` | {target_pct_of_total:.2f}% | {realized_pct:.2f}% | "
            f"{dataset_link(entry)} |"
        )

    # Footer line summarising the realized totals so readers don't have
    # to add the column themselves.
    estimated_tokens = stats.get("estimated_tokens", 0)
    train_docs = stats.get("train_documents", 0)
    val_docs = stats.get("val_documents", 0)
    lines.append("")
    lines.append(
        f"_Realized: ~{estimated_tokens / 1e9:.2f}B tokens "
        f"({train_docs:,} train + {val_docs:,} val docs). "
        f"Supply-bound sources route their deficit to FineWeb._"
    )

    return "\n".join(lines)


def _format_data_mix_table_design_only() -> str:
    """
    Render DATA_MIX as a design-only table. Used as a defensive fallback
    inside _format_data_mix_table when blend_stats has zero chars.
    """
    lines = [
        "| Source | Target Share | Link |",
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
    token_tgt     = corpus_tokens_display(size)
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

    pretrain_table = _format_data_mix_table(size)

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
| Response-control SFT | Generated locally by `finetune/data/response_control.py` | 5K examples |
| Code SFT | [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) + handcrafted body-only completions | ~75K examples + small handcrafted set |
""",
        "chat": f"""\
**Pretraining corpus** — {token_tgt} tokens blended across the following sources:

{pretrain_table}

**Fine-tuning and alignment**

| Stage | Dataset | Size |
|---|---|---|
| Chat SFT | [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~1M examples |
| Response-control SFT | Generated locally by `finetune/data/response_control.py` | 5K examples |
| Code SFT | [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) + handcrafted body-only completions | ~75K examples + small handcrafted set |
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
| Instruct | [{HF_USERNAME}/slm-{size}-instruct](https://huggingface.co/{HF_USERNAME}/slm-{size}-instruct) | Chat + response-control + code SFT |
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

**Hardware:** NVIDIA H200 (pretraining on 1× H200, fine-tuning on 1× H200)
{eval_section}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{HF_USERNAME}/{hub_name}",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "{HF_USERNAME}/{hub_name}",
    trust_remote_code=True,
)

messages = [
    {{"role": "system", "content": "Answer clearly and concisely."}},
    {{"role": "user", "content": "Explain what a transformer is."}},
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True,
)

endofturn_id = tokenizer.convert_tokens_to_ids("<|endofturn|>")

output = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    eos_token_id=[tokenizer.eos_token_id, endofturn_id],
)

input_len = inputs["input_ids"].shape[1]
print(tokenizer.decode(output[0][input_len:], skip_special_tokens=True))
```

`trust_remote_code=True` loads the custom SLM architecture bundled alongside the model weights — no local install of the `tohio/slm` codebase required.

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


def _validate_bundled_files(checkpoint: Path) -> None:
    """
    Confirm architecture .py files at checkpoint root match the live source.

    Existence alone isn't enough — a stale bundled copy from an earlier
    training run can ship buggy code to the Hub silently. We compare bytes
    against the live source to catch drift.
    """
    import hashlib
    missing, mismatched = [], []
    for filename in BUNDLED_SOURCE_FILES:
        ckpt_file = checkpoint / filename
        live_file = MODEL_PKG_DIR / filename
        if not ckpt_file.is_file():
            missing.append(filename)
            continue
        if hashlib.sha256(ckpt_file.read_bytes()).digest() != \
           hashlib.sha256(live_file.read_bytes()).digest():
            mismatched.append(filename)

    if missing:
        raise FileNotFoundError(
            f"Architecture files missing from {checkpoint}: {missing}. "
            f"Re-save the checkpoint or run the relevant training stage."
        )
    if mismatched:
        raise RuntimeError(
            f"Bundled architecture files at {checkpoint} differ from live "
            f"source: {mismatched}. Re-sync via "
            f"`cp model/<file>.py {checkpoint}/<file>.py` for each, or "
            f"re-save the checkpoint."
        )
    log.info(
        f"Bundled remote code validated: "
        f"{len(BUNDLED_SOURCE_FILES)} files match live source"
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
        # Confirm architecture files are bundled at the checkpoint root —
        # catches missing files before a real push is attempted.
        _validate_bundled_files(checkpoint)
        card = generate_model_card(size, variant, hub_name, n_params, eval_scores)
        log.info(f"Model card preview ({len(card):,} chars, first 400):")
        log.info(card[:400].replace("\n", "\n  "))
        return

    if not HF_TOKEN:
        log.error("HF_TOKEN not set in .env")
        sys.exit(1)
    login(token=HF_TOKEN)

    _validate_model(model, tokenizer, config)

    # Confirm architecture files are bundled at the checkpoint root.
    # SLMForCausalLM.save_pretrained() writes them automatically during
    # training; this validates rather than re-bundles since training
    # already did the work.
    _validate_bundled_files(checkpoint)

    model_card = generate_model_card(size, variant, hub_name, n_params, eval_scores)
    card_path  = checkpoint / "README.md"
    with open(card_path, "w") as f:
        f.write(model_card)
    log.info(f"Model card written to {card_path} ({len(model_card):,} chars)")

    # Single-commit push of the entire checkpoint dir — weights, config
    # (with auto_map), tokenizer, README.md, and the bundled .py files.
    # Using push_to_hub on the model would omit the README and any other
    # files at the checkpoint root, so we upload the folder directly.
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


def _too_repetitive(tokens: list[int], max_repeat_run: int = 8) -> bool:
    """Return True when generation contains an obvious repeated-token run.

    This is intentionally conservative: normal text can repeat words, but a
    healthy checkpoint should not emit the same token 8+ times in a row during
    a short export validation prompt. The check applies to every model size and
    variant as export hygiene, not just to 125M.
    """
    if not tokens:
        return True

    run = 1
    for prev, cur in zip(tokens, tokens[1:]):
        if cur == prev:
            run += 1
            if run >= max_repeat_run:
                return True
        else:
            run = 1

    return False


def _as_eos_id_list(eos_ids) -> list[int]:
    """Normalize an int/list/tuple EOS config to a clean list of IDs."""
    if eos_ids is None:
        return []
    if isinstance(eos_ids, int):
        return [eos_ids]
    return [int(eos_id) for eos_id in eos_ids if eos_id is not None]


def _validate_model(model, tokenizer, config) -> None:
    """Generate a short sequence and reject empty or degenerate output."""
    import torch
    from inference.utils import resolve_special_token_ids

    log.info("Validating model...")
    model.eval()
    special_ids = resolve_special_token_ids(tokenizer)

    messages = [
        {"role": "system", "content": "Answer clearly and concisely."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    # apply_chat_template can return either a Tensor or a BatchEncoding
    # depending on transformers version / tokenizer config — normalize.
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    input_length = input_ids.shape[1]

    eos_ids = _as_eos_id_list(special_ids.eos_list)
    endofturn_id = tokenizer.convert_tokens_to_ids("<|endofturn|>")
    if isinstance(endofturn_id, int) and endofturn_id >= 0 and endofturn_id not in eos_ids:
        eos_ids.append(endofturn_id)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=special_ids.pad,
            eos_token_id=eos_ids,
        )

    new_tokens = output[0][input_length:].tolist()
    for stop_id in eos_ids:
        if stop_id in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(stop_id)]

    if len(new_tokens) == 0:
        raise RuntimeError(
            "Validation failed: model produced only stop tokens. "
            "This suggests the checkpoint is broken (e.g. NaN weights, "
            "wrong tied-weight restore, corrupted save). Aborting export."
        )

    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if _too_repetitive(new_tokens):
        raise RuntimeError(
            "Validation failed: model produced highly repetitive output. "
            f"Decoded output: {decoded[:200]!r}"
        )

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