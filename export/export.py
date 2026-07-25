"""
export/export.py
-----------------
Export a trained SLM checkpoint to a native Hugging Face package.

Training checkpoints use the in-repository SLM implementation. Its module
layout and state-dict keys intentionally match Transformers' Llama decoder
contract, so export converts the configuration and saves the weights as a
native LlamaForCausalLM package. The published model loads anywhere with:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("<username>/slm-125m")

Four variants are exported per model size:

    Variant     Checkpoint                                    Hub repo
    --------    ----------                                    --------
    base        results/runs/{size}/pretrain/final                      <user>/slm-{size}
    instruct    results/runs/{size}/sft_instruct/final                      <user>/slm-{size}-instruct
    chat        results/runs/{size}/dpo_chat/final                           <user>/slm-{size}-chat
    code        results/runs/{size}/sft_code/final                      <user>/slm-{size}-code

Data mix and token targets are imported from config/data_mix.py — the
single source of truth for design intent. The model card additionally
loads data/runs/{size}/curated/blend_stats.json (if present and matching --size) to
render the realized per-source breakdown alongside the design targets,
so the published card reflects what actually shipped — not just what
was planned. Falls back to design-only with a caveat note if blend_stats
is missing or scale-mismatched.

The source training checkpoint is never mutated. Export writes a separate
artifact under results/exports/{size}/{variant}, validates source/native
logit and greedy-generation parity, then performs a clean AutoModel load with
trust_remote_code disabled before any Hub upload.
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

from config import (                                                # noqa: E402
    DATA_MIX, CODE_SUBMIX, dataset_link, corpus_tokens_display,
)
from config.paths import (                                          # noqa: E402
    curated_dir,
    dpo_chat_dir,
    export_dir as export_size_dir,
    metadata_dir,
    pretrain_dir,
    sft_code_dir,
    sft_instruct_dir,
    tokenizer_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HF_USERNAME = os.environ.get("HF_USERNAME")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")

OBSOLETE_REMOTE_CODE_PATTERNS = [
    "*.py",
    "**/*.py",
    "attention.py",
    "block.py",
    "config.py",
    "mlp.py",
    "model.py",
    "norm.py",
    "slm_remote/",
    "slm_remote/**",
]

# blend_stats.json is written by curator/scripts/curate.py at the end of the
# blend stage. Reading from data/runs/<size>/curated/ matches the curator's output
# location regardless of how DATA_DIR is set.
# Blend stats path is target-scoped; see _load_blend_stats(size).

VARIANTS: dict[str, dict] = {
    "base": {
        "checkpoint":    lambda size: pretrain_dir(size) / "final",
        "hub_suffix":    "",
        "description":   "base pretrained model",
        "pipeline_tag":  "text-generation",
    },
    "instruct": {
        "checkpoint":    lambda size: sft_instruct_dir(size) / "final",
        "hub_suffix":    "-instruct",
        "description":   "instruction-tuned via supervised fine-tuning",
        "pipeline_tag":  "text-generation",
    },
    "chat": {
        "checkpoint":    lambda size: dpo_chat_dir(size) / "final",
        "hub_suffix":    "-chat",
        "description":   "chat-aligned from instruct via general DPO preference learning",
        "pipeline_tag":  "text-generation",
    },
    "code": {
        "checkpoint":    lambda size: sft_code_dir(size) / "final",
        "hub_suffix":    "-code",
        "description":   "code-specialized from instruct via code SFT",
        "pipeline_tag":  "text-generation",
    },
}


def _load_blend_stats(size: str) -> dict:
    """Load curation blend stats when available.

    Preferred location:
      data/runs/{size}/metadata/blend_stats.json

    Legacy fallback:
      data/runs/{size}/curated/blend_stats.json
    """
    candidates = [
        metadata_dir(size) / "blend_stats.json",
        curated_dir(size) / "blend_stats.json",
    ]

    for blend_stats_path in candidates:
        if blend_stats_path.exists():
            with blend_stats_path.open("r", encoding="utf-8") as f:
                return json.load(f)

    return {}


def _format_data_mix_table(size: str) -> str:
    """
    Render the pretraining data mix as a markdown table.

    If data/runs/{size}/curated/blend_stats.json exists and matches `size`, the table
    shows both target % and realized % per source, so the model card
    reflects what actually shipped in the published corpus. Otherwise it
    falls back to a design-target-only view with a caveat note that the
    realized mix may differ.

    Top-level vs code sub-sources:
        DATA_MIX has a logical "code" bucket — the actual code
        sources live in CODE_SUBMIX. blend_stats.json's source_mix dict
        contains the 5 expanded code sub-sources (no "code" entry). To
        render correctly we expand "code" into its sub-sources here when
        rendering, with realized% pulled from blend_stats per-source.
    """
    stats = _load_blend_stats(size)

    if not stats:
        # Design-only fallback: render DATA_MIX percentages without
        # realized numbers, plus a caveat that reality may have drifted.
        lines = [
            "| Source | Target Share | Link |",
            "|---|---|---|",
        ]
        for name, entry in DATA_MIX.items():
            if name == "code":
                code_top_pct = entry["pct"]
                for code_name, code_entry in CODE_SUBMIX.items():
                    target_pct = (code_entry["pct"] / 100.0) * code_top_pct
                    lines.append(
                        f"| `{code_name}` | {target_pct:.2f}% | "
                        f"{dataset_link(code_entry)} |"
                    )
                continue

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
    # CODE_SUBMIX[name].pct of the current DATA_MIX['code'] share.
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
    estimated_tokens = stats.get("estimated_tokens_from_chars", 0)
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
    Render the design target table with concrete source rows.

    DATA_MIX contains a logical "code" bucket, so expand that bucket into
    CODE_SUBMIX rows instead of showing a single abstract code row.
    """
    lines = [
        "| Source | Target Share | Link |",
        "|---|---|---|",
    ]

    for name, entry in DATA_MIX.items():
        if name == "code":
            code_top_pct = entry["pct"]
            for code_name, code_entry in CODE_SUBMIX.items():
                target_pct = (code_entry["pct"] / 100.0) * code_top_pct
                lines.append(
                    f"| `{code_name}` | {target_pct:.2f}% | "
                    f"{dataset_link(code_entry)} |"
                )
            continue

        lines.append(f"| `{name}` | {entry['pct']:.1f}% | {dataset_link(entry)} |")

    return "\n".join(lines)


def _read_json(path: Path, label: str) -> dict:
    """Read a required JSON artifact with a useful export error."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found at {path}. Export will not guess training "
            "provenance; regenerate the checkpoint with the current trainer."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_dataset_row(label: str, manifest: dict) -> str:
    """Render one model-card row from an SFT/DPO data manifest."""
    contract = manifest.get("contract", {})
    source = contract.get("source", {})
    dataset = source.get("dataset")
    revision = source.get("revision")
    if not dataset or not revision:
        raise ValueError(
            f"{label} manifest is missing contract.source.dataset/revision"
        )

    files = manifest.get("files", {})
    records = sum(
        int(file_info.get("records", 0))
        for file_info in files.values()
        if isinstance(file_info, dict)
    )
    dataset_url = f"https://huggingface.co/datasets/{dataset}/tree/{revision}"
    return (
        f"| {label} | [{dataset}]({dataset_url}) | "
        f"`{revision[:12]}` | {records:,} |"
    )


def _parent_checkpoint(checkpoint: Path, audit_name: str) -> Path:
    """Resolve the exact parent checkpoint recorded by a training audit."""
    audit = _read_json(checkpoint / audit_name, f"{audit_name} training audit")
    parent = audit.get("base_model")
    if not parent:
        raise ValueError(f"{audit_name} does not record base_model")
    return Path(os.path.expandvars(parent))


def _fine_tuning_table(size: str, variant: str, checkpoint: Path) -> str:
    """Build model-card training rows from immutable preparation manifests."""
    if variant == "base":
        return ""

    rows = [
        "| Stage | Dataset | Revision | Prepared records |",
        "|---|---|---:|---:|",
    ]

    if variant == "instruct":
        instruct_checkpoint = checkpoint
    else:
        audit_name = "dpo_run_audit.json" if variant == "chat" else "sft_run_audit.json"
        instruct_checkpoint = _parent_checkpoint(checkpoint, audit_name)

    instruct_manifest = _read_json(
        instruct_checkpoint / "sft_data_manifest.json",
        "instruct SFT data manifest",
    )
    rows.append(_manifest_dataset_row("Instruct SFT", instruct_manifest))

    if variant == "code":
        code_manifest = _read_json(
            checkpoint / "sft_data_manifest.json",
            "code SFT data manifest",
        )
        rows.append(_manifest_dataset_row("Code SFT", code_manifest))
    elif variant == "chat":
        dpo_manifest = _read_json(
            checkpoint / "dpo_data_manifest.json",
            "DPO data manifest",
        )
        rows.append(_manifest_dataset_row("DPO alignment", dpo_manifest))

    return "\n".join(rows)


def generate_model_card(
    size: str,
    variant: str,
    hub_name: str,
    n_params: int,
    checkpoint: Path,
    config,
    hf_username: str,
) -> str:
    size_upper    = size.upper()
    variant_cfg   = VARIANTS[variant]
    description   = variant_cfg["description"]
    pipeline_tag  = variant_cfg["pipeline_tag"]
    token_tgt     = corpus_tokens_display(size)
    param_str     = f"{n_params / 1e6:.1f}M ({n_params:,} parameters)"

    if variant == "base":
        base_model_yaml = ""
    elif variant in {"chat", "code"}:
        base_model_yaml = f"base_model: {hf_username}/slm-{size}-instruct"
    else:
        base_model_yaml = f"base_model: {hf_username}/slm-{size}"

    variant_section = {
        "base": f"""\
This is the **base** variant — pretrained from a {token_tgt} curation target with no fine-tuning.
It is suitable for research and as a starting point for further fine-tuning.
Use [`{hf_username}/slm-{size}-instruct`](https://huggingface.co/{hf_username}/slm-{size}-instruct) for instruction following or
[`{hf_username}/slm-{size}-chat`](https://huggingface.co/{hf_username}/slm-{size}-chat) for aligned conversation.
""",
        "instruct": f"""\
This is the **instruct** variant — the base model supervised fine-tuned on general instruction data.
It is the sibling base for both the general chat-DPO branch and the code-specialized branch.
Use [`{hf_username}/slm-{size}-chat`](https://huggingface.co/{hf_username}/slm-{size}-chat) for the DPO-aligned version preferred for open-ended conversation.
Use [`{hf_username}/slm-{size}`](https://huggingface.co/{hf_username}/slm-{size}) for the raw base model.
""",
        "chat": f"""\
This is the **chat** variant — the instruct model further aligned via general Direct Preference Optimization (DPO).
This is the recommended variant for conversational and assistant use cases.
Use [`{hf_username}/slm-{size}-instruct`](https://huggingface.co/{hf_username}/slm-{size}-instruct) for the SFT-only version.
Use [`{hf_username}/slm-{size}`](https://huggingface.co/{hf_username}/slm-{size}) for the raw base model.
""",
        "code": f"""\
This is the **code** variant — the instruct model further specialized with code SFT.
Use [`{hf_username}/slm-{size}-chat`](https://huggingface.co/{hf_username}/slm-{size}-chat) for general assistant use.
""",
    }[variant]

    pretrain_table = _format_data_mix_table(size)

    fine_tuning_table = _fine_tuning_table(size, variant, checkpoint)
    training_section = {
        "base": f"""\
**Pretraining corpus** — {token_tgt} curation target blended across the following sources:

{pretrain_table}
""",
        "instruct": f"""\
**Pretraining corpus** — {token_tgt} curation target blended across the following sources:

{pretrain_table}

**Fine-tuning**

{fine_tuning_table}
""",
        "chat": f"""\
**Pretraining corpus** — {token_tgt} curation target blended across the following sources:

{pretrain_table}

**Fine-tuning and alignment**

{fine_tuning_table}
""",
        "code": f"""\
**Pretraining corpus** — {token_tgt} curation target blended across the following sources:

{pretrain_table}

**Fine-tuning**

{fine_tuning_table}
""",
    }[variant]

    if variant == "base":
        usage_section = f"""\
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{hf_username}/{hub_name}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
"""
    else:
        usage_section = f"""\
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{hf_username}/{hub_name}"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

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
output = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id,
)
input_len = inputs["input_ids"].shape[1]
print(tokenizer.decode(output[0][input_len:], skip_special_tokens=True))
```
"""

    return f"""---
license: mit
language:
  - en
library_name: transformers
pipeline_tag: {pipeline_tag}
tags:
  - causal-lm
  - decoder-only
  - llama-compatible
  - rope
  - gqa
  - swiglu
  - {variant}
{base_model_yaml}
---

# {hub_name}

A {size_upper} decoder-only language model ({description}) from the SLM model
family. This release is intended for research, evaluation, and controlled
experimentation.

## Intended Use

{variant_section}

## Model Family

| Variant | Hub | Description |
|---|---|---|
| Base | [{hf_username}/slm-{size}](https://huggingface.co/{hf_username}/slm-{size}) | Pretrained only |
| Instruct | [{hf_username}/slm-{size}-instruct](https://huggingface.co/{hf_username}/slm-{size}-instruct) | Instruct SFT |
| Chat | [{hf_username}/slm-{size}-chat](https://huggingface.co/{hf_username}/slm-{size}-chat) | Instruct + general DPO |
| Code | [{hf_username}/slm-{size}-code](https://huggingface.co/{hf_username}/slm-{size}-code) | Instruct + code SFT |

## Architecture

| Component | Choice | Rationale |
|---|---|---|
| Positional encoding | RoPE | Better length generalisation, relative position awareness |
| Normalization | RMSNorm | Faster than LayerNorm, modern standard |
| Activation | SwiGLU | Better gradient flow, used by LLaMA and Mistral |
| Attention | GQA | Reduces KV cache memory at inference |
| Bias | None | Simpler, modern standard |
| Embeddings | Tied | Reduces parameters, effective at small scale |
| Layers | {config.num_hidden_layers} | Decoder blocks |
| Hidden size | {config.hidden_size} | Model width |
| Attention heads | {config.num_attention_heads} query / {config.num_key_value_heads} KV | Grouped-query layout |
| Context | {config.max_position_embeddings:,} tokens | Native trained context |
| RoPE theta | {config.rope_theta:g} | Rotary frequency base |
| Vocab size | {config.vocab_size:,} | Custom BPE tokenizer trained on the pretraining corpus |
| Parameters | {param_str} | |

The model was trained with the repository's SLM implementation and exported
to the equivalent native Transformers `LlamaForCausalLM` format. The Hub
package contains no executable model code and does not require
`trust_remote_code`.

## Training

{training_section}

## Usage

{usage_section}

## Evaluation

Benchmark scores are not embedded in this release. The export gate validates
checkpoint integrity, native Transformers loading, source/export logit and
greedy-generation parity, cached/uncached generation parity, tokenizer
compatibility, and the absence of remote model code. These are packaging and
behavioral-integrity checks, not measures of downstream task quality.

Use the repository's benchmark and sanity evaluation tools before selecting
the model for an application.

## Limitations

- **Scale:** At {size_upper} parameters this model is significantly smaller than frontier models. It will underperform on complex reasoning, long-context tasks, and domains not well-represented in the pretraining data.
- **Hallucination:** Like all language models, this model can generate plausible-sounding but factually incorrect content. Outputs should not be used as a source of truth without independent verification.
- **Safety:** DPO alignment provides basic harmlessness training but does not guarantee safe outputs in all contexts. This model has not undergone red-teaming or adversarial safety evaluation.
- **Languages:** Training data is predominantly English. Performance on other languages will be significantly degraded.
- **Code:** Code generation is primarily Python-oriented, reflecting the code sub-mix distribution used in pretraining and SFT.

## License

Released under the MIT License.

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



def _export_tokenizer_to_checkpoint_root(tokenizer, tokenizer_path: Path, checkpoint: Path) -> None:
    """
    Save/copy tokenizer artifacts to the checkpoint root so standard Hub loading works:

        AutoTokenizer.from_pretrained(repo_id)

    The tokenizer may also exist under checkpoint/tokenizer/, but the Hub root
    must contain tokenizer.json/tokenizer_config.json/etc. for normal use.
    """
    import shutil

    tokenizer.save_pretrained(str(checkpoint))

    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "slm_tokenizer.json",
    ]:
        src = tokenizer_path / filename
        dst = checkpoint / filename
        if src.exists():
            shutil.copy2(src, dst)
            log.info(f"Copied tokenizer artifact to checkpoint root: {filename}")

    special_tokens_map_path = checkpoint / "special_tokens_map.json"
    if not special_tokens_map_path.exists():
        special_tokens_map = {}
        for key, value in getattr(tokenizer, "special_tokens_map", {}).items():
            if isinstance(value, list):
                special_tokens_map[key] = [str(item) for item in value]
            else:
                special_tokens_map[key] = str(value)

        with special_tokens_map_path.open("w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, indent=2)
            f.write("\n")
        log.info("Created special_tokens_map.json from tokenizer.special_tokens_map")

    required = [
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Tokenizer export incomplete at checkpoint root. Missing: {missing}"
        )

    log.info("Tokenizer artifacts exported to checkpoint root")

def _checkpoint_dtype(checkpoint: Path):
    """Read the stored tensor dtype without materializing the checkpoint."""
    import safetensors
    import torch

    weights_path = checkpoint / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Native export requires an unsharded safetensors checkpoint at "
            f"{weights_path}"
        )

    with safetensors.safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys:
            raise RuntimeError(f"Checkpoint contains no tensors: {weights_path}")
        dtype = handle.get_tensor(keys[0]).dtype

    supported = {torch.float32, torch.float16, torch.bfloat16}
    if dtype not in supported:
        raise ValueError(f"Unsupported checkpoint dtype for export: {dtype}")
    return dtype


def _native_llama_config(source_config, tokenizer, source_dtype):
    """Translate SLMConfig fields to the equivalent native LlamaConfig."""
    from transformers import LlamaConfig

    if len(tokenizer) != source_config.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer={len(tokenizer):,}, "
            f"model={source_config.vocab_size:,}"
        )

    return LlamaConfig(
        vocab_size=source_config.vocab_size,
        hidden_size=source_config.hidden_size,
        intermediate_size=source_config.intermediate_size,
        num_hidden_layers=source_config.num_hidden_layers,
        num_attention_heads=source_config.num_attention_heads,
        num_key_value_heads=source_config.num_key_value_heads,
        hidden_act="silu",
        max_position_embeddings=source_config.max_position_embeddings,
        initializer_range=source_config.initializer_range,
        rms_norm_eps=source_config.rms_norm_eps,
        use_cache=source_config.use_cache,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=source_config.tie_word_embeddings,
        rope_theta=source_config.rope_theta,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=source_config.attention_dropout,
        mlp_bias=False,
        dtype=str(source_dtype).removeprefix("torch."),
    )


def _convert_to_native_llama(source_model, tokenizer, source_dtype):
    """Create a native LlamaForCausalLM and load the SLM state dict strictly."""
    from transformers import LlamaForCausalLM

    native_config = _native_llama_config(
        source_model.config,
        tokenizer,
        source_dtype,
    )
    native_model = LlamaForCausalLM(native_config).to(dtype=source_dtype)
    load_result = native_model.load_state_dict(source_model.state_dict(), strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "SLM-to-Llama state-dict conversion was not exact: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    native_model.tie_weights()
    native_model.eval()
    return native_model


def _parity_batch(tokenizer):
    """Build one deterministic chat-formatted batch for round-trip checks."""
    import torch

    messages = [
        {"role": "system", "content": "Answer clearly and concisely."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        input_ids = encoded
        attention_mask = None
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _capture_parity_reference(model, tokenizer) -> dict:
    """Capture source logits and greedy tokens before native conversion."""
    import torch
    from inference.utils import resolve_special_token_ids

    model.eval()
    input_ids, attention_mask = _parity_batch(tokenizer)
    special_ids = resolve_special_token_ids(tokenizer)
    generation_kwargs = {
        "max_new_tokens": 16,
        "do_sample": False,
        "use_cache": False,
        "pad_token_id": special_ids.pad,
        "eos_token_id": special_ids.eos_list,
    }
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.detach().float().cpu()
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        ).detach().cpu()

    return {
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
        "logits": logits,
        "generated": generated,
        "generation_kwargs": generation_kwargs,
    }


def _validate_round_trip_parity(model, reference: dict, source_dtype) -> None:
    """Reject an exported package whose logits or greedy output changed."""
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=reference["input_ids"],
            attention_mask=reference["attention_mask"],
            use_cache=False,
        ).logits.detach().float().cpu()
        generated = model.generate(
            input_ids=reference["input_ids"],
            attention_mask=reference["attention_mask"],
            **reference["generation_kwargs"],
        ).detach().cpu()

    if source_dtype == torch.float32:
        rtol, atol = 1e-5, 1e-5
    elif source_dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 2e-2, 2e-2

    try:
        torch.testing.assert_close(
            logits,
            reference["logits"],
            rtol=rtol,
            atol=atol,
        )
    except AssertionError as exc:
        max_abs = (logits - reference["logits"]).abs().max().item()
        raise RuntimeError(
            "Native export changed model logits "
            f"(max_abs_diff={max_abs:.6g}, rtol={rtol}, atol={atol})"
        ) from exc

    if not torch.equal(generated, reference["generated"]):
        raise RuntimeError(
            "Native export changed deterministic greedy generation. "
            "Refusing to publish a behaviorally different checkpoint."
        )
    log.info("Native round-trip parity passed (logits and greedy tokens)")


def _write_generation_config(export_dir: Path, tokenizer) -> None:
    """Write explicit generation stop-token metadata."""
    from transformers import GenerationConfig
    from inference.utils import resolve_special_token_ids

    special_ids = resolve_special_token_ids(tokenizer)
    generation_config = GenerationConfig(
        bos_token_id=special_ids.bos,
        eos_token_id=special_ids.eos_list,
        pad_token_id=special_ids.pad,
        do_sample=False,
        use_cache=True,
    )
    generation_config.save_pretrained(str(export_dir))


def _validate_native_package(
    export_dir: Path,
    source_config,
    tokenizer,
    source_dtype,
):
    """Load the staged package through Auto* with remote code disabled."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config_path = export_dir / "config.json"
    config_json = _read_json(config_path, "native export config")
    if config_json.get("model_type") != "llama":
        raise RuntimeError("Export config model_type must be 'llama'")
    if config_json.get("architectures") != ["LlamaForCausalLM"]:
        raise RuntimeError(
            "Export config architectures must be ['LlamaForCausalLM']"
        )
    if "auto_map" in config_json:
        raise RuntimeError("Native export must not contain auto_map")

    python_files = sorted(path.name for path in export_dir.glob("*.py"))
    if python_files:
        raise RuntimeError(
            f"Native export must not bundle executable model code: {python_files}"
        )

    loaded_config = AutoConfig.from_pretrained(
        str(export_dir),
        trust_remote_code=False,
        local_files_only=True,
    )
    expected_fields = {
        "vocab_size": source_config.vocab_size,
        "hidden_size": source_config.hidden_size,
        "intermediate_size": source_config.intermediate_size,
        "num_hidden_layers": source_config.num_hidden_layers,
        "num_attention_heads": source_config.num_attention_heads,
        "num_key_value_heads": source_config.num_key_value_heads,
        "max_position_embeddings": source_config.max_position_embeddings,
    }
    for field, expected in expected_fields.items():
        actual = getattr(loaded_config, field)
        if actual != expected:
            raise RuntimeError(
                f"Native config mismatch for {field}: "
                f"expected {expected}, got {actual}"
            )

    loaded_tokenizer = AutoTokenizer.from_pretrained(
        str(export_dir),
        trust_remote_code=False,
        local_files_only=True,
    )
    if len(loaded_tokenizer) != len(tokenizer):
        raise RuntimeError(
            "Tokenizer vocabulary changed during export: "
            f"{len(tokenizer)} -> {len(loaded_tokenizer)}"
        )

    loaded_model = AutoModelForCausalLM.from_pretrained(
        str(export_dir),
        trust_remote_code=False,
        local_files_only=True,
    )
    if loaded_model.config.model_type != "llama":
        raise RuntimeError("Clean AutoModel load did not resolve native Llama")
    loaded_dtype = next(loaded_model.parameters()).dtype
    if loaded_dtype != source_dtype:
        raise RuntimeError(
            f"Native load changed checkpoint dtype: "
            f"expected {source_dtype}, got {loaded_dtype}"
        )
    log.info("Clean AutoConfig/AutoTokenizer/AutoModel load passed")
    return loaded_model


def _remote_code_artifacts(repo_files: list[str]) -> list[str]:
    """Return executable Python artifacts that can trigger remote-code loading."""
    return sorted(
        path
        for path in repo_files
        if path.endswith(".py") or path == "slm_remote" or path.startswith("slm_remote/")
    )


def _validate_published_native_package(
    api,
    repo_id: str,
    revision: str,
    expected_tokenizer_size: int,
) -> None:
    """Verify the exact published commit has a code-free native HF contract."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    repo_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
    )
    remote_code = _remote_code_artifacts(repo_files)
    if remote_code:
        raise RuntimeError(
            "Published repository still contains executable Python model code: "
            f"{remote_code}"
        )
    if not any(path.endswith(".safetensors") for path in repo_files):
        raise RuntimeError("Published repository contains no safetensors weights")

    config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            token=HF_TOKEN,
        )
    )
    config_json = _read_json(config_path, "published config")
    if "auto_map" in config_json:
        raise RuntimeError("Published config still contains auto_map")
    if config_json.get("model_type") != "llama":
        raise RuntimeError("Published config model_type must be 'llama'")
    if config_json.get("architectures") != ["LlamaForCausalLM"]:
        raise RuntimeError(
            "Published config architectures must be ['LlamaForCausalLM']"
        )

    published_config = AutoConfig.from_pretrained(
        repo_id,
        revision=revision,
        token=HF_TOKEN,
        trust_remote_code=False,
    )
    if published_config.model_type != "llama":
        raise RuntimeError("Published AutoConfig did not resolve native Llama")

    published_tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        revision=revision,
        token=HF_TOKEN,
        trust_remote_code=False,
    )
    if len(published_tokenizer) != expected_tokenizer_size:
        raise RuntimeError(
            "Published tokenizer vocabulary differs from the validated export: "
            f"{expected_tokenizer_size} -> {len(published_tokenizer)}"
        )

    log.info(
        "Published commit %s passed the native, no-remote-code contract",
        revision,
    )


def _write_export_manifest(
    export_dir: Path,
    source_checkpoint: Path,
    size: str,
    variant: str,
    source_config,
    n_params: int,
    source_dtype,
) -> None:
    """Record the conversion contract without hashing multi-GB weights."""
    manifest = {
        "schema_version": 1,
        "format": "transformers_native_llama",
        "source": {
            "size": size,
            "variant": variant,
            "stage": source_checkpoint.parent.name,
            "checkpoint": source_checkpoint.name,
        },
        "source_model_type": source_config.model_type,
        "export_model_type": "llama",
        "source_dtype": str(source_dtype).removeprefix("torch."),
        "parameters": n_params,
        "architecture": {
            "vocab_size": source_config.vocab_size,
            "hidden_size": source_config.hidden_size,
            "intermediate_size": source_config.intermediate_size,
            "num_hidden_layers": source_config.num_hidden_layers,
            "num_attention_heads": source_config.num_attention_heads,
            "num_key_value_heads": source_config.num_key_value_heads,
            "max_position_embeddings": source_config.max_position_embeddings,
            "rope_theta": source_config.rope_theta,
            "tie_word_embeddings": source_config.tie_word_embeddings,
        },
    }
    (export_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def export(
    size: str,
    variant: str,
    model_path: Path | None = None,
    dry_run: bool = False,
    private: bool = False,
) -> None:
    import gc
    import shutil

    from model import SLMConfig, SLMForCausalLM

    if not dry_run and not HF_USERNAME:
        log.error(
            "HF_USERNAME not set in the environment. "
            "Add HF_USERNAME=<your-hub-username> to .env before running export."
        )
        sys.exit(1)

    hf_username = HF_USERNAME or "local"
    variant_cfg = VARIANTS[variant]
    checkpoint  = model_path or variant_cfg["checkpoint"](size)
    hub_suffix  = variant_cfg["hub_suffix"]
    hub_name    = f"slm-{size}{hub_suffix}"
    repo_id     = f"{hf_username}/{hub_name}"
    export_parent = export_size_dir(size)
    export_dir = export_parent / variant
    staging_dir = export_parent / f".{variant}.staging"

    log.info(f"=== SLM Export ===")
    log.info(f"Size:       {size}")
    log.info(f"Variant:    {variant}")
    log.info(f"Checkpoint: {checkpoint}")
    log.info(f"Artifact:   {export_dir}")
    log.info(f"Hub:        {repo_id}")
    log.info(f"Dry run:    {dry_run}")

    if not checkpoint.exists():
        log.error(f"Checkpoint not found: {checkpoint}")
        log.error(
            f"Run the training pipeline first. For chat variant: "
            f"make pretrain sft-instruct sft-code dpo-chat SIZE={size}"
        )
        sys.exit(1)

    source_dtype = _checkpoint_dtype(checkpoint)
    log.info(f"Checkpoint dtype: {source_dtype}")
    log.info("Loading source SLM checkpoint...")
    config = SLMConfig.from_pretrained(str(checkpoint))
    model = SLMForCausalLM.from_pretrained(
        str(checkpoint),
        dtype=source_dtype,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    tokenizer_path = checkpoint / "tokenizer"
    if not (tokenizer_path / "tokenizer_config.json").exists():
        tokenizer_path = tokenizer_dir(size)
    tokenizer = load_tokenizer(tokenizer_path)
    log.info(f"Tokenizer loaded from {tokenizer_path}")

    _validate_model(model, tokenizer, config)
    parity_reference = _capture_parity_reference(model, tokenizer)

    export_parent.mkdir(parents=True, exist_ok=True)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    log.info("Converting SLM checkpoint to native LlamaForCausalLM...")
    native_model = _convert_to_native_llama(model, tokenizer, source_dtype)
    native_model.save_pretrained(
        str(staging_dir),
        safe_serialization=True,
    )
    _export_tokenizer_to_checkpoint_root(tokenizer, tokenizer_path, staging_dir)
    _write_generation_config(staging_dir, tokenizer)
    _write_export_manifest(
        staging_dir,
        checkpoint,
        size,
        variant,
        config,
        n_params,
        source_dtype,
    )

    model_card = generate_model_card(
        size=size,
        variant=variant,
        hub_name=hub_name,
        n_params=n_params,
        checkpoint=checkpoint,
        config=config,
        hf_username=hf_username,
    )
    card_path = staging_dir / "README.md"
    card_path.write_text(model_card, encoding="utf-8")
    log.info(f"Model card written to {card_path} ({len(model_card):,} chars)")

    # Release both in-memory conversion models before the required clean load.
    del native_model
    del model
    gc.collect()

    clean_model = _validate_native_package(
        staging_dir,
        config,
        tokenizer,
        source_dtype,
    )
    _validate_round_trip_parity(clean_model, parity_reference, source_dtype)
    _validate_model(clean_model, tokenizer, clean_model.config)
    del clean_model
    gc.collect()

    if export_dir.exists():
        shutil.rmtree(export_dir)
    staging_dir.replace(export_dir)
    log.info(f"Native export artifact ready: {export_dir}")

    if dry_run:
        log.info("Dry run — native artifact validated; skipping Hub push")
        log.info(f"Would push to: https://huggingface.co/{repo_id}")
        return

    if not HF_TOKEN:
        log.error("HF_TOKEN not set in .env")
        sys.exit(1)
    from huggingface_hub import HfApi, login
    login(token=HF_TOKEN)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    log.info(f"Pushing {export_dir} to {repo_id} (single commit)...")
    commit_info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(export_dir),
        commit_message=f"Export {hub_name} ({n_params / 1e6:.1f}M params)",
        # Remove files left by the previous auto_map/remote-code packaging
        # contract in the same commit that uploads the native package.
        delete_patterns=OBSOLETE_REMOTE_CODE_PATTERNS,
        ignore_patterns=[
            "__pycache__",
            "*.pyc",
        ],
    )

    revision = getattr(commit_info, "oid", None)
    if not revision:
        raise RuntimeError(
            "Hub upload did not return a commit revision; cannot verify the "
            "published artifact"
        )
    _validate_published_native_package(
        api=api,
        repo_id=repo_id,
        revision=revision,
        expected_tokenizer_size=len(tokenizer),
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

    generation_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": 32,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "pad_token_id": special_ids.pad,
        "eos_token_id": eos_ids,
    }

    with torch.no_grad():
        output = model.generate(
            input_ids,
            use_cache=True,
            **generation_kwargs,
        )
        uncached_output = model.generate(
            input_ids,
            use_cache=False,
            **generation_kwargs,
        )

    if not torch.equal(output, uncached_output):
        cached_new = output[0][input_length:].tolist()
        uncached_new = uncached_output[0][input_length:].tolist()
        mismatch = next(
            (
                index
                for index, (cached_id, uncached_id) in enumerate(
                    zip(cached_new, uncached_new)
                )
                if cached_id != uncached_id
            ),
            min(len(cached_new), len(uncached_new)),
        )
        raise RuntimeError(
            "Validation failed: cached and uncached greedy generation diverged "
            f"at generated token {mismatch}. Refusing to export a checkpoint "
            "with unreliable KV-cache generation."
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
  python export/export.py --size 125m --variant code
  python export/export.py --size 125m --variant chat --dry-run
        """,
    )
    parser.add_argument("--size",    type=str,  required=True, choices=["125m", "350m", "1b"])
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=list(VARIANTS.keys()),
        help="base | instruct | chat | code",
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
