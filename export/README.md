# Export

## Purpose

Native Hugging Face export for SLM base, instruct, chat, and code variants.

Training checkpoints use the repository's `SLMConfig` and
`SLMForCausalLM`. Export converts the equivalent configuration and state dict
to Transformers' built-in `LlamaForCausalLM` package. Published models contain
no executable architecture code and load without `trust_remote_code`.

## How It Fits In

Export is the publication boundary between repository-native training
checkpoints and portable Transformers packages; see
[Architecture](../docs/ARCHITECTURE.md).

## Variants

| Variant | Source checkpoint | Hub repository |
|---|---|---|
| Base | `results/runs/<size>/pretrain/final` | `tohio/slm-<size>` |
| Instruct | `results/runs/<size>/sft_instruct/final` | `tohio/slm-<size>-instruct` |
| Chat | `results/runs/<size>/dpo_chat/final` | `tohio/slm-<size>-chat` |
| Code | `results/runs/<size>/sft_code/final` | `tohio/slm-<size>-code` |

Native artifacts are written separately from training checkpoints:

```text
results/exports/<size>/<variant>/
```

Set `EXPORTS_DIR` to override the export root. Source checkpoints are never
modified.

## Commands

Validate and build a local native artifact without pushing:

```bash
make export-base-local SIZE=125m
make export-instruct-local SIZE=125m
make export-chat-local SIZE=125m
make export-code-local SIZE=125m
make export-local SIZE=125m
```

Build, validate, and push:

```bash
make export-base SIZE=125m
make export-instruct SIZE=125m
make export-chat SIZE=125m
make export-code SIZE=125m
make export SIZE=125m
```

Direct calls:

```bash
python export/export.py --size 125m --variant chat --dry-run
python export/export.py --size 125m --variant chat
python export/export.py --size 125m --variant chat --private
```

Hub pushes require:

```bash
HF_USERNAME=tohio
HF_TOKEN=...
```

`.env` is loaded automatically. A local `--dry-run` does not require Hub
credentials.

## Native conversion contract

The SLM and Llama packages use the same decoder structure and parameter names:

- token embedding and tied LM head
- pre-norm decoder blocks
- bias-free grouped-query attention
- RoPE
- RMSNorm
- bias-free SwiGLU MLP

Export maps the SLM configuration to `LlamaConfig`, loads every state-dict key
strictly, and saves with safe serialization. The exported `config.json` must
contain:

```json
{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  "tie_word_embeddings": true
}
```

It must not contain `auto_map`, and the artifact root must not contain model
architecture Python files.

When pushing over an older SLM Hub repository, export deletes the obsolete
root architecture files and `slm_remote/` directory in the same Hub commit.
Uploading a clean local folder alone would not remove stale remote files.

## Artifact contents

Each native artifact contains:

```text
README.md
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
export_manifest.json
```

Tokenizer files are copied to the artifact root. `generation_config.json`
records PAD, BOS, EOS, and end-of-turn stop IDs from the tokenizer rather than
hardcoded numeric values.

`export_manifest.json` records the logical source stage/variant, source dtype,
parameter count, native format, and architecture values. It does not expose
host filesystem paths and intentionally does not hash multi-gigabyte model
weights during export.

## Validation

Every local or Hub export must pass:

1. Source checkpoint generation hygiene.
2. Strict SLM-to-Llama state-dict loading.
3. Configuration and tokenizer contract checks.
4. Clean `AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM` loading with
   `trust_remote_code=False`.
5. Source/export logit parity within dtype-specific tolerance.
6. Exact deterministic greedy-generation parity.
7. Cached versus uncached generation parity on the exported model.
8. Rejection of `auto_map` or bundled Python model files.
9. Exact-revision Hub checks for native config resolution, tokenizer size,
   safetensors weights, and absence of executable Python files.

The existing native artifact is replaced only after the staged artifact passes
all local checks. A Hub upload reports success only after the published commit
passes the remote contract checks.

## Model-card metadata

Architecture values and parameter counts come from the loaded checkpoint.
Non-base variants require the SFT/DPO data manifests copied into their final
checkpoints. Dataset names, immutable revisions, and prepared record counts are
rendered from those manifests; export fails instead of publishing guessed or
stale training provenance.

Realized pretraining mix metadata is read from:

```text
data/runs/<size>/metadata/blend_stats.json
```

Legacy fallback:

```text
data/runs/<size>/curated/blend_stats.json
```

If neither exists, the model card labels the pretraining table as the design
mix rather than realized data.

Generated cards include the model identity, variant lineage, architecture,
context length, training provenance, intended use, explicit evaluation status,
limitations, native loading example, and MIT license. Update card content in
`generate_model_card()`; do not hand-edit an exported `README.md`.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tohio/slm-125m-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

The same code loads a local native artifact:

```python
model_id = "results/exports/125m/chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
```
