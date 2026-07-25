# Export

This directory converts completed SLM checkpoints into standard Transformers
Llama packages for local use or publication on the Hugging Face Hub.

## Variants

| Variant | Source checkpoint | Default Hub repository |
|---|---|---|
| Base | `$RESULTS_DIR/runs/<size>/pretrain/final` | `tohio/slm-<size>` |
| Instruct | `$RESULTS_DIR/runs/<size>/sft_instruct/final` | `tohio/slm-<size>-instruct` |
| Chat | `$RESULTS_DIR/runs/<size>/dpo_chat/final` | `tohio/slm-<size>-chat` |
| Code | `$RESULTS_DIR/runs/<size>/sft_code/final` | `tohio/slm-<size>-code` |

Exported artifacts are independent of training checkpoints:

```text
$EXPORTS_DIR/<size>/<variant>/
```

`EXPORTS_DIR` defaults to `$RESULTS_DIR/exports`. Export never edits the
source checkpoint.

## Usage

Build and validate one local artifact without a Hub push:

```bash
make export-chat-local SIZE=125m
```

Build all four local variants:

```bash
make export-local SIZE=125m
```

Build, validate, and publish:

```bash
make export-base SIZE=125m
make export-instruct SIZE=125m
make export-chat SIZE=125m
make export-code SIZE=125m
```

Equivalent direct commands:

```bash
python export/export.py \
  --size 125m \
  --variant chat \
  --dry-run

python export/export.py \
  --size 125m \
  --variant chat \
  --private
```

Override the mapped source checkpoint with `--model PATH`. Hub publication
reads `HF_USERNAME` and `HF_TOKEN` from the environment or `.env`; local
`--dry-run` export does not require Hub credentials.

## Conversion contract

The project model and Transformers Llama model share the same decoder
structure: tied token embeddings, pre-normalized decoder blocks, grouped-query
attention, RoPE, RMSNorm, and a bias-free SwiGLU MLP.

Export maps `SLMConfig` to `LlamaConfig`, loads all converted state-dict keys
strictly, and saves safetensors. The resulting package identifies:

```json
{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  "tie_word_embeddings": true
}
```

Tokenizer files and the chat template are copied into the artifact root.
Generation configuration resolves PAD, BOS, EOS, and end-of-turn IDs from that
tokenizer.

## Artifact contents

A completed export includes:

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

Large models may use multiple safetensor shards and an index file.
`export_manifest.json` records the source stage, variant, architecture,
parameter count, source dtype, and package format.

## Validation

Every local and Hub export must pass:

1. source-checkpoint generation hygiene;
2. strict state-dict conversion;
3. model configuration and tokenizer compatibility;
4. standard `AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM` loading;
5. source/export logit parity within the dtype tolerance;
6. exact deterministic greedy-generation parity;
7. cached/uncached generation parity; and
8. package-content validation.

The destination is replaced only after the staged artifact passes all checks.

For non-base variants, export also requires the SFT or DPO data manifest copied
into the final checkpoint. Model-card dataset names, revisions, and record
counts come from those manifests. Pretraining provenance is read from:

```text
$DATA_DIR/runs/<size>/metadata/blend_stats.json
```

If realized blend metadata is unavailable, the model card labels the mix as
the configured design rather than presenting it as observed data.

## Load an export

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tohio/slm-125m-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

The same APIs load a local artifact:

```python
model_dir = "results/exports/125m/chat"
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    local_files_only=True,
)
```
