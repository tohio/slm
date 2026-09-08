# Export

This directory converts completed SLM checkpoints into standard Transformers
Llama packages for local use or publication on the Hugging Face Hub.

## Variants

| Variant | Source checkpoint | Example Hub repository (`HF_USERNAME=tohio`) |
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

Override the mapped source checkpoint with `--model PATH`; its recorded size and
stage must still match `--size` and `--variant`. Hub publication
reads `HF_USERNAME` and `HF_TOKEN` from the environment or `.env`; local
`--dry-run` export does not require Hub credentials.

Run the complete native conversion and clean-load acceptance contract for one
real checkpoint:

```bash
make test-export-acceptance \
  SIZE=125m \
  EXPORT_VARIANT=base
```

In a supported vLLM serving environment, load that package and generate one
bounded response:

```bash
make test-vllm-export \
  SIZE=125m \
  EXPORT_VARIANT=base
```

## Conversion contract

Only 125M/350M/1B production profiles are exportable. Before loading model weights,
export checks the checkpoint config against the declared profile and its
checksum-verified pretraining/SFT/DPO run audit. The audit must identify the
requested training stage and recipe; relabeling a Mini/base checkpoint through
`--size`/`--variant` is rejected. Missing or legacy unverifiable audits require
restoring genuine provenance, not manufacturing metadata.

The tokenizer must be bundled with this checkpoint, either in `tokenizer/` or
at its root. There is no fallback to a tokenizer selected merely by model size.
Never retrain a replacement tokenizer for already learned embeddings.

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
training_provenance.json
```

Large models may use multiple safetensor shards and an index file.
`export_manifest.json` records the source stage, variant, architecture,
parameter count, source dtype, package format, and verified run-contract hash.
Its stage comes from the audit, not a guess from the source directory name.

## Validation

Every local and Hub export, including `test-export-acceptance`, must pass:

1. source-checkpoint generation hygiene;
2. strict state-dict conversion;
3. model configuration and tokenizer compatibility;
4. standard `AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM` loading;
5. source/export logit parity within the dtype tolerance;
6. exact deterministic greedy-generation parity;
7. cached/uncached generation parity; and
8. package-content validation.

The destination is replaced only after the staged artifact passes all checks.

## Portable provenance

Export validates its card inputs before allocating/converting the model. New
final checkpoints carry `training_provenance.json`, containing stage audits,
prepared-data manifests and checkpoint fingerprints through the pretraining →
instruct → code/chat lineage. No ancestor weights or training corpus are copied
into the bundle. Its checksum, audits, manifests and parent links must match;
a final can be moved/restored without the original parent directories.

Pretraining provenance comes from the checkpoint's audited `dataset_size` and
source run, not the model's export size and not a current `blend_stats.json`.
Cards distinguish the full train/val/test corpus, selected unique training tokens,
observed consumed input tokens, and planned schedule. Source shares describe the
complete source train corpus, not necessarily the selected prefix. Missing legacy
measurements are labelled **not recorded**, not replaced with planning defaults.

Legacy finals without an ancestry bundle may supply authentic relocated parents:

```bash
.venv/bin/python export/export.py --size 125m --variant chat --dry-run \
  --model /restore/chat/final \
  --provenance-parent /restore/instruct/final \
  --provenance-parent /restore/pretrain/final
```

Parents are matched against recorded config/weight/tokenizer identities, not
filenames. The original audited path is used only as a verified legacy fallback;
an unrelated checkpoint at that path is never trusted. A missing or mismatched
ancestor fails before conversion. Source finals are not modified to manufacture
missing history. The native package includes the verified **source** provenance
and its digest in `export_manifest.json`, separate from converted model weights.

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
