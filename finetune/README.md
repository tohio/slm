# Supervised Fine-Tuning

This directory prepares external instruction datasets and trains the instruct,
code-instruction, and optional raw code-completion branches.

Synthetic example generation is maintained in
[`tohio/slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data);
this repository consumes datasets but does not generate them.

## Contents

| Path | Purpose |
|---|---|
| `configs/sft_data_sources.yaml` | Pinned dataset, adapter, limits, and validation policy |
| `configs/sft_instruct_*.yaml` | Instruct SFT recipes |
| `configs/sft_code_*.yaml` | Code-instruction SFT recipes |
| `configs/code_completion_*.yaml` | Optional raw code-completion recipes |
| `data/prepare_sft.py` | Normalize, validate, deduplicate, split, and manifest SFT data |
| `data/prepare_code_completion.py` | Derive leakage-checked completion pairs |
| `train_sft.py` | Assistant-only-loss instruct/code trainer |
| `train_code_completion.py` | Prompt-masked raw completion trainer |

## Model branches

```text
pretrain/final
  └─ sft_instruct/final
       ├─ dpo_chat/final
       └─ sft_code/final
            └─ sft_code_completion/final  (optional)
```

Instruct SFT starts from the reinitialized base checkpoint. Code SFT starts
from instruct SFT so it retains the general assistant behavior learned by the
first stage.

## Dataset contract

`configs/sft_data_sources.yaml` is the source registry used by preparation and
training:

| Stage | Dataset | Adapter | Full-profile cap |
|---|---|---|---:|
| Instruct | `HuggingFaceH4/ultrachat_200k` | conversational messages | 50,000 |
| Code | `ise-uiuc/Magicoder-OSS-Instruct-75K` | problem/solution instruction | 75,000 |

Each source is pinned to an immutable revision. The 125M, 350M, and 1B
profiles use the same selected population; `mini` uses 1,000 records per
stage for pipeline validation.

Preparation writes:

```text
$DATA_DIR/runs/<size>/sft_instruct/
  train.jsonl
  val.jsonl
  manifest.json

$DATA_DIR/runs/<size>/sft_code/
  train.jsonl
  val.jsonl
  manifest.json
```

The preparation stage normalizes records to the project's conversation
schema, validates required roles/content, rejects excessive invalid or
duplicate rates, and splits by normalized user prompt. The manifest records
the source revision, selection settings, tokenizer hash, file hashes,
retention statistics, and zero prompt overlap between train and validation.
Existing output is reused only when its manifest matches the current contract.

## Prepare data

Prepare both SFT stages:

```bash
make prepare-sft SIZE=125m
```

Prepare or replace data explicitly after changing the source contract:

```bash
python finetune/data/prepare_sft.py \
  --size 125m \
  --stage both \
  --force
```

The target tokenizer must already be available at
`$DATA_DIR/runs/<size>/tokenizer/`; preparation uses its exact chat rendering
and token budgets.

## Train

Generate hardware-specific recipes on the training host:

```bash
make config-gen-sft SIZE=125m GPUS=1
```

Train instruct and code branches:

```bash
make sft-instruct SIZE=125m GPUS=1
make sft-code SIZE=125m GPUS=1
```

Resume the latest compatible checkpoints:

```bash
make sft-instruct-resume SIZE=125m GPUS=1
make sft-code-resume SIZE=125m GPUS=1
```

Run preprocessing and compatibility checks without optimization:

```bash
python finetune/train_sft.py \
  --config finetune/configs/sft_instruct_125m.yaml \
  --preflight-only
```

The trainer requires the tokenizer chat template and its generation markers.
It applies assistant-only loss, reports retained/supervised tokens, enforces
the configured minimum retention ratio, validates context and vocabulary
compatibility, and keeps packing disabled so examples cannot attend across
packed boundaries.

Before training starts, it writes `sft_run_audit.json`. The promoted `final/`
directory contains the lowest-validation-loss checkpoint, the run audit, and
the prepared-data manifest:

```text
$RESULTS_DIR/runs/<size>/sft_instruct/final/
$RESULTS_DIR/runs/<size>/sft_code/final/
```

## Optional raw code completion

Derive completion examples from the already split code-instruction dataset:

```bash
make prepare-code-completion SIZE=125m
make sft-code-completion SIZE=125m
```

The preparation stage checks prompt and pair leakage. Training masks prompt
tokens, uses token-weighted validation loss, writes resumable checkpoints, and
promotes the best validation checkpoint to:

```text
$RESULTS_DIR/runs/<size>/sft_code_completion/final/
```

Resume directly:

```bash
python finetune/train_code_completion.py \
  --config finetune/configs/code_completion_125m.yaml \
  --resume
```

Evaluate the branch with HumanEval:

```bash
make eval-code-completion SIZE=125m
```

HumanEval executes generated code; run it only in an isolated environment.

## Validation

Run contract and one-step integration tests before a full job:

```bash
python -m pytest \
  tests/test_sft_data_contract.py \
  tests/test_training_args.py \
  tests/test_trl_smoke.py \
  -q
```

Validate completed stage artifacts:

```bash
make test-sft-instruct SIZE=125m
make test-sft-code SIZE=125m
```
