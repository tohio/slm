# Supervised fine-tuning

This directory consumes external SFT datasets and trains the instruct, code,
and optional raw code-completion branches. Synthetic data is generated and
repaired in the separate `slm-synthetic-data` project; no synthetic examples
are created here.

## Data contract

`configs/sft_data_sources.yaml` is the only source registry. It pins the Hub
dataset, immutable revision, split, source format, validation thresholds, and
per-size row cap. The current temporary sources are:

| Stage | Source | Purpose |
|---|---|---|
| Instruct | `HuggingFaceH4/ultrachat_200k` | General assistant behavior |
| Code | `ise-uiuc/Magicoder-OSS-Instruct-75K` | Code instruction following |

The 125M, 350M, and 1B recipes use the same source population and cap so model
comparisons are not confounded by different SFT data. `mini` has a smaller cap
for pipeline validation.

To replace a source later, update only `configs/sft_data_sources.yaml`. A
conversational replacement must expose a `messages` or `conversations` column;
the `magicoder` adapter accepts problem/solution-style records.

Preparation:

```bash
make prepare-sft SIZE=125m
# Replace previously prepared data only after intentionally changing contract:
python finetune/data/prepare_sft.py --size 125m --stage both --force
```

Each prepared directory contains `train.jsonl`, `val.jsonl`, and
`manifest.json`. Preparation fails when invalid or duplicate rates exceed the
configured thresholds. Exact duplicates are counted and recorded, not silently
used. Validation is grouped by normalized user prompt, and the manifest must
report zero train/validation prompt overlap. An existing dataset is reused only
when its manifest matches the current source contract.

Prepared paths:

```text
data/runs/<size>/sft_instruct/
data/runs/<size>/sft_code/
data/runs/<size>/code_completion/
```

## Training branches

```text
pretrain/final
  -> sft_instruct/final
       -> DPO chat alignment (alignment/)
       -> sft_code/final
            -> optional sft_code_completion/final
```

Run instruct and code SFT:

```bash
make sft-instruct SIZE=125m GPUS=1
make sft-instruct-resume SIZE=125m GPUS=1
make sft-code SIZE=125m GPUS=1
make sft-code-resume SIZE=125m GPUS=1
```

The trainer requires the tokenizer's native chat template with
`{% generation %}` markers and uses assistant-only loss. Before optimization
it validates vocabulary/special-token/context compatibility, tokenizes both
splits, records supervised-token and retention statistics, and aborts below
`data.min_retention_ratio`. `data.loss_type: chunked_nll` is explicit in every
recipe. Packing remains disabled because the custom attention implementation
does not yet enforce packed-example boundaries.

`sft_run_audit.json` is written before training. `final/` contains the
lowest-validation-loss checkpoint plus the run audit and data manifest.
Run the exact preprocessing checks without spending a training run:

```bash
python finetune/train_sft.py \
  --config finetune/configs/sft_instruct_125m.yaml --preflight-only
```

## Raw code completion

This optional stage derives prompt/body pairs from the already split code SFT
data and checks for prompt or pair leakage:

```bash
make prepare-code-completion SIZE=125m
make sft-code-completion SIZE=125m
make eval-code-completion SIZE=125m
```

The raw trainer masks prompt tokens, reports truncation/supervision statistics,
uses token-weighted validation loss, writes recoverable `checkpoint-<step>`
directories, and promotes `best/` to `final/`. Resume directly with:

```bash
python finetune/train_code_completion.py \
  --config finetune/configs/code_completion_125m.yaml --resume
```

## Validation

Run cheap checks before a full training job:

```bash
python -m pytest tests/test_training_args.py tests/test_trl_smoke.py -q
make test-sft-instruct SIZE=mini
make test-sft-code SIZE=mini
```
