# DPO Alignment

This directory prepares external preference pairs and trains the chat branch
with Direct Preference Optimization (DPO).

Synthetic preference generation is maintained in
[`tohio/slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data);
this repository validates and consumes preference datasets.

## Contents

| Path | Purpose |
|---|---|
| `configs/dpo_data_sources.yaml` | Pinned preference source and validation policy |
| `configs/dpo_chat_*.yaml` | Size-specific DPO recipes |
| `data/prepare_dpo.py` | Normalize, validate, split, and manifest preference data |
| `train_dpo.py` | Preflight, reference-log-probability computation, DPO training, and promotion |

## Input model

DPO starts from the instruct checkpoint:

```text
$RESULTS_DIR/runs/<size>/sft_instruct/final/
```

It produces the independent chat-aligned branch:

```text
$RESULTS_DIR/runs/<size>/dpo_chat/final/
```

The code SFT branch also starts from instruct SFT; DPO is not applied on top of
the code checkpoint.

## Dataset contract

`configs/dpo_data_sources.yaml` pins
`HuggingFaceH4/ultrafeedback_binarized` to an immutable revision and defines
its adapter, row limits, token budgets, split policy, and quality thresholds.

The 125M, 350M, and 1B profiles use the same deterministic 10,000-record
population. `mini` uses 1,000 records with reduced token budgets.

Prepared records use TRL's conversational preference schema:

```json
{
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Question"}
  ],
  "chosen": [
    {"role": "assistant", "content": "Preferred response"}
  ],
  "rejected": [
    {"role": "assistant", "content": "Rejected response"}
  ],
  "source": "ultrafeedback_binarized",
  "dpo_type": "general_preference"
}
```

Preparation renders prompts with the target tokenizer, verifies a shared
prompt prefix, enforces prompt/total token budgets, rejects identical or
reversed pairs, controls duplicates, and splits by normalized complete prompt.
The resulting manifest records source and tokenizer hashes, selection and
retention statistics, file hashes, and zero prompt overlap.

## Prepare data

```bash
make prepare-dpo SIZE=125m
```

Prepared output:

```text
$DATA_DIR/runs/<size>/dpo_chat/
  train.jsonl
  val.jsonl
  manifest.json
```

After intentionally changing the source contract, replace a prior prepared
dataset with:

```bash
python alignment/data/prepare_dpo.py \
  --size 125m \
  --force
```

Unmanifested or contract-mismatched data is not reused silently.

## Train

Generate the size- and hardware-specific recipe:

```bash
make config-gen-dpo SIZE=125m GPUS=1
```

Start or resume training:

```bash
make dpo-chat SIZE=125m GPUS=1
make dpo-chat-resume SIZE=125m GPUS=1
```

Run the bounded mini stage:

```bash
make dpo-chat-mini SIZE=mini GPUS=1
```

Run all preprocessing and compatibility checks without optimization or a
reference-model pass:

```bash
python alignment/train_dpo.py \
  --config alignment/configs/dpo_chat_125m.yaml \
  --preflight-only
```

The active recipes use sigmoid DPO with `beta: 0.1`, zero label smoothing,
dropout disabled, and reverse-KL divergence. Reference log probabilities are
precomputed from the untouched initial policy, preserving a fixed SFT
reference without retaining a second model during optimization.

The trainer validates model/tokenizer compatibility, prepared-data and
tokenizer hashes, preference schema, split isolation, TRL retention,
non-empty completions, and evaluation/checkpoint cadence. It writes
`dpo_run_audit.json` before optimization and promotes the
lowest-validation-loss checkpoint to `final/` with the audit and data
manifest.

## Validation

Run contract and one-step integration tests:

```bash
python -m pytest \
  tests/test_dpo_data_contract.py \
  tests/test_training_args.py \
  tests/test_trl_smoke.py \
  -q
```

Validate a completed DPO artifact:

```bash
make test-dpo-chat SIZE=125m
```
