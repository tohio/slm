# DPO alignment

This directory consumes external preference data and trains the chat-aligned
SLM branch with Direct Preference Optimization. Synthetic preference generation
and repair belong to the separate `slm-synthetic-data` repository.

## Data contract

`configs/dpo_data_sources.yaml` pins the Hub dataset, split, immutable
revision, deterministic row cap, token budgets, and quality thresholds.

The temporary source is
`HuggingFaceH4/ultrafeedback_binarized@3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`.
The 125M, 350M, and 1B recipes consume the same deterministic 10,000-row
population with identical 1,024/2,048 prompt/total token budgets. `mini`
consumes 1,000 rows with smaller budgets.

To replace UltraFeedback later, update only `configs/dpo_data_sources.yaml`.
This repository should not create, repeat, or silently repair synthetic
preference pairs. The `conversational_preference` adapter accepts either
string responses or assistant-message lists under `chosen` and `rejected`.

Prepare data:

```bash
make prepare-dpo SIZE=125m
```

Existing unmanifested or mismatched data requires intentional replacement:

```bash
python alignment/data/prepare_dpo.py --size 125m --force
```

Prepared output:

```text
data/runs/<size>/dpo_chat/
  train.jsonl
  val.jsonl
  manifest.json
```

Records use TRL’s conversational preference schema:

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "chosen": [{"role": "assistant", "content": "preferred response"}],
  "rejected": [{"role": "assistant", "content": "rejected response"}],
  "source": "ultrafeedback_binarized",
  "dpo_type": "general_preference"
}
```

Preparation uses the target tokenizer’s exact chat rendering. It rejects
prefix mismatches, over-budget pairs, identical responses, excessive
duplicates, and reversed preferences. Splitting is grouped by the normalized
complete prompt, guaranteeing zero prompt overlap. File and tokenizer hashes
are recorded in the manifest.

## Training

DPO starts from:

```text
results/runs/<size>/sft_instruct/final
```

Commands:

```bash
make dpo-chat SIZE=125m GPUS=1
make dpo-chat-resume SIZE=125m GPUS=1
make dpo-chat-mini SIZE=mini GPUS=1
```

The recipe explicitly uses standard sigmoid DPO, `beta: 0.1`, reverse KL,
zero label smoothing, and disabled dropout. Reference log probabilities are
precomputed from the untouched initial policy. This preserves a fixed SFT
reference while avoiding a second resident model during optimization.

Before optimization, the runner validates the model/tokenizer contract, data
and tokenizer hashes, preference schema, split isolation, TRL preprocessing
retention, non-empty completions, and evaluation/checkpoint cadence.

Run those checks without a reference pass or optimization:

```bash
python alignment/train_dpo.py \
  --config alignment/configs/dpo_chat_125m.yaml \
  --preflight-only
```

The lowest-validation-loss checkpoint is saved to:

```text
results/runs/<size>/dpo_chat/final
```

`final/` also contains `dpo_run_audit.json` and `dpo_data_manifest.json`.

## Validation

```bash
python -m pytest \
  tests/test_dpo_data_contract.py \
  tests/test_training_args.py \
  tests/test_trl_smoke.py \
  -q

make test-dpo-chat SIZE=mini
```
