# Utility Scripts

This directory contains diagnostics and recovery helpers that span pipeline
stages. Production logic owned by one stage belongs in that stage's directory.

## Contents

| File | Purpose |
|---|---|
| `sanity_train.py` | Known-good synthetic/Hugging Face training diagnostic |
| `run_lm_eval.py` | Register the local architecture and invoke lm-evaluation-harness |
| `pretrain_hf_125m.py` | Standalone 125M pretraining diagnostic on a Hub dataset |
| `sft_model_comparison.py` | Controlled SmolLM2/SLM response-to-SFT comparison |
| `vllm_smoke.py` | Load one native export in vLLM and generate a bounded response |

## Training diagnostic

`sanity_train.py` isolates model/trainer behavior from the curated-data path.
It can create a small synthetic run or save a diagnostic checkpoint.

```bash
make sanity-train-tiny
make sanity-train-small
make sanity-train
```

Save the selected diagnostic output:

```bash
make sanity-train-save SANITY_SIZE=tiny
```

Direct examples:

```bash
python scripts/sanity_train.py \
  --arch mini \
  --target-tokens 50000000

python scripts/sanity_train.py \
  --arch 125m \
  --target-tokens 2500000000 \
  --save
```

The script exposes architecture, target-token, batch, learning-rate, warmup,
logging/evaluation cadence, scratch-directory, save, and token-reuse controls.
It is diagnostic evidence only; it does not create a pipeline pretraining
checkpoint.

## Evaluation wrapper

Normal evaluation should use the Make targets or `eval/eval.py`. The lower
level wrapper is useful when invoking harness-specific options directly:

```bash
python scripts/run_lm_eval.py \
  --model hf \
  --model_args "pretrained=results/runs/125m/sft_code/final,dtype=bfloat16" \
  --tasks humaneval \
  --num_fewshot 0 \
  --batch_size 1 \
  --apply_chat_template \
  --output_path results/eval/debug_humaneval \
  --log_samples \
  --limit 5
```

HumanEval executes generated code; run it only in an isolated environment.

## Standalone 125M pretraining path

`pretrain_hf_125m.py` downloads a selected text dataset, creates an isolated
125M run, and invokes tokenizer/tokenization/pretraining commands. Use it to
distinguish trainer/model failures from curator or validation failures.

Inspect its required safeguards and defaults before running:

```bash
python scripts/pretrain_hf_125m.py --help
```

This path is not part of the normal stage graph and must not overwrite an
existing run unless the explicit replacement/backup options are supplied.

## Controlled SFT comparison

The comparison harness evaluates whether the local 125M base model is a valid
candidate and how it responds to the same bounded SFT experiment as
SmolLM2-135M.

First create the native base artifact and run preflight:

```bash
make export-base-local SIZE=125m
make compare-sft-preflight
```

Then run the bounded comparison:

```bash
make compare-sft
```

Direct invocation:

```bash
python scripts/sft_model_comparison.py \
  --tohio-model results/exports/125m/base \
  --train-examples 32 \
  --eval-examples 32 \
  --max-steps 60 \
  --output-dir results/diagnostics/sft-comparison
```

The harness performs checkpoint integrity, prompt-sensitivity, and cache-parity
checks; selects one common set of pinned records; builds labels itself; and
reports tokenizer-specific exposure and evaluation outputs. Use
`--preflight-only` to stop before dataset selection and training.

## vLLM export smoke

After a native export passes its conversion and clean-load checks, run one
offline vLLM generation in the serving environment:

```bash
make test-vllm-export \
  SIZE=125m \
  EXPORT_VARIANT=base
```

The script requires a local native Llama export, applies the packaged chat
template, and fails on an empty generation.

## Conventions

- Keep stage production commands with their owning stage.
- Make diagnostics fail loudly instead of repairing inputs silently.
- Require explicit paths or opt-in flags for operations that replace
  checkpoints or prepared data.
- Write reusable results under `$RESULTS_DIR`; use scratch storage only for
  disposable intermediates.
