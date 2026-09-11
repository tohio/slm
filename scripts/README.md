# Utility Scripts

This directory contains diagnostics and recovery helpers that span pipeline
stages. Production logic owned by one stage belongs in that stage's directory.

## Contents

| File | Purpose |
|---|---|
| `sanity_train.py` | Matched-weight SLM/native Llama check and FineWeb-Edu learning control |
| `run_lm_eval.py` | Register the local architecture and invoke lm-evaluation-harness |
| `pretrain_hf_125m.py` | Shared isolated HF preparation and production-Trainer control; DCLM-Edu default |
| `sft_model_comparison.py` | Controlled SmolLM2/SLM response-to-SFT comparison |
| `vllm_smoke.py` | Load one native export in vLLM and generate a bounded response |

## HF reference-data and implementation controls

These diagnostics separate questions instead of assuming a cause from generation
quality. Use the installed **training** environment. Required W&B key/project
settings remain required for actual training. A numerical `check` does not start
an online experiment or train a checkpoint.

Both entry points read the selected generated pretraining YAML. There is no
separate architecture registry: `--config pretrain/configs/gpt_mini.yaml` uses
that file's actual Mini dimensions. `--size`/legacy `--arch`, when supplied,
must agree with the YAML. Do not regenerate or edit the recipe between control
arms. Start from scratch; do not resume the completed curated-data model.

### 1. Check the model implementation without a dataset download

Use the existing tokenized training data and its exact original tokenizer:

```bash
make sanity-train SIZE=mini SANITY_STAGE=check \
  SANITY_RUN_DIR="$PWD/data/slm/results/diagnostics/mini-implementation-check" \
  SANITY_TOKENIZER="$PWD/data/slm/data/runs/mini/tokenizer" \
  SANITY_TOKENIZED="$PWD/data/slm/data/runs/mini/tokenized"
```

The paths above match the default layout; substitute your configured paths when
different. No dataset or trained checkpoint is changed. The check uses **fresh,
identical initial tensors**, not the final Mini model's weights.

`checks/implementation_check.json` records the initial state hash, architecture,
input identity, package/source versions, tolerances and per-check differences.
The comparison checks unpadded FP32 logits, independently computed next-token
loss, the production Trainer's target counting/loss path, causal invariance,
named gradients, optimizer grouping, and corresponding AdamW updates. It uses
two short contexts drawn from the selected training stream by default; it does
not claim full-context, compiled, BF16, distributed, or learned-capability
acceptance. Failed checks produce a nonzero exit and a failure report. Do not
increase tolerances merely to make a discrepancy pass.

### 2. Substitute HF data while keeping the learning components fixed

`sanity_train.py` defaults to `HuggingFaceFW/fineweb-edu`, `sample-10BT`.
`pretrain_hf_125m.py` defaults to `HuggingFaceTB/dclm-edu`; the historical filename
is retained, but `--config` now also supports Mini and other existing profiles.
The two entry points share preparation and training code, not two optimizer loops.
`--stage all` launches each phase in a fresh Python process so the FP32 numerical
comparison cannot leave Accelerate precision state in the subsequent BF16 run.
Neither trains a new tokenizer implicitly. The default is the selected size's
existing tokenizer; specify `--tokenizer-dir` to make the control explicit.

For a Mini control matching the completed run's **1,476,931,584 usable train
tokens**, run these stages separately:

```bash
make sanity-train SIZE=mini SANITY_STAGE=prepare \
  SANITY_RUN_DIR="$PWD/data/slm/results/diagnostics/mini-fineweb-control" \
  SANITY_TARGET_TOKENS=1476931584 \
  SANITY_EVAL_TOKENIZED="$PWD/data/slm/data/runs/mini/tokenized"

make sanity-train SIZE=mini SANITY_STAGE=check \
  SANITY_RUN_DIR="$PWD/data/slm/results/diagnostics/mini-fineweb-control" \
  SANITY_EVAL_TOKENIZED="$PWD/data/slm/data/runs/mini/tokenized"

make sanity-train SIZE=mini SANITY_STAGE=train \
  SANITY_RUN_DIR="$PWD/data/slm/results/diagnostics/mini-fineweb-control" \
  SANITY_EVAL_TOKENIZED="$PWD/data/slm/data/runs/mini/tokenized"
```

Preparation streams/shuffles the source with a recorded seed and resolves the
requested dataset revision to an immutable commit **before loading**. Supply
`--dataset-revision <commit>` directly to reproduce that selection elsewhere;
otherwise `main` is resolved once and the resolved SHA is retained in the cache.
Streaming shuffle is a bounded-buffer/shard shuffle, not a globally uniform
sample. It does not establish semantic quality merely because data is hosted
on HF.

Splits are assigned by normalized whole-document hashes **before** packing;
normalized exact duplicates are removed, not split across training/holdouts.
The diagnostic defaults remain 99/0.5/0.5 by document hash. This does **not**
implement the pending project-wide 98/1.5/0.5 TODO. Fractions are expectations,
not exact document or token quotas. Validation/test must each contain at least
four full windows. All three files, their holdout identities and their binary
checksums must be complete before the bundle is published. A stopped/short
source or corrupt cache fails; preallocated zeros are never counted as tokens.

`SANITY_EVAL_TOKENIZED` supplies the **same validation distribution** for the
curated/HF comparison in addition to each control's own heldout. HF preparation
requires the original adjacent `validated/val.jsonl` and `test.jsonl` to match
their token sidecars and excludes normalized exact copies of those documents.
This is not a near-duplicate/benchmark-decontamination claim. Additional exact
exclusions can be supplied with repeatable `--exclude-jsonl` arguments. Do not
select changes based on repeated final-test results.

`--target-tokens` counts usable unique **training** tokens; holdouts are extra.
Preparation preserves the last whole document, and training uses a bounded
prefix of complete windows. Without a target, HF preparation defaults to 50M;
existing tokenized inputs default to their whole usable training stream. On
subsequent `check`/`train` stages, an omitted target uses the recorded cache target.
The production schedule resolver determines epochs/updates/warmup from those
selected training tokens. `--max-steps` is an explicit, recorded bounded-run
override, not evidence that a production minimum has been met.

Training reuses `pretrain.data.tokenize_data`'s literal-safe BOS/document/EOS
encoder, `PretrainingDataset`, `build_training_args`, the schedule resolver,
and `SLMTrainer`. It intentionally bypasses raw curation/quality filtering and
the production entry point's **15-source mixture** gate. It does not fake the
missing sources or weaken any production gate. Reference manifests are labelled
`diagnostic_not_production_curation`; checkpoints carry `reference_run_audit.json`,
not a publishable pretraining audit. Normal production export therefore remains
unavailable for these diagnostic outputs.

### 3. Optional native learning arm on identical HF tokens

After the shared preparation, train the native implementation using exactly the
same configuration, tokenizer, initial SLM tensors, token order, and Trainer:

```bash
make sanity-train SIZE=mini SANITY_STAGE=train SANITY_BACKEND=llama \
  SANITY_RUN_DIR="$PWD/data/slm/results/diagnostics/mini-fineweb-control" \
  SANITY_EVAL_TOKENIZED="$PWD/data/slm/data/runs/mini/tokenized"
```

Both arms save initial tensor hashes so identical starts can be verified. The
native arm uses the existing in-memory conversion helper; it does not publish
or invoke the production export CLI. Training is single-process. Choose one
visible GPU, for example `CUDA_VISIBLE_DEVICES=0`; more than one visible CUDA
GPU is rejected rather than silently using DataParallel. These diagnostics do
not add another distributed trainer. `--device cpu` is intended for bounded
checks; use `CUDA_VISIBLE_DEVICES=''` for a CPU control on a GPU host.

Results are under:

```text
<run-dir>/data/runs/<size>/
  reference/{train,val,test}.jsonl + test_contract.json
  tokenizer/
  tokenized/{train,val,test}.bin + sidecars + test_contract.json + _SUCCESS.json
  reference_data.json
<run-dir>/checks/implementation_check.json
<run-dir>/results/runs/<size>/{slm,llama}/pretrain/
  reference_run_audit.json
  baseline.json
  checkpoint-*/
  final/                         # diagnostic checkpoint and its tokenizer
  learning_report.json
```

The learning report contains heldout metrics, actual consumed input tokens,
source identity through its immutable run audit, and fixed greedy continuations with explicit BOS. Diagnostic
training also reuses the production qualitative-probe callback; use
`SANITY_PROBE_EVERY_STEPS=<N>` / `--probe-every-steps <N>` to tighten the cadence
for a bounded control without editing the production recipe. Probe JSON is written
under the diagnostic pretrain output's `probes/` directory and remains non-gating. All
training saves a checkpoint; `sanity-train-save`/`--save` are compatibility
aliases, not separate retention policies. `sanity-train-small` and
`sanity-train-tiny` choose 500M and 50M train-token targets respectively, without
changing the selected `SIZE` or maintaining special miniature architectures.
A small execution check is not a capability benchmark.

### Safety, reuse, and interpretation

Use a dedicated `--run-dir`. Existing non-diagnostic directories, production
run directories, symlinked output trees and input/output overlaps are rejected.
The old destructive `--backup-existing`/`--force --no-backup` and arbitrary shell
command hooks are removed. No `.env`, model recipe, tokenizer vocabulary,
existing dataset, or Mini checkpoint is overwritten. Stale unmanifested token
files are not migrated. A killed preparation can leave `.reference-partial-*`;
inspect that private partial directory rather than treating it as a cache.

`--reuse-tokens` (Make: `SANITY_REUSE_TOKENS=1`) permits `prepare`/`all` to reuse
only a matching complete bundle; `train` and `check` always verify their inputs.
Changes to source/tokenizer revisions, seeds, selection, exclusions, package versions or encoding
logic require a new data control. For nondefault source/split/shuffle options, repeat those options when reusing
that bundle; mismatches are rejected, not silently ignored. Training has its own immutable input audit;
`--resume [checkpoint-path]` uses the shared complete-checkpoint resolver and
never silently starts fresh. Retain the audit and tokenizer/data bundle for
resume. Use a fresh directory for another numerical check rather than replacing
its report.

For a separate **reference-tokenizer** experiment, explicitly pass
`--reference-tokenizer mistralai/Mistral-7B-v0.1` and optionally
`--tokenizer-revision <commit>`. IDs and vocabulary are checked/aligned before
initialization. No learned embeddings are resized/reinitialized. This changes
tokenization as well as corpus selection and is **not a data-only comparison**.

A passing numerical comparison supports the tested implementation contracts;
it does not certify learned capability. Better HF learning through the same
SLM components supports a data/selection explanation, but does not prove all
custom-data issues. If both native and SLM fail similarly, investigate shared
preprocessing/optimization/evaluation. Success in these diagnostics is not
acceptance of the production orchestration, restore, or publication paths.

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

## `pretrain_reference_independent.py`

Fully independent model-complete pretraining control. It shares the selected
pretraining YAML **values** and the exact raw `train.jsonl`, `val.jsonl`, and
`test.jsonl` documents with the comparison run, but intentionally shares no SLM
training implementation artifacts.

The control uses the ungated TinyLlama base/intermediate repository tokenizer by
default (`TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T`). TinyLlama uses
the Llama-2 architecture/tokenizer family and exposes a 32K tokenizer, matching
the Mini recipe vocabulary without embedding resize. The tokenizer revision is
resolved to an immutable Hub commit and recorded.

The independent path does **not** import or use `model/`, the local SLM
tokenizer, existing tokenized `.bin` files, production packing,
`PretrainingDataset`, `SLMTrainer`, the SLM schedule resolver,
`_convert_to_native_llama`, export, or SLM inference code. It independently
performs external tokenization and BOS/EOS packing, constructs
`transformers.LlamaConfig` + `LlamaForCausalLM` directly, and trains with stock
`transformers.Trainer`.

The raw JSONL splits are intentionally shared because data is the controlled
variable. Use the same raw FineWeb-Edu `reference/` directory as the matched
control. A result difference therefore identifies a difference somewhere in the
**complete SLM pretraining stack versus the independent native-HF stack**; by
itself it does not localize that difference to the transformer block implementation.
Use the matched `SANITY_STAGE=check` control for that narrower localization.

Example matching the recent two-pass Mini experiment values:

```bash
CUDA_VISIBLE_DEVICES=0 make sanity-pretrain-independent \
  SIZE=mini \
  SANITY_INDEPENDENT_REFERENCE_DATA="$PWD/data/slm/results/diagnostics/mini-fineweb-control/data/runs/mini/reference" \
  SANITY_INDEPENDENT_RUN_DIR="$PWD/data/slm/results/diagnostics/mini-fineweb-independent" \
  SANITY_INDEPENDENT_EPOCHS=2 \
  SANITY_INDEPENDENT_LR=1e-4 \
  SANITY_INDEPENDENT_BATCH_SIZE=32 \
  SANITY_INDEPENDENT_GRAD_ACCUM=1
```

Actual training requires W&B and exactly one visible CUDA GPU. Use a fresh run
directory. No production checkpoint or tokenizer is modified.
