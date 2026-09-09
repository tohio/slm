# Pretraining

This directory converts the validated corpus to memory-mapped token arrays and
trains a base SLM model from scratch.

## Contents

| Path | Purpose |
|---|---|
| `data/tokenize_data.py` | Encode validated JSONL with the run tokenizer |
| `data/dataset.py` | Read fixed-length windows from memory-mapped token arrays |
| `train.py` | Build the model, run causal-language-model training, resume, and save |
| `configs/gpt_*.yaml` | Size-specific model and optimization recipes |

Post-training branches are owned by `finetune/` and `alignment/`.

## Data contract

Tokenization reads:

```text
$DATA_DIR/runs/<size>/validated/train.jsonl
$DATA_DIR/runs/<size>/validated/val.jsonl
$DATA_DIR/runs/<size>/validated/test.jsonl
$DATA_DIR/runs/<size>/validated/test_contract.json
$DATA_DIR/runs/<size>/tokenizer/slm_tokenizer.json
```

It writes:

```text
$DATA_DIR/runs/<size>/tokenized/train.bin
$DATA_DIR/runs/<size>/tokenized/train.json
$DATA_DIR/runs/<size>/tokenized/val.bin
$DATA_DIR/runs/<size>/tokenized/val.json
$DATA_DIR/runs/<size>/tokenized/test.bin
$DATA_DIR/runs/<size>/tokenized/test.json
$DATA_DIR/runs/<size>/tokenized/test_contract.json
$DATA_DIR/runs/<size>/tokenized/token_mixture.json
```

`token_mixture.json` expands the configured code bucket into concrete sources
and compares those intended shares with the combined train/validation/test token
counts measured by the tokenizer. It records percentage-point deviations but
does not impose an uncalibrated deviation threshold. Tokenization fails on
unknown or corpus-wide missing sources and inconsistent counts; pretraining
requires the report to match both the current data-mix contract and split
metadata.

The JSON sidecars record the input digest, tokenizer fingerprint, binary
format, and realized source/document/token counts. Training requires those
sidecars and rejects a tokenizer that does not match the tokenized corpus.

Functional and production recipes resolve `max_steps` from the verified tokenized training
count and the configured epoch contract at preflight/training time. The static
YAML step and warmup values remain planning fallbacks; runtime preserves their
warmup ratio. The smoke recipe remains fixed at eight optimizer steps.
Curation establishes train/val/test membership; pretraining does not
split documents again. Validation is training-time; test is final-only.

The smoke profile uses the 21.7M-parameter architecture and capped 1M-token
corpus only to exercise execution and artifact contracts. The mini profile is
a 69.9M-parameter functional pilot trained for one epoch over its realized
selected tokenized corpus, with a fail-closed minimum of 1.4B usable train
tokens. Mini is optional and is not a second lightweight smoke test.

`dataset.py` memory maps the flat token arrays and returns fixed-length causal
language-model windows without loading the entire corpus into memory.

## Configuration

Available profiles:

```text
pretrain/configs/gpt_smoke.yaml
pretrain/configs/gpt_mini.yaml
pretrain/configs/gpt_125m.yaml
pretrain/configs/gpt_350m.yaml
pretrain/configs/gpt_1b.yaml
```

Generate the selected recipe on the target training host:

```bash
make config-gen-pretrain SIZE=125m GPUS=1
```

Smoke generates pretraining recipes only. Mini uses the same pretraining,
SFT, and DPO generation flow as production sizes, while retaining its smaller
post-training sample/step budgets.

To generate pretraining, SFT, and DPO recipes together:

```bash
make config-gen SIZE=125m GPUS=1
```

Configuration generation uses the selected model size, GPU type/count, VRAM
policy, and effective-batch targets. Inspect the resulting YAML before
starting an expensive run.

## Usage

Encode and verify the corpus:

```bash
make tokenize SIZE=125m
```

Equivalent direct command:

```bash
python pretrain/data/tokenize_data.py \
  --size 125m \
  --chunk-size 256 \
  --verify
```

Training resolves the tokenizer and all binaries/metadata directly from
`DATASET_SIZE` (default `SIZE`). No global-tokenizer activation/copy is needed.
For migration and cross-size budgets, read
[`PRETRAINING_DATA.md`](../docs/PRETRAINING_DATA.md) before replacing an old
Mini tokenizer or resuming an old checkpoint.

Start or resume pretraining:

```bash
make pretrain-preflight SIZE=125m GPUS=1
make pretrain SIZE=125m GPUS=1

make pretrain-resume-preflight SIZE=125m GPUS=1
make pretrain-resume SIZE=125m GPUS=1
```

The preflight validates CUDA/BF16 availability, visible GPU count, model and
tokenizer contracts, tokenized manifests/binaries, output-directory state,
token budget, and resume provenance without allocating model weights.

Multi-GPU behavior is unchanged: the Make target passes `GPUS` to Accelerate
as `--num_processes`, producing one process per GPU:

```bash
make pretrain SIZE=350m GPUS=4
```

Run the bounded execution rehearsal:

```bash
make pretrain-smoke SIZE=smoke GPUS=1
```

Run the Mini plumbing pilot (this regenerates its hardware config for `GPUS`):

```bash
make pretrain-mini SIZE=mini GPUS=1
```

Direct invocation:

```bash
accelerate launch \
  pretrain/train.py \
  --config pretrain/configs/gpt_125m.yaml

accelerate launch \
  pretrain/train.py \
  --config pretrain/configs/gpt_125m.yaml \
  --resume
```

## Outputs and resume

Checkpoints and the promoted base model are stored at:

```text
$RESULTS_DIR/runs/<size>/pretrain/checkpoint-<step>/
$RESULTS_DIR/runs/<size>/pretrain/final/
```

The run root contains `pretrain_run_audit.json`. It binds the resolved
training configuration, tokenizer fingerprint, tokenized-data identity,
process count, and distributed strategy. `--resume` requires the audit and
selected complete checkpoint and refuses changed inputs instead of starting over.
Missing optimizer, scheduler, per-rank RNG, Trainer or required FP16 scaler state
causes an error even when model weights are present. An earlier checkpoint can be
selected with `RESUME_CHECKPOINT=...`; see [recovery requirements](../docs/TRAIN.md#recovery-checkpoint-completeness).

The configured seed is applied before fresh weight initialization. On resume,
Trainer restores the selected checkpoint and its RNG state; construction does
not overwrite saved learned weights. A recorded seed is not a guarantee of
bitwise results across different hardware/runtime versions.

The audit and a small `training_provenance.json` bundle are copied into `final/`.
The bundle captures the source dataset size/run, verified tokenizer-measured
source mix, selected unique token budget, and observed input-token count when
counting covered the full run. Legacy resumes without that guarantee report the
count as unrecorded. No later export reads a mutable model-size data directory
to reconstruct this history. The final checkpoint is the parent of instruct SFT
and is consumed without post-pretraining embedding mutation.

Run the base generation smoke check after a completed pretraining run:

```bash
make smoke-gen SIZE=125m
```

## Validation

Run the architecture and training-contract tests before a paid run:

```bash
make test-pretrain-ready SIZE=125m GPUS=1
```

Validate a completed pretraining artifact:

```bash
make test-training SIZE=125m
```

The artifact test expects an existing checkpoint; it is not a substitute for
pretraining and does not launch another full run.

## Final evaluation and generation probes

`generation_probes` configures fixed pretraining prompts, sparse cadence, and
deterministic decoding. Raw base-model probes explicitly prepend exactly one BOS
token to match the BOS/document/EOS pretraining contract; they do not rely on the
tokenizer's `add_special_tokens` behavior. Final reports separate test-prefix completions, generic
prompts, and supplied corpus-supported QA. Use `make pretrain-probes` for saved
checkpoints and `make eval-pretrain-final` for completed models with matching test
provenance. These are not training pass/fail gates. See
[pretraining data and evaluation](../docs/PRETRAINING_DATA.md).

## Isolated reference-data diagnostic

To separate model implementation from corpus effects, use the existing
[HF control scripts](../scripts/README.md). They read the generated recipe and
reuse this stage's encoder, packed dataset, training arguments, schedule resolver,
and `SLMTrainer`. They do not overwrite a production run or pretend a single HF
source satisfies the production mixture contract. Compare identical initial
weights against native Llama first; a passing numerical check is not a claim
about learned generation quality.
