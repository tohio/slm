# SLM command reference

This file documents the Makefile command surface. The Makefile is the source of truth.

Default variables:

```bash
SIZE=125m
GPUS=1
INSTALLER=pip
DATA_DIR=data
RESULTS_DIR=results
EXPORTS_DIR=results/exports
RUN_ID=
ARTIFACT_BACKEND=s3
ARTIFACT_STAGES=raw,tokenized,tokenizer,metadata
DATASET_SIZE=$(SIZE)
DATASET_RUN_ID=$(RUN_ID)
```

Common overrides:

```bash
make <target> SIZE=350m
make <target> GPUS=4
make <target> DATA_DIR=/data/slm/data
make <target> RUN_ID=125m-20260629-a8f3c9
```

---

## New-host workflows

On a new CPU curation host, bootstrap the curation models and validate smoke
before starting a larger run:

```bash
make setup-curate DATA_DIR=/data/slm/data
make curate SIZE=smoke
make validate SIZE=smoke
make tokenizer SIZE=smoke
make tokenizer-test SIZE=smoke
make tokenize SIZE=smoke
```

Then optionally select Mini (at least 1.4B usable training tokens) or use
`curate-all` directly for a complete production-size workflow. Mini is not a
mandatory intermediate:

```bash
make curate-all \
  SIZE=125m \
  WORKERS=62 \
  DATA_DIR=/data/slm/data
```

Run the complete model-training workflow on a new GPU host:

```bash
make train-all \
  SIZE=125m \
  GPUS=1 \
  RUN_ID=125m-20260629-a8f3c9 \
  DATA_DIR=/data/slm/data
```

Both commands require common settings including W&B API key/project. Transfer
code validates only its selected storage credentials; model publication requires
`HF_USERNAME`.
`curate-all` creates the corpus and uploads the stages selected by `ARTIFACT_STAGES`. `train-all` restores that artifact run and
trains the base, instruct, code, and chat variants. It is restricted to new
training runs; use the stage-specific resume targets after an interruption.

See [`CURATION.md`](CURATION.md) and [`TRAIN.md`](TRAIN.md) for the complete
contracts and stage-by-stage commands.

---

## Setup

Two host roles, each with pip/venv (default), uv, or conda:

```bash
make setup-curate DATA_DIR=/data/slm/data
make setup-curate INSTALLER=uv DATA_DIR=/data/slm/data
make setup-curate INSTALLER=conda DATA_DIR=/data/slm/data

make setup-train DATA_DIR=/data/slm/data
make setup-train INSTALLER=uv DATA_DIR=/data/slm/data
make setup-train INSTALLER=conda DATA_DIR=/data/slm/data
```

Choose one installer per host; these are alternatives, not a sequence to run in
one checkout. `setup-curate` installs `requirements-curation.txt` and owns KenLM,
FastText/orjson, and curation verification. `setup-train` installs the complete
GPU/evaluation stack from `requirements-training.txt`. Both include shared
`requirements.txt`. All installers use `.venv`. See [setup details](../infra/README.md).

Curation setup includes FastText and KenLM runtime assets and persists `DATA_DIR`
in `.env`; do not run separate download commands after successful setup. These
internal helpers remain available for deliberate asset repair only:

```bash
make download-fasttext-model
make download-kenlm-model
```

The curation prerequisite and training environment checks are internal workflow
safeguards, not additional setup steps to remember.

To set up training and restore a source dataset in the same command:

```bash
make setup-train SIZE=mini DATASET_SIZE=350m DATASET_RUN_ID=350m-YYYYMMDD-abcdef
```

No run ID means setup only. Setup restoration defaults to
`tokenized,tokenizer,metadata`; override `ARTIFACT_STAGES` to include validated
text for test-document completions. Source artifacts remain under `DATASET_SIZE`.

---

## Curation

Run smoke first on a new host:

```bash
make curate SIZE=smoke
make validate SIZE=smoke

make curate SIZE=mini WORKERS=62
make validate SIZE=mini

make curate SIZE=125m WORKERS=62
```

These are alternative curation profiles, not a required sequence.
`curate SIZE=smoke` is the bounded curation run; validation and tokenization are
subsequent stages. `curate SIZE=mini` is an optional larger curation run. Runtime depends on host and network
conditions; fixed duration estimates are intentionally not published.

Stage-specific curation:

```bash
make curate-download SIZE=125m
make curate-filter SIZE=125m WORKERS=62
make curate-dedup SIZE=125m WORKERS=62
make curate-blend SIZE=125m
```

To replace a legacy or stale raw source only after a clean staged download
succeeds:

```bash
make curate-download SIZE=125m FORCE=1
```

Upload only curated artifacts through the RUN_ID flow:

```bash
make curate-upload SIZE=125m
```

---

## Validation

```bash
make validate SIZE=125m
```

Upload only validated artifacts through the RUN_ID flow:

```bash
make validate-upload SIZE=125m
```

---

## Tokenizer and tokenization

```bash
make tokenizer SIZE=125m
make tokenizer-test SIZE=125m
make tokenize SIZE=125m
```

---

## Artifacts

Artifacts are grouped by `RUN_ID`.

Upload:

```bash
make artifacts-upload SIZE=125m
make artifacts-upload SIZE=125m RUN_ID=125m-20260629-a8f3c9
make artifacts-upload SIZE=125m ARTIFACT_STAGES="validated,tokenized,tokenizer,metadata"
```

Download:

```bash
make artifacts-download SIZE=125m RUN_ID=125m-20260629-a8f3c9
make artifacts-download SIZE=125m RUN_ID=125m-20260629-a8f3c9 ARTIFACT_STAGES="validated,tokenized,tokenizer,metadata"
```

Set `ARTIFACT_BACKEND=hf` to route `curated`/`validated` stages to an HF Dataset
repository and the remaining stages to an HF Storage Bucket. S3 remains the
default; no simultaneous upload occurs. Stage selection and explicit overwrite
remain user-controlled, with no retention profiles. See
[artifact routing and restoration](PRETRAINING_DATA.md#user-selected-artifact-transfer)
for credentials, generic Bucket objects, and integrity behavior.

Valid stages:

```text
raw, curated, validated, tokenized, tokenizer, metadata
```

---

## Config generation

Stage configs:

```bash
make config-gen-pretrain SIZE=125m GPUS=1
make config-gen-sft SIZE=125m GPUS=1
make config-gen-dpo SIZE=125m GPUS=1
make config-gen SIZE=125m GPUS=1
```

Hardware override:

```bash
make config-gen SIZE=125m GPUS=4 GPU=h200
make config-gen SIZE=1b GPUS=N GPU=b200 MODE=aggressive
```

The same configuration flow generates the DDP launch file internally for
`GPUS > 1`. Replace `N` with the GPU count you choose and use that count for
configuration and training. Mini uses this flow; Smoke remains separate.

---

## Pretraining

All stage-specific resume targets accept `RESUME_CHECKPOINT=/path/to/checkpoint-N`
to select a verified earlier checkpoint under the configured run output. Without
it, the latest numeric checkpoint is selected. Incomplete recovery state is an
error, never a request to reset the optimizer or silently fall back. See
[recovery requirements](TRAIN.md#recovery-checkpoint-completeness).

```bash
make pretrain-preflight SIZE=125m GPUS=1
make pretrain-mini SIZE=mini GPUS=1
make pretrain SIZE=125m GPUS=1
make pretrain-resume-preflight SIZE=125m GPUS=1
make pretrain-resume SIZE=125m GPUS=1
make pretrain-smoke SIZE=smoke GPUS=1
make smoke-gen SIZE=125m
```

Output:

```text
results/runs/<size>/pretrain/final
```

Bounded readiness gates:

```bash
make test-pretrain-ready SIZE=125m GPUS=1
make test-pretrain-resume-ready SIZE=125m GPUS=1
```

---

## SFT

Prepare data (optionally add `SFT_TOOL_DATA=/path/to/reviewed-tools.jsonl` to
merge reviewed tool/no-tool conversations into instruct preparation):

```bash
make prepare-sft SIZE=125m
```

Sources and immutable Hub revisions are configured in
`finetune/configs/sft_data_sources.yaml`. Prepared splits include a provenance
and integrity manifest; changing the source contract requires an intentional
rerun with `finetune/data/prepare_sft.py --force`.

Training verifies the preparation manifest and split isolation before sampling.
SFT/DPO require the actual parent checkpoint's bundled tokenizer. New-run and
resume protection is documented in [training](TRAIN.md#post-training-identity-and-resume).

Instruct SFT:

```bash
make sft-instruct SIZE=125m GPUS=1
make sft-instruct-resume SIZE=125m GPUS=1
make sft-instruct-mini SIZE=mini GPUS=1
```

Compatibility aliases:

```bash
make sft SIZE=125m GPUS=1
make sft-mini SIZE=mini GPUS=1
```

Code SFT:

```bash
make sft-code SIZE=125m GPUS=1
make sft-code-resume SIZE=125m GPUS=1
make sft-code-mini SIZE=mini GPUS=1
```

Raw code-completion path:

```bash
make prepare-code-completion SIZE=125m
make sft-code-completion SIZE=125m
make eval-code-completion SIZE=125m
```

Outputs:

```text
results/runs/<size>/sft_instruct/final
results/runs/<size>/sft_code/final
results/runs/<size>/sft_code_completion/final
```

---

## DPO

Prepare data:

```bash
make prepare-dpo SIZE=125m
```

The pinned source and preference-quality contract are configured in
`alignment/configs/dpo_data_sources.yaml`. Prepared data includes a manifest;
changing the contract requires an intentional
`alignment/data/prepare_dpo.py --force` run. Preparation resolves the instruct
checkpoint from `DPO_CHAT_CONFIG`, shared with training. Optional
`DPO_BASE_MODEL=/path/to/instruct/final` overrides the same parent for both paths.
The preparer also accepts `--training-config` and `--base-model` directly.
Tokenizer/template changes invalidate the DPO rendering fingerprint.

Train:

```bash
make dpo-chat SIZE=125m GPUS=1
make dpo-chat-resume SIZE=125m GPUS=1
make dpo-chat-mini SIZE=mini GPUS=1
```

Compatibility aliases:

```bash
make dpo SIZE=125m GPUS=1
make dpo-resume SIZE=125m GPUS=1
make dpo-mini SIZE=mini GPUS=1
```

Output:

```text
results/runs/<size>/dpo_chat/final
```

---

## Evaluation

Benchmark evaluation is optional. All normal `eval-*` targets use `SIZE` and
`runs/<size>/` for Mini, 125M, 350M, and 1B; Smoke remains separate.

```bash
make eval-pretrain-final SIZE=mini
make pretrain-probes SIZE=mini
```

Final pretraining evaluation uses the matched test data/provenance. Generic
probes and corpus-supported QA are separate from test loss/perplexity.

```bash
make eval-base SIZE=125m
make eval-instruct SIZE=125m
make eval-chat SIZE=125m
make eval-code SIZE=125m
make eval SIZE=125m
make eval-base SIZE=mini
```

Sanity evaluation:

```bash
make eval-sanity-base SIZE=125m
make eval-sanity-instruct SIZE=125m
make eval-sanity-chat SIZE=125m
make eval-sanity-code SIZE=125m
make eval-sanity SIZE=125m
```

`eval` and `eval-sanity` default to the chat variant.

---

## Export

Build and validate local native artifacts without publishing:

```bash
make export-base-local SIZE=125m
make export-instruct-local SIZE=125m
make export-chat-local SIZE=125m
make export-code-local SIZE=125m
make export-local SIZE=125m
```

Build, validate, and publish:

```bash
make export-base SIZE=125m
make export-instruct SIZE=125m
make export-chat SIZE=125m
make export-code SIZE=125m
make export SIZE=125m
```

Hub names:

```text
tohio/slm-<size>
tohio/slm-<size>-instruct
tohio/slm-<size>-chat
tohio/slm-<size>-code
```

---

## Serving

```bash
make serve SIZE=125m
make serve-local SIZE=125m
```

---

## Tests

Data pipeline:

```bash
make test-curator
make test-validate
make test-tokenizer
make test-data-pipeline
```

GPU pipeline:

```bash
make test-training
make test-sft-instruct
make test-sft-code
make test-dpo-chat
make test-gpu-pipeline
```

Compatibility aliases:

```bash
make test-sft-chat
make test-dpo
```

Shared/curation tests (no real corpus required):

```bash
make test-data-unit       # either installed role
make test-curation-unit   # curation role, includes shared tests
```

Training/common unit tests (CPU-executed tests within the pinned training
stack installed on a GPU host by `make setup-train`):

```bash
make test-model
make test-export
make test-data-unit
make test-training-args
make test-config-gen
make test-accel-gen
make test-comparison
make test-misc
make test-unit
```

GPU environment gate:

```bash
make test-gpu-gate
make test-upgrade-gpu
```

The two targets are equivalent. Run the gate once per GPU image or dependency
upgrade; it does not download training data or load a trained checkpoint.

---

## Diagnostics

```bash
make compare-sft-preflight
make compare-sft
make sanity-train
make sanity-train-small
make sanity-train-tiny
make sanity-train-save SIZE=mini SANITY_TARGET_TOKENS=50000000
```

`sanity-train` uses the actual `SIZE`/`SANITY_CONFIG` recipe. Select
`SANITY_STAGE=prepare|check|train|all`, an isolated `SANITY_RUN_DIR`, and optionally
`SANITY_TOKENIZED` for read-only existing inputs. The default HF corpus is
FineWeb-Edu and the default tokenizer is the existing size-specific tokenizer.
`SANITY_TARGET_TOKENS` selects usable train tokens; holdouts are extra.
`SANITY_MAX_STEPS` bounds the training control, and `SANITY_PROBE_EVERY_STEPS`
overrides only the diagnostic generation-probe cadence (for example, `250`) while
reusing the production probe callback. `SANITY_BACKEND=llama` selects the matched native control arm.
`SANITY_EVAL_TOKENIZED` supplies a shared validation distribution and requires
its original validated val/test documents for exact exclusion from HF selection.
Small/tiny are token-budget presets, not alternate architectures. These commands
never replace production checkpoints. See [control design and safeguards](../scripts/README.md).

---

## Cleanup

These targets delete local files. Confirm `DATA_DIR`, `RESULTS_DIR`, and
`SIZE` before running them.

```bash
make clean-data SIZE=125m
make clean-results
make clean-logs
make clean
```
