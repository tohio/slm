# Testing

This guide defines when to run each test layer without repeating expensive
data or training stages.

## Test policy

1. Run CPU contracts before starting a paid or long-running stage after code
   changes.
2. Run the GPU acceptance gate once for each new GPU image, GPU type, or
   dependency/CUDA upgrade.
3. Run the matching artifact test immediately after the stage that produced
   the artifact.
4. Do not launch full curation or training merely to execute a test. Artifact
   tests inspect outputs that already exist.

## Test layers

| Gate | Command | Required input |
|---|---|---|
| CPU-executed model/training contracts | `make test-unit` | Already installed pinned GPU training environment |
| Shared data contracts | `make test-data-unit` | Curation or training environment; no real corpus required |
| Curation-only unit contracts | `make test-curation-unit` | Curation environment; no model assets or real corpus |
| GPU acceptance | `make test-gpu-gate` | Supported NVIDIA environment |
| New pretraining readiness | `make test-pretrain-ready SIZE=<size> GPUS=<n>` | GPU environment and restored tokenized artifacts |
| Resume readiness | `make test-pretrain-resume-ready SIZE=<size> GPUS=<n>` | Compatible pretraining audit and checkpoint |
| Curator artifacts | `make test-curator SIZE=<size>` | Completed curated corpus |
| Validation artifacts | `make test-validate SIZE=<size>` | Completed validated corpus |
| Tokenizer behavior | `make tokenizer-test SIZE=<size>` | Trained tokenizer |
| Complete data pipeline | `make test-data-pipeline SIZE=<size>` | Curated, validated, tokenizer, and tokenized artifacts |
| Pretraining artifact | `make test-training SIZE=<size>` | Final base checkpoint |
| Instruct SFT artifact | `make test-sft-instruct SIZE=<size>` | Final instruct checkpoint |
| Code SFT artifact | `make test-sft-code SIZE=<size>` | Final code checkpoint |
| DPO artifact | `make test-dpo-chat SIZE=<size>` | Final chat checkpoint |
| Complete GPU pipeline | `make test-gpu-pipeline SIZE=<size>` | All final training checkpoints |
| Native export acceptance | `make test-export-acceptance SIZE=<size> EXPORT_VARIANT=<variant>` | Completed source checkpoint |
| vLLM export smoke | `make test-vllm-export SIZE=<size> EXPORT_VARIANT=<variant>` | Native export, CUDA, and vLLM environment |

## CPU model and training contracts

On a supported NVIDIA GPU host, prepare the training environment once using
`make setup-train`. In that installed environment, run:

```bash
make test-unit
```

These tests execute model contracts on CPU; that does **not** mean setup supports
a CPU-only workstation. `setup-train` requires a working NVIDIA driver and
installs the pinned CUDA training/evaluation stack. No separate CPU installer is
provided. Curation uses `setup-curate` and must not be layered into that `.venv`.

The gate covers architecture, native checkpoint loading, configuration, data
contracts, export, training arguments, generated configurations, one-step
synthetic SFT/DPO, and repository consistency. Each model-facing Make target
runs `check-training-env` first and fails before pytest if the pinned training
versions are not installed. Focused targets are listed in
[`COMMANDS.md`](COMMANDS.md).

## Shared and curation-only unit scope

`test-data-unit` selects the existing data-config, curator-state, realized-mixture,
fresh tokenization, SFT-data, and DPO-data contract modules. It works with either
role's dependencies and includes complete split-writing and post-training
manifest/identity checks with bounded fixtures.

On the curation host, `make test-curation-unit` runs those shared modules plus
benchmark contamination, Common Crawl source, curation audit stats, dedup
partitioning, exact overlap, long-document segmentation, near overlap, quality
filtering, sensitive-content, and KenLM-validation modules. These are the
existing suites; the gate does not run production curation or download models.

`test-unit` remains the training/common aggregate. It does not select
DataTrove-dependent suites. Direct pytest discovery is broader than either role
gate and can include tests needing the other environment or real artifacts.

## GPU acceptance

```bash
make test-gpu-gate
```

This dataset-free gate checks the pinned CUDA environment, native compute
capability, BF16, eager and compiled optimization, and cached/uncached
generation. It downloads no dataset and loads no trained checkpoint.

## Data pipeline gates

Run each stage gate before proceeding:

```bash
make test-curator SIZE=125m
make test-validate SIZE=125m
make tokenizer-test SIZE=125m
```

After binary tokenization, run the aggregate:

```bash
make test-data-pipeline SIZE=125m
```

The aggregate reruns the artifact checks but does not rebuild the corpus,
retrain the tokenizer, or retokenize the data.

## Training artifact gates

Before starting a new pretraining run:

```bash
make test-pretrain-ready SIZE=125m GPUS=1
```

Before resuming:

```bash
make test-pretrain-resume-ready SIZE=125m GPUS=1
```

These bounded gates run the training/configuration contracts, CUDA acceptance,
and pretraining preflight. The preflight itself does not allocate model weights
or optimize; the model/TRL/CUDA acceptance tests do allocate tiny fixture models
and perform bounded synthetic steps. With `GPUS>1`, preflight verifies that the requested number of
devices is visible; the actual training command retains the existing
Accelerate multi-process launch.

Run the matching gate after each completed training stage:

```bash
make test-training SIZE=125m
make test-sft-instruct SIZE=125m
make test-sft-code SIZE=125m
make test-dpo-chat SIZE=125m
```

The Make targets require the requested artifacts. A missing final checkpoint
fails instead of silently skipping.

## Export and serving acceptance

Build one real native package and require source/native logit parity,
deterministic generation parity, and clean AutoConfig/AutoTokenizer/AutoModel
loading:

```bash
make test-export-acceptance SIZE=125m EXPORT_VARIANT=base
```

The serving environment can then load that artifact with vLLM and generate one
bounded response:

```bash
make test-vllm-export SIZE=125m EXPORT_VARIANT=base
```

## Smoke rehearsal and optional Mini

On the curation host, exercise the bounded `smoke` namespace through the writer,
not merely source processing:

```bash
make curate-smoke
make validate SIZE=smoke
make tokenizer SIZE=smoke
make tokenizer-test SIZE=smoke
make tokenize SIZE=smoke
```

Transfer the selected matched artifacts using the existing artifact workflow.
On a separately prepared training host, after restoration:

```bash
make pretrain-smoke SIZE=smoke GPUS=1
```

Mini is optional and substantially larger: approximately 69.9M parameters and
at least 1.4B usable selected training tokens. It is not required before a
production profile. Follow [curation](CURATION.md) and [training](TRAIN.md) using
`SIZE=mini`, keeping setup roles separate and generating config for the selected
`GPUS` count. Neither smoke nor Mini demonstrates production model quality.

Data artifact gates run on the curation host; GPU artifact gates run on the
training host. Do not assume the combined `test-artifacts` alias makes both
incompatible dependency stacks available in one environment.

## Controlled SFT comparison

Run the fail-fast model checks before the comparison performs dataset
selection or optimization:

```bash
make export-base-local SIZE=125m
make compare-sft-preflight
make compare-sft
```

The comparison uses the same pinned record identities, ordering, optimizer
update schedule, and completion-only objective for both models. The report
records each tokenizer's sequence and supervised-token totals.

## See Also

- [Command reference](COMMANDS.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [`tests/` component guide](../tests/README.md)

## Focused regression coverage

Existing pipeline tests cover train/val/test outputs and integrity. Keep additions
small and in the relevant existing suite. `tests/model/test_rope_loading.py`
checks native checkpoint reload without caller-side repair; it requires the
training stack and does not establish the real Mini checkpoint's loss baseline.
No separate retention, test-contract, split-integration, or diagnostics suites
are required. Tool-call parsing/SFT contract checks live in the existing relevant
contract tests; full model and live-provider behavior require the actual runtime.
