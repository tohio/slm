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
| CPU contracts | `make test-unit` | Installed CPU environment |
| GPU acceptance | `make test-gpu-gate` | Supported NVIDIA environment |
| Curator artifacts | `make test-curator SIZE=<size>` | Completed curated corpus |
| Validation artifacts | `make test-validate SIZE=<size>` | Completed validated corpus |
| Tokenizer behavior | `make tokenizer-test SIZE=<size>` | Trained tokenizer |
| Complete data pipeline | `make test-data-pipeline SIZE=<size>` | Curated, validated, tokenizer, and tokenized artifacts |
| Pretraining artifact | `make test-training SIZE=<size>` | Final base checkpoint |
| Instruct SFT artifact | `make test-sft-instruct SIZE=<size>` | Final instruct checkpoint |
| Code SFT artifact | `make test-sft-code SIZE=<size>` | Final code checkpoint |
| DPO artifact | `make test-dpo-chat SIZE=<size>` | Final chat checkpoint |
| Complete GPU pipeline | `make test-gpu-pipeline SIZE=<size>` | All final training checkpoints |

## CPU contracts

```bash
make test-unit
```

This gate covers architecture, configuration, data contracts, export,
training arguments, generated configurations, one-step synthetic SFT/DPO, and
repository consistency. Focused targets are listed in
[`COMMANDS.md`](COMMANDS.md).

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

Run the matching gate after each completed training stage:

```bash
make test-training SIZE=125m
make test-sft-instruct SIZE=125m
make test-sft-code SIZE=125m
make test-dpo-chat SIZE=125m
```

The Make targets require the requested artifacts. A missing final checkpoint
fails instead of silently skipping.

## Bounded pipeline rehearsal

Use the `mini` profile only when a code change requires an end-to-end pipeline
rehearsal and suitable artifacts do not already exist:

```bash
make curate-mini
make validate SIZE=mini
make tokenizer SIZE=mini
make tokenizer-test SIZE=mini
make tokenize SIZE=mini
make restore-size-tokenizer SIZE=mini
make pretrain-mini SIZE=mini GPUS=1
make prepare-sft SIZE=mini
make sft-instruct-mini SIZE=mini GPUS=1
make sft-code-mini SIZE=mini GPUS=1
make prepare-dpo SIZE=mini
make dpo-chat-mini SIZE=mini GPUS=1
make test-artifacts SIZE=mini
```

This validates execution and artifact contracts; it is not a quality
benchmark.

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
