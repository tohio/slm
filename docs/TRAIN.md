# Training

This guide restores a curated artifact run on a new GPU host and trains the
base, instruct, code, and chat model variants.

## Required inputs

| Input | Purpose |
|---|---|
| `SIZE` | Model profile: `mini`, `125m`, `350m`, or `1b` |
| `GPUS` | Number of visible GPUs used by configuration and training |
| `RUN_ID` | Artifact run produced by the curation workflow |
| `DATA_DIR` | Persistent data and cache root on the GPU host |
| `.env` | AWS, Hugging Face, W&B, cache, results, and export configuration |

Every variable in `.env.sample` must have a real value in `.env`. The GPU host
must use an NVIDIA driver compatible with the repository's pinned CUDA 13
training stack.

## Prepare the host

Clone and configure the repository:

```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
vi .env
```

The `RUN_ID` comes from the final output of
[`curate-all`](CURATION.md#complete-workflow).

## Complete workflow

On a fresh GPU host:

```bash
make train-all \
  SIZE=125m \
  GPUS=1 \
  RUN_ID=125m-YYYYMMDD-abcdef \
  DATA_DIR=/data/slm/data
```

`train-all` performs the following sequence:

1. Validates every `.env` value and the required Make inputs.
2. Installs the GPU environment and pinned CUDA training dependencies.
3. Restores the tokenized corpus, tokenizer, and metadata identified by
   `RUN_ID`.
4. Runs the dataset-free CUDA, BF16, compile, and generation acceptance gate.
5. Generates hardware-specific pretraining, SFT, and DPO configurations.
6. Pretrains the base model.
7. Finalizes and validates the base checkpoint's chat/control-token
   embeddings.
8. Prepares, trains, and validates the instruct SFT branch.
9. Trains and validates the independent code SFT branch.
10. Prepares, trains, and validates the DPO chat branch.

The target is for a new run only. If a pretraining output directory already
exists, it stops and directs the operator to the stage-specific resume
commands.

Evaluation, export, publication, and serving remain explicit operations after
training.

## Stage-by-stage workflow

### Restore the curation run

```bash
make check-env
make setup-gpu \
  DATA_DIR=/data/slm/data \
  SIZE=125m \
  RUN_ID=125m-YYYYMMDD-abcdef

make test-gpu-gate
```

`setup-gpu` restores `tokenized`, `tokenizer`, and `metadata` artifacts and
activates the restored size-specific tokenizer.

### Generate training configurations

Generate recipes on the target GPU host:

```bash
make config-gen SIZE=125m GPUS=1
```

For an explicit hardware or VRAM policy:

```bash
make config-gen \
  SIZE=350m \
  GPUS=4 \
  GPU=h200 \
  MODE=conservative
```

Inspect the generated pretraining, SFT, and DPO YAML files before starting a
paid run.

### Pretrain the base model

Start a new run:

```bash
make pretrain SIZE=125m GPUS=1
```

Resume an interrupted run:

```bash
make pretrain-resume SIZE=125m GPUS=1
```

The promoted checkpoint is:

```text
$RESULTS_DIR/runs/<size>/pretrain/final/
```

### Finalize chat/control-token embeddings

Before SFT, finalize the chat-specific embedding rows:

```bash
make reinit-embeds SIZE=125m
make test-training SIZE=125m
```

This intentionally updates the promoted base checkpoint in place and creates
a timestamped backup first. Only the configured chat/control-token embedding
rows change. The rewritten `pretrain/final` is the canonical base used by
post-training.

### Train the instruct branch

```bash
make prepare-sft SIZE=125m
make sft-instruct SIZE=125m GPUS=1
make test-sft-instruct SIZE=125m
```

Resume:

```bash
make sft-instruct-resume SIZE=125m GPUS=1
```

The promoted checkpoint is:

```text
$RESULTS_DIR/runs/<size>/sft_instruct/final/
```

### Train the code branch

Code SFT starts from the completed instruct checkpoint:

```bash
make sft-code SIZE=125m GPUS=1
make test-sft-code SIZE=125m
```

Resume:

```bash
make sft-code-resume SIZE=125m GPUS=1
```

### Train the chat branch

DPO also starts from the completed instruct checkpoint:

```bash
make prepare-dpo SIZE=125m
make dpo-chat SIZE=125m GPUS=1
make test-dpo-chat SIZE=125m
```

Resume:

```bash
make dpo-chat-resume SIZE=125m GPUS=1
```

The code and chat branches are independent:

```text
pretrain/final
└── sft_instruct/final
    ├── sft_code/final
    └── dpo_chat/final
```

## Evaluate and export

Evaluate completed variants:

```bash
make eval-base SIZE=125m
make eval-instruct SIZE=125m
make eval-code SIZE=125m
make eval-chat SIZE=125m
```

Build and validate native local packages:

```bash
make export-local SIZE=125m
```

Publish all validated variants using the configured Hugging Face account:

```bash
make export SIZE=125m
```

## Troubleshooting

For CUDA, dependency, checkpoint, data-restore, and training-stage failures,
see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## See Also

- [Curation](CURATION.md)
- [Testing](TESTING.md)
- [Hardware guide](HARDWARE.md)
- [Command reference](COMMANDS.md)
- [Pretraining component guide](../pretrain/README.md)
- [SFT component guide](../finetune/README.md)
- [DPO component guide](../alignment/README.md)
