# Curation

This guide takes a new data-processing host from infrastructure setup to an
uploaded, training-ready corpus.

## Required inputs

| Input | Purpose |
|---|---|
| `SIZE` | Data/model profile: `smoke`, `mini`, `125m`, `350m`, or `1b` |
| `WORKERS` | CPU workers reserved for parallel curation |
| `DATA_DIR` | Persistent storage root for datasets and artifacts |
| `.env` | AWS, Hugging Face, W&B, cache, results, and export configuration |

Fill the required common settings, including `HF_TOKEN`, `WANDB_API_KEY`, and
`WANDB_PROJECT`; W&B is required. Configure storage credentials when uploading
or restoring artifacts. `HF_USERNAME` and search-provider fields may stay blank
when model publication and web search are unused.

Before running curation, use the Hugging Face account associated with
`HF_TOKEN` to accept the terms for the gated sources in the active data mix:

- [`bigcode/the-stack-dedup`](https://huggingface.co/datasets/bigcode/the-stack-dedup)
- [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol)
- [`nvidia/Nemotron-CC-Math-v1`](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1)

Dataset access belongs to the account, not the token. Create `HF_TOKEN` from
the same account after accepting the terms.

## Prepare the host

Mount persistent storage before cloning the repository. For a secondary
volume, follow [`DISK_SETUP.md`](DISK_SETUP.md), then verify the mount:

```bash
df -h /data
```

Clone and configure the repository:

```bash
git clone https://github.com/tohio/slm.git
cd slm
cp .env.sample .env
vi .env
```

## Fresh-host workflow

Bootstrap a fresh CPU curation host explicitly before starting source work:

```bash
make setup-curate DATA_DIR=/data/slm/data

```

Setup also prepares and checks the required FastText and KenLM model assets. It
persists the selected `DATA_DIR` in `.env`; later Make commands use that setting
and the repository environment directly. No separate download or activation step
is required. See [infrastructure](../infra/README.md) for installer details.

Run smoke first:

```bash
make curate-smoke
make validate SIZE=smoke
make tokenizer SIZE=smoke
make tokenizer-test SIZE=smoke
make tokenize SIZE=smoke
```

Mini is optional, not a mandatory intermediate before a production dataset.
It requires at least 1.4B usable selected training tokens after tokenization.
To choose Mini, begin with:

```bash
make curate-mini
make validate SIZE=mini
```

Curation runtime varies with CPU count, network bandwidth, cache state, storage
throughput, and Common Crawl availability. Fixed wall-clock estimates are not
part of the operator contract.

For a complete production-size workflow after bootstrap and smoke validation:

```bash
make curate-all \
  SIZE=125m \
  WORKERS=62 \
  DATA_DIR=/data/slm/data \
  ARTIFACT_STAGES=validated,tokenized,tokenizer,metadata
```

`curate-all` performs the following sequence:

1. Validates `.env`, required Make inputs, and curation model prerequisites.
2. Verifies the pinned curation environment.
3. Curates, filters, deduplicates, and blends the configured sources.
4. Establishes test from training during blending without changing validation,
   audits all three overlap relationships, and validates all three splits.
5. Trains and validates the dataset-run-specific BPE tokenizer on train only.
6. Tokenizes and integrity-checks train, validation, and test binaries.
7. Runs each artifact gate without rebuilding completed stages.
8. Uploads only the stages selected by `ARTIFACT_STAGES` to the chosen backend;
   no retention profile or implicit train-file omission is applied.
9. Prints the `RUN_ID` required by the training host.

Choose `WORKERS` below the available CPU count. On a 64-vCPU host,
`WORKERS=62` is the standard starting point.

## Stage-by-stage workflow

Use the individual commands when inspecting a stage or resuming after an
interruption.

### Infrastructure

```bash
make check-env
make setup-curate DATA_DIR=/data/slm/data
.venv/bin/python infra/verify_environment.py --profile curation
```

Every curation execution target uses the same prerequisite gate. If FastText or
either KenLM model file is missing, curation stops before source processing and
prints the commands required to install the missing assets. The download targets
remain explicit; curation does not silently fetch models.

### Corpus construction

```bash
make curate SIZE=125m WORKERS=62
make test-curator SIZE=125m
```

`make curate-smoke` exercises curation only; follow it through validation and
tokenization for the bounded data-pipeline check. `make curate-mini` starts the
optional Mini curation run. Both write to
their own `$DATA_DIR/runs/<size>` namespace. Run smoke first on a new host.

### Validation

```bash
make validate SIZE=125m
make test-validate SIZE=125m
```

### Tokenizer and binary tokenization

```bash
make tokenizer SIZE=125m
make tokenizer-test SIZE=125m
make tokenize SIZE=125m
make test-data-pipeline SIZE=125m
```

### Artifact upload

```bash
make artifacts-upload \
  SIZE=125m \
  WORKERS=62 \
  ARTIFACT_STAGES="validated,tokenized,tokenizer,metadata"

cat "$DATA_DIR/runs/125m/RUN_ID"
```

Record the printed `RUN_ID`. The GPU host uses it to restore the exact
validated holdouts, all tokenized splits, tokenizer, and metadata.

## Resume behavior

Rerun the failed stage with the same `SIZE`, paths, worker count, and
configuration. Manifest-complete stages are reused only when their recorded
inputs, implementation, configuration, and outputs still match.

Do not use `FORCE=1` as a normal resume mechanism. It is reserved for a
specifically diagnosed stale or invalid stage.

Once test membership exists, blend reuse also compares the current deduplicated
input signatures, budget, mix, seed, and implementation with the saved blend
manifest. Changed inputs/policy stop the command; use a new data root for a new
corpus. The command never silently replaces an established test split. Restoring
or training from completed artifacts does not require raw/dedup intermediates.

## Outputs

The workflow writes size-scoped artifacts under:

```text
$DATA_DIR/runs/<size>/
├── raw/
├── filtered/
├── dedup_scratch/
├── curated/
├── validated/
├── tokenizer/
├── tokenized/
├── metadata/
└── RUN_ID
```

## Troubleshooting

For dataset access, storage, stage validation, resume, and artifact-upload
failures, see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## See Also

- [Training](TRAIN.md)
- [Testing](TESTING.md)
- [Command reference](COMMANDS.md)
- [Curation component guide](../curator/README.md)

## Split and transfer contracts

Normal blending produces train/val/test; validation and tokenization process all
three. There is no special regeneration command. The setup command selects the
curation requirements automatically. See [pretraining data](PRETRAINING_DATA.md)
for deterministic membership, the Mini token floor, HF Dataset/Bucket routing,
and matched artifact restoration. Upload selection remains yours.
