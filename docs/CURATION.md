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

Every variable in `.env.sample` must have a real value in `.env`; blank values
and `...` placeholders are rejected.

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
make setup-data-dir DATA_DIR=/data/slm/data
source .venv/bin/activate

make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model    DATA_DIR=/data/slm/data
make check-curation-prereqs  DATA_DIR=/data/slm/data
```

Run smoke first:

```bash
make curate-smoke DATA_DIR=/data/slm/data
make validate SIZE=smoke DATA_DIR=/data/slm/data
```

If smoke passes, run the functional mini-scale curation:

```bash
make curate-mini DATA_DIR=/data/slm/data
make validate SIZE=mini DATA_DIR=/data/slm/data
```

Curation runtime varies with CPU count, network bandwidth, cache state, storage
throughput, and Common Crawl availability. Fixed wall-clock estimates are not
part of the operator contract.

For a complete production-size workflow after bootstrap and smoke validation:

```bash
make curate-all \
  SIZE=125m \
  WORKERS=62 \
  DATA_DIR=/data/slm/data
```

`curate-all` performs the following sequence:

1. Validates `.env`, required Make inputs, and curation model prerequisites.
2. Verifies the pinned curation environment.
3. Curates, filters, deduplicates, and blends the configured sources.
4. Validates the curated train and validation splits.
5. Trains and validates the size-specific BPE tokenizer.
6. Tokenizes both splits into memory-mapped binaries.
7. Runs each artifact gate without rebuilding completed stages.
8. Uploads `tokenized`, `tokenizer`, and `metadata` artifacts to S3.
9. Prints the `RUN_ID` required by the training host.

Choose `WORKERS` below the available CPU count. On a 64-vCPU host,
`WORKERS=62` is the standard starting point.

## Stage-by-stage workflow

Use the individual commands when inspecting a stage or resuming after an
interruption.

### Infrastructure

```bash
make check-env
make setup-data-dir DATA_DIR=/data/slm/data
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model DATA_DIR=/data/slm/data
make check-curation-prereqs DATA_DIR=/data/slm/data
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

Use `make curate-smoke` for the bounded pipeline-validation run. Use
`make curate-mini` for the functional mini-scale curation run. Both write to
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
  ARTIFACT_STAGES="tokenized,tokenizer,metadata"

cat "$DATA_DIR/runs/125m/RUN_ID"
```

Record the printed `RUN_ID`. The GPU host uses it to restore the exact
tokenized corpus, tokenizer, and metadata.

## Resume behavior

Rerun the failed stage with the same `SIZE`, paths, worker count, and
configuration. Manifest-complete stages are reused only when their recorded
inputs, implementation, configuration, and outputs still match.

Do not use `FORCE=1` as a normal resume mechanism. It is reserved for a
specifically diagnosed stale or invalid stage.

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
