# Curation

## Purpose

`curator/` builds the pretraining corpus. It loads configured sources, applies
source-aware quality filters, performs deterministic exact and fuzzy
deduplication, allocates source budgets, and produces shuffled train and
validation JSONL files. Post-curation perplexity validation, tokenizer
training, and binary tokenization are separate stages.

## Contents

```text
curator/
├── filters/              quality, PII, and deduplication logic
├── scripts/
│   ├── curate.py         curation orchestrator
│   ├── sample_source.py  inspect stage outputs
│   └── upload_s3.py      run-scoped artifact transfer
├── sources/              one loader per concrete data source
└── state.py              completion manifests and artifact fingerprints
```

## How It Fits In

The curator reads the data contract in `config/data_mix.py` and writes:

```text
$DATA_DIR/runs/<size>/curated/train.jsonl
$DATA_DIR/runs/<size>/curated/val.jsonl
```

`validation/` consumes those files. `pretrain/data/tokenize_data.py` later
measures the authoritative retained token counts; character-derived token
counts in curation reports are planning estimates.

## Data Contract

`config/data_mix.py` is the single source of truth for:

- top-level source percentages and the code sub-mix;
- fixed-supply and supplemental caps;
- deterministic cross-source deduplication priority;
- the FineWeb overflow sink;
- `mini`, 125M, 350M, and 1B corpus targets;
- validation split fraction and curation constants.

Print the active mix and corpus targets directly from the configuration:

```bash
python - <<'PY'
from config.data_mix import CODE_SUBMIX, DATA_MIX, TARGET_CONFIGS

for name, item in DATA_MIX.items():
    print(f"{name:32} {item['pct']:8.4f}%")

print("\nCode sub-mix")
for name, item in CODE_SUBMIX.items():
    print(f"{name:32} {item['pct']:8.4f}%")

print("\nCorpus targets")
for size, item in TARGET_CONFIGS.items():
    print(size, item["corpus_tokens"], "tokens", item["epochs"], "epoch(s)")
PY
```

The configured source families include broad and educational web text,
Wikipedia, books, academic papers, math and specialized technical corpora,
StackExchange, curated synthetic signals, and a five-source code sub-mix.
Exact dataset IDs and revisions belong in the configuration and source
loaders, not a duplicated README table.

## Pipeline Stages

```text
download
  source loader → $DATA_DIR/runs/<size>/raw/<source>/

filter
  raw source → $DATA_DIR/runs/<size>/filtered/<source>/

dedup
  filtered source → $DATA_DIR/runs/<size>/filtered/<source>_deduped/

blend
  all complete dedup outputs → curated/{train,val}.jsonl
```

Every reusable stage directory contains `_SUCCESS.json`. A stage is reused only
when its manifest version, implementation/configuration fingerprint, input
signature, and output signature match. File presence alone is not treated as
completion.

Raw `common_crawl` text receives the line-level checks from Datatrove's
`FineWebQualityFilter`: punctuated-line fraction, short-line fraction,
duplicate-line character fraction, and newline-to-word ratio. These web-corpus
metrics are not applied to already-curated FineWeb variants or to reference,
book, academic, code, math, and synthetic sources.

The Common Crawl extractor also runs Datatrove's integrated `URLFilter` before
Trafilatura. URLs matching its blocked domains, URLs, hard terms, soft-term
threshold, or banned substrings are discarded before HTML extraction.

Raw Common Crawl fuzzy deduplication uses FineWeb's 64-bit SHA-1, five-word,
14×8 MinHash configuration independently for each configured crawl snapshot.
Exact normalized deduplication still operates across snapshots and sources.

Exact deduplication runs across sources in `DEDUP_PRIORITY` order so the
preferred copy is retained. Fuzzy MinHash deduplication runs within each
eligible source. Template-like synthetic sources skip fuzzy matching but still
participate in exact deduplication.

Each filtered and deduplicated source manifest stores durable audit metadata.
Filter manifests record input, accepted, rejected, per-reason, and FastText
error counts. Dedup manifests record exact and fuzzy removals and the final
document count, including the Common Crawl partition contract.

Quality-filter routing is fail-closed. Every configured source belongs to one
explicit corpus family, and each record's `source` field must match the source
directory being processed. Adding an unclassified source or mixing source tags
in a shard stops filtering instead of silently selecting a different profile.

After blending and splitting, every train and validation record receives the
same normalized exact hash used by deduplication. Curation fails if validation
contains exact duplicates or if any normalized document occurs in both splits.
The full-corpus result is stored in `curated/exact_overlap_report.json` and in
`blend_stats.json`.

The same full-corpus pass checks both exact normalized benchmark inputs and
lm-eval's case- and punctuation-normalized 13-word benchmark n-grams for all
six configured evaluation benchmarks. Dataset commits, task definitions, and
the n-gram size are immutable configuration. Any match blocks blend completion
and is recorded without benchmark text in
`curated/benchmark_contamination_report.json`.

A separate disk-backed audit builds a validation MinHash index and checks every
training document with the same five-word, 14×8 configuration used for fuzzy
deduplication. It reports cross-split candidate clusters without deleting or
rewriting records, and blocks completion on any match. Results are stored in
`curated/near_overlap_report.json`.

The shared full-corpus pass also audits sensitive content. Service-shaped
credentials and private-key headers are marked for credential review. Email
addresses, international phone numbers, and SSN-shaped values are marked for
identifier review. Pattern matches do not establish that credentials are
active or identifiers are private, so this audit does not automatically block,
remove, or rewrite records. The durable report stores counts, locations, and
one-way match fingerprints—not matched values or surrounding text—in
`curated/sensitive_content_report.json`.

The blend stage:

- converts target token shares into character budgets;
- applies fixed-source and supplemental caps;
- routes source deficits to the configured overflow sink;
- stages every source to its effective budget;
- chooses in-memory or disk-backed shuffling from
  `SHUFFLE_RAM_BUDGET_GB`;
- creates the train/validation split from one shuffled population.

## Prerequisites

Set `DATA_DIR` to persistent storage and complete every variable in `.env`
before starting the pipeline.

```bash
make setup-data-dir DATA_DIR=/data/slm/data
source .venv/bin/activate
make download-fasttext-model DATA_DIR=/data/slm/data
```

`HF_TOKEN` is required for authenticated or gated Hub sources. Accept each
enabled dataset's terms before starting a full run:

- [`bigcode/the-stack-dedup`](https://huggingface.co/datasets/bigcode/the-stack-dedup)
  — primary source in the code sub-mix.
- [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol)
  — supplemental code source.
- [`nvidia/Nemotron-CC-Math-v1`](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1)
  — mathematical pretraining corpus.

Sign in with the Hugging Face account associated with `HF_TOKEN` before
accepting access.

## Usage

Bounded pipeline validation:

```bash
make curate-mini
```

Full target:

```bash
make curate SIZE=125m WORKERS=62
```

Run individual stages:

```bash
make curate-download SIZE=125m
make curate-filter SIZE=125m WORKERS=62
make curate-dedup SIZE=125m WORKERS=62
make curate-blend SIZE=125m WORKERS=62
```

Run selected sources through download, filter, dedup, and statistics:

```bash
python curator/scripts/curate.py \
  --target 125m \
  --stage all \
  --sources wikipedia,fineweb_edu \
  --workers 16
```

Source-scoped `--stage all` intentionally stops before blend because blending
requires every configured source.

Rebuild a raw source whose completion manifest does not match:

```bash
python curator/scripts/curate.py \
  --target 125m \
  --stage download \
  --sources wikipedia \
  --force
```

The new download is written to an isolated staging directory and promoted only
after completion.

## Inspection

Print readable records from any stage:

```bash
python curator/scripts/sample_source.py \
  --size 125m \
  --stage raw \
  --source wikipedia \
  --limit 10
```

`--stage` accepts `raw`, `filtered`, `deduped`, `curated`, or `validated`.
Use `--random` for deterministic random sampling and `--max-chars` to limit
record display length.

## Outputs

```text
$DATA_DIR/runs/<size>/
├── raw/<source>/
├── filtered/<source>/
├── filtered/<source>_deduped/
├── dedup_scratch/
└── curated/
    ├── train.jsonl
    ├── val.jsonl
    ├── blend_stats.json
    ├── exact_overlap_report.json
    ├── benchmark_contamination_report.json
    ├── near_overlap_report.json
    ├── sensitive_content_report.json
    └── _SUCCESS.json
```

`blend_stats.json` records document and character counts, target budgets,
source deficits, overflow contribution, and validation counts by source.

## Run-Scoped Artifact Transfer

The first artifact upload creates:

```text
$DATA_DIR/runs/<size>/RUN_ID
```

Artifact storage uses:

```text
s3://$S3_BUCKET/$S3_PREFIX/<size>/<run_id>/<stage>/
```

Upload curation outputs:

```bash
make curate-upload SIZE=125m
```

Upload or restore an explicit stage set:

```bash
make artifacts-upload \
  SIZE=125m \
  RUN_ID=125m-YYYYMMDD-abcdef \
  ARTIFACT_STAGES="curated,metadata"

make artifacts-download \
  SIZE=125m \
  RUN_ID=125m-YYYYMMDD-abcdef \
  ARTIFACT_STAGES="curated,metadata"
```

Downloads require an explicit `RUN_ID`. A restored stage is accepted only when
its artifact metadata matches the requested run and stage.

## Gotchas

- Full curation is storage- and network-intensive; use persistent storage and
  a session manager such as `tmux`.
- `WORKERS` controls filter, dedup, blend, and artifact-transfer parallelism.
- Do not delete `_SUCCESS.json` while expecting a stage to remain reusable.
- Do not change `config/data_mix.py` under an existing run ID.
- Run `make test-curator SIZE=<size>` after curation and before validation.
- See [`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md) for access,
  storage, resume, and artifact-transfer diagnostics.
