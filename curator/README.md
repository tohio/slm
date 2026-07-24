# curator

Data curation for SLM pretraining. This folder owns source loading, filtering, deduplication, blending, sampling, and RUN_ID artifact transfer.

---

## Owns

- source loaders in `curator/sources/`
- heuristic quality filters in `curator/filters/quality.py`
- exact and fuzzy dedup in `curator/filters/dedup.py`
- curation orchestration in `curator/scripts/curate.py`
- source sampling in `curator/scripts/sample_source.py`
- artifact upload/download in `curator/scripts/upload_s3.py`

Validation, tokenizer training, tokenization, and model training live in other folders.

---

## Key files

```text
curator/
├── state.py
├── filters/
│   ├── quality.py
│   └── dedup.py
├── scripts/
│   ├── curate.py
│   ├── sample_source.py
│   └── upload_s3.py
└── sources/
    └── hf.py
```

Shared curation settings live in `config/data_mix.py`. Do not duplicate source percentages, token targets, `CHARS_PER_TOKEN`, or Common Crawl crawl settings in curator code.

---

## Data flow

```text
source loaders
  ↓
quality filters
  ↓
dedup
  ↓
blend + train/val split
  ↓
data/runs/<size>/curated/train.jsonl
data/runs/<size>/curated/val.jsonl
data/runs/<size>/curated/blend_stats.json
```

The blend stage samples validation data from the same shuffled distribution as train.
Every reusable stage directory contains `_SUCCESS.json`. A stage is reused only
when this manifest matches its inputs, configuration, and output snapshot.
RUN_ID uploads verify these manifests first and mirror each selected stage, so
obsolete remote shards cannot survive a replacement upload.

---

## Source mix

`config/data_mix.py` is the source of truth.

Top-level mix:

| Source | Share |
|---|---:|
| Common Crawl | 5.0% |
| FineWeb | 10.0% |
| FineWeb-Edu | 31.5% |
| Wikipedia | 10.0% |
| PG-19 | 2.5% |
| Common Pile peS2o filtered | 5.0% |
| Nemotron CC Math | 7.0% |
| StackExchange | 1.0% |
| Synthetic arithmetic | 0.1475% |
| Synthetic task code | 0.3934% |
| Educational QA/MCQ math | 0.1475% |
| Educational QA/MCQ general | 0.2459% |
| Factual restraint | 0.0657% |
| Nemotron Specialized | 12.0% |
| Code bucket | 15.0% |

Code bucket split:

| Code source | Share of code bucket |
|---|---:|
| The Stack v1 dedup | 83.0% |
| CodeSearchNet | 15.0% |
| The Stack smol | 1.0% |
| Jupyter parsed | 0.5% |
| CoNaLa | 0.5% |

Synthetic/generated supplements have source-specific caps in `config/data_mix.py`. Underfilled synthetic sources overflow first to Nemotron Specialized, then FineWeb-Edu, then FineWeb.

---

## Token targets

| Size | Curation target | Epochs | Consumed target |
|---|---:|---:|---:|
| `mini` | 1M | 1 | 1M |
| `125m` | 10B | 2 | 20B |
| `350m` | 25B | 2 | 50B |
| `1b` | 75B | 1 | 75B |

---

## Commands

Mini run:

```bash
make curate-mini
```

Full curation:

```bash
make curate SIZE=125m WORKERS=62
```

Stage-specific runs:

```bash
make curate-download SIZE=125m
make curate-filter SIZE=125m WORKERS=62
make curate-dedup SIZE=125m WORKERS=62
make curate-blend SIZE=125m
```

Direct calls:

```bash
python curator/scripts/curate.py --target 125m
python curator/scripts/curate.py --target mini --mini
python curator/scripts/curate.py --target 125m --stage download
python curator/scripts/curate.py --target 125m --sources wikipedia,fineweb_edu
```

Hugging Face sources are resolved to immutable dataset commit SHAs before
loading. `--force` is required to replace legacy or stale raw directories;
replacement happens only after a new isolated download completes.

```bash
make curate-download SIZE=125m FORCE=1
# equivalent direct call:
python curator/scripts/curate.py --target 125m --stage download --force
```

---

## Sampling

Use `sample_source.py` to inspect actual records written by a source/stage.

```bash
python curator/scripts/sample_source.py --size 125m --stage raw --source wikipedia --limit 10
python curator/scripts/sample_source.py --size 125m --stage filtered --source wikipedia --limit 10
python curator/scripts/sample_source.py --size 125m --stage deduped --source wikipedia --limit 10
python curator/scripts/sample_source.py --size 125m --stage curated --source wikipedia --limit 10
python curator/scripts/sample_source.py --size 125m --stage validated --source wikipedia --limit 10
```

Valid sample stages:

```text
raw, filtered, deduped, curated, validated
```

---

## RUN_ID artifacts

Artifacts are stored by `RUN_ID`, not date.

Local layout:

```text
data/runs/<size>/RUN_ID
data/runs/<size>/<stage>/
```

S3 layout:

```text
<S3_PREFIX>/<size>/<run_id>/<stage>/
```

Valid artifact stages:

```text
raw, curated, validated, tokenized, tokenizer, metadata
```

Upload:

```bash
make artifacts-upload SIZE=125m
make artifacts-upload SIZE=125m RUN_ID=125m-20260629-a8f3c9
```

Download:

```bash
make artifacts-download SIZE=125m RUN_ID=125m-20260629-a8f3c9
```

---

## Outputs

```text
data/runs/<size>/raw/
data/runs/<size>/filtered/
data/runs/<size>/dedup_scratch/
data/runs/<size>/curated/train.jsonl
data/runs/<size>/curated/val.jsonl
data/runs/<size>/curated/blend_stats.json
```

`blend_stats.json` records intended character budgets, initial source
shortfalls, and unresolved shortfalls. Character-derived token totals are
explicitly estimates. Authoritative realized token and source counts are in
`tokenized/train.json` and `tokenized/val.json`.

---

## Notes

- Run long curation jobs inside `tmux`.
- Use persistent storage for `DATA_DIR`.
- Use `WORKERS` to control parallel curation stages.
- Generated/template-like sources bypass fuzzy MinHash dedup but still run exact dedup.
- Exact cross-source dedup retains cleaner/reference sources before broad web
  sources. Fuzzy MinHash runs within each source and is an LSH probability
  contract, not a strict Jaccard threshold.
- Restore artifacts by `RUN_ID`, not date.
