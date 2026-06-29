# curator

Data curation pipeline for SLM pretraining. This folder downloads source datasets, filters documents, deduplicates, blends train/val splits, and prepares reusable artifacts.

---

## Responsibility

`curator/` owns:

- source loaders
- quality filtering
- deduplication
- source blending
- train/val split creation
- artifact upload/download through `upload_s3.py`
- source sampling for manual inspection

Validation, tokenizer training, and pretraining live in other folders.

---

## Data flow

```text
source loaders
  ↓
quality filters
  ↓
dedup
  ↓
blend
  ↓
data/runs/<size>/curated/train.jsonl
data/runs/<size>/curated/val.jsonl
data/runs/<size>/metadata/blend_stats.json
```

---

## Source mix

The source mix is defined in `config/data_mix.py`.

| Source | Target Share | Notes |
|---|---:|---|
| Common Crawl | 5% | direct WARC via trafilatura |
| FineWeb | 10% | broad web text |
| FineWeb-Edu | 31.5% | educational/explanatory web text |
| Wikipedia | 10% | encyclopedia text |
| pg19 | 2.5% | public-domain books |
| peS2o | 5% | academic/scientific prose |
| Nemotron CC Math | 7% | math/STEM text |
| StackExchange | 1% | Q&A-style web text |
| Synthetic arithmetic | 0.1475% | arithmetic signal |
| Synthetic task code | 0.3934% | task-shaped code examples |
| Educational QA/MCQ math | 0.1475% | math MCQ examples |
| Educational QA/MCQ general | 0.2459% | general MCQ examples |
| Factual restraint | 0.0657% | uncertainty/restraint examples |
| Nemotron Specialized | 12% | specialized supplement |
| Code total | 15% | split across code sub-sources |

---

## Token targets

| Size | Curation target | Epochs | Consumed target |
|---|---:|---:|---:|
| `125m` | 10B | 2 | 20B |
| `350m` | 25B | 2 | 50B |
| `1b` | 75B | 1 | 75B |

---

## Files

```text
curator/
├── constants.py
├── filters/
│   ├── quality.py
│   └── dedup.py
├── scripts/
│   ├── curate.py
│   ├── sample_source.py
│   └── upload_s3.py
└── sources/
```

---

## Commands

Mini curation:

```bash
make curate-mini
make test-curator
```

Full curation:

```bash
make curate SIZE=125m WORKERS=62
```

Stage-specific runs:

```bash
make curate-download SIZE=125m
make curate-filter   SIZE=125m WORKERS=62
make curate-dedup    SIZE=125m WORKERS=62
make curate-blend    SIZE=125m
```

Sample source records:

```bash
python curator/scripts/sample_source.py --source fineweb_edu --target mini --limit 5
```

---

## RUN_ID artifacts

Artifacts are stored by run ID:

```text
data/runs/<size>/RUN_ID
data/runs/<size>/raw/
data/runs/<size>/curated/
data/runs/<size>/validated/
data/runs/<size>/tokenized/
data/runs/<size>/tokenizer/
data/runs/<size>/metadata/
```

S3 layout:

```text
<S3_PREFIX>/<size>/<run_id>/<stage>/
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

Valid stages:

```text
raw, curated, validated, tokenized, tokenizer, metadata
```

---

## Quality filters

The quality filter stage removes documents with:

- extreme length
- low alphabetic content
- high symbol or boilerplate ratio
- repeated lines
- malformed text patterns
- obvious non-prose content

Validation adds language/perplexity checks after curation.

---

## Deduplication

The curator uses exact and fuzzy dedup depending on source type. Generated/template-like sources may use exact dedup only so useful repeated task structure is not collapsed.

---

## Blend output

The blend stage writes shuffled train/val JSONL and realized mix metadata.

```text
data/runs/<size>/curated/train.jsonl
data/runs/<size>/curated/val.jsonl
data/runs/<size>/metadata/blend_stats.json
```

`blend_stats.json` is the source of truth for realized mix in exported model cards.

---

## Operational notes

- Run long curation jobs inside `tmux`.
- Use a persistent disk for `DATA_DIR`.
- Use `WORKERS` to control curation parallelism.
- Restore by `RUN_ID`, not date.
