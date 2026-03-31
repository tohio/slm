# curator

Data curation pipeline for SLM pretraining. Downloads raw data from three sources, applies quality filters, deduplicates, blends to target token ratios, and uploads to S3.

---

## Pipeline

```
Wikipedia EN        ──┐
CodeSearchNet       ──┤──► quality filter ──► dedup ──┐
Common Crawl WARCs  ──┘                               ├──► blend ──► train.jsonl ──► S3
```

---

## Data Sources

| Source | Target % | Target tokens (125M) | Notes |
|---|---|---|---|
| Common Crawl | 70% | 2.1B | Broad web coverage, aggressive filtering needed |
| Wikipedia EN | 20% | 600M | High quality, factual, structured |
| CodeSearchNet | 10% | 300M | 6 languages, docstring + code pairs |

Token targets scale proportionally for larger models:

| Model | Total tokens | CC segments |
|---|---|---|
| `slm-125m` | 3B | 10 |
| `slm-350m` | 10B | 40 |
| `slm-1b` | 25B | 100 |

---

## Structure

```
curator/
├── sources/
│   ├── wikipedia.py         Wikipedia EN via HuggingFace datasets
│   ├── code_search_net.py   CodeSearchNet via HuggingFace datasets
│   └── common_crawl.py      Common Crawl WARCs via S3 + trafilatura
├── filters/
│   ├── quality.py           Heuristic quality filters (FineWeb/Gopher-style)
│   └── dedup.py             Exact + MinHash LSH deduplication
└── scripts/
    ├── curate.py            Main pipeline entry point
    └── upload_s3.py         S3 upload/download utilities
```

---

## Getting Started

**Prerequisites**

```bash
pip install -r requirements.txt
cp .env.sample .env
# Set S3_BUCKET, AWS credentials, DATA_DIR in .env
```

**Run the full pipeline**

```bash
# 125M dataset (~3B tokens)
python curator/scripts/curate.py --target 125m

# 350M dataset (~10B tokens)
python curator/scripts/curate.py --target 350m

# 1B dataset (~25B tokens)
python curator/scripts/curate.py --target 1b
```

**Run individual stages**

```bash
python curator/scripts/curate.py --target 125m --stage download
python curator/scripts/curate.py --target 125m --stage filter
python curator/scripts/curate.py --target 125m --stage dedup
python curator/scripts/curate.py --target 125m --stage blend
python curator/scripts/curate.py --target 125m --stage upload
```

**S3 utilities**

```bash
# Upload curated data to S3
python curator/scripts/upload_s3.py upload --src data/curated --dst curated

# Download curated data from S3
python curator/scripts/upload_s3.py download --src curated --dst data/curated

# List S3 contents
python curator/scripts/upload_s3.py list --prefix curated
```

---

## Data Directory Layout

```
data/
├── raw/
│   ├── wikipedia/              raw Wikipedia JSONL shards
│   ├── code/                   raw CodeSearchNet JSONL shards
│   └── common_crawl/           raw Common Crawl JSONL shards
├── filtered/
│   ├── wikipedia/              quality filtered
│   ├── wikipedia_deduped/      + deduplicated
│   ├── code/                   quality filtered
│   ├── code_deduped/           + deduplicated
│   ├── common_crawl/           quality filtered
│   └── common_crawl_deduped/   + deduplicated
└── curated/
    ├── train.jsonl             final blended dataset
    └── blend_stats.json        source mix breakdown
```

---

## Quality Filters

Heuristics adapted from FineWeb and Gopher applied to all sources:

| Filter | Threshold | Catches |
|---|---|---|
| Min length | 200 chars | Stubs, empty pages |
| Max length | 100k chars | Extremely long documents |
| Mean word length | 3–10 chars | Gibberish, SEO spam |
| Symbol ratio | < 10% symbols/words | Symbol-heavy spam |
| Bullet ratio | < 90% bullet lines | Pure list content |
| Ellipsis ratio | < 30% ellipsis lines | Truncated content |
| Alpha ratio | > 70% alpha chars | Numeric/code spam (skipped for code source) |
| Repeated lines | < 30% duplicates | Boilerplate, repeated content |
| Stop words | ≥ 2 EN stop words | Non-English content (skipped for code source) |

---

## Deduplication

Two-stage deduplication applied after quality filtering:

- **Exact dedup** — SHA-256 hash of normalized text. Zero false positives. Catches verbatim copies.
- **Fuzzy dedup** — MinHash LSH with Jaccard threshold 0.8. Catches near-duplicates (same article, minor edits). 128 hash permutations.

The dedup index is persisted across shards and sources — a document seen in Wikipedia will not appear again in Common Crawl.

---

## Output Format

Each record in the final `train.jsonl`:

```json
{
  "text": "...",
  "source": "wikipedia | code | common_crawl",
  "title": "...",        // wikipedia only
  "url": "...",          // wikipedia + common_crawl
  "language": "python",  // code only
  "crawl": "CC-MAIN-2024-10"  // common_crawl only
}
```

---

## Key Design Decisions

**Why these three sources?** Wikipedia provides factual, high-quality general text. CodeSearchNet adds coding capability from the start — without it the base model has poor code understanding, making code SFT harder. Common Crawl provides scale and diversity that neither Wikipedia nor CodeSearchNet can match alone.

**Why 70/20/10?** Following empirical findings from The Pile, RedPajama, and FineWeb — web data dominates for scale, Wikipedia provides quality signal, and 10% code is sufficient for the base model before code SFT.

**Why trafilatura over BeautifulSoup?** trafilatura is specifically designed for main content extraction from web pages. It handles boilerplate removal (navigation, ads, footers) significantly better than generic HTML parsers.

**Why MinHash LSH at 0.8 threshold?** The 0.8 Jaccard threshold is the standard used by FineWeb and most production pipelines. Lower thresholds remove too much valid content, higher thresholds miss near-duplicates. 128 permutations gives a good accuracy/memory tradeoff.

**Why per-stage resumability?** Curation runs take hours to days. Each stage checks for existing output before processing — safe to interrupt and restart at any stage without reprocessing completed work.

---

## Infrastructure

Recommended hardware for the 125M dataset:

| Stage | Hardware | Est. time |
|---|---|---|
| Download (Wikipedia + CodeSearchNet) | CPU instance | ~30 min |
| Download (Common Crawl, 10 segments) | CPU instance, high bandwidth | ~2 hrs |
| Filter | CPU instance, 4+ cores | ~1 hr |
| Dedup | CPU instance, 16GB+ RAM | ~2 hrs |
| Blend + upload | CPU instance | ~30 min |

For the 1B dataset (100 CC segments), Common Crawl download and dedup times scale linearly.