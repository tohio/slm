# curator

Data curation pipeline for SLM pretraining. Downloads raw data from three sources, applies quality filters, deduplicates, blends to target token ratios, and uploads to S3.

---

## Pipeline

```
Wikipedia EN        ──┐
CodeSearchNet       ──┤──► quality filter ──► dedup ──► blend ──► train.jsonl ──► S3
Common Crawl WARCs  ──┘
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
| `mini` | 1M | 2 |
| `slm-125m` | 3B | 350 |
| `slm-350m` | 10B | 1,200 (600 × 2 crawls) |
| `slm-1b` | 25B | 3,000 (1,000 × 3 crawls) |

---

## Structure

```
curator/
├── sources/
│   ├── wikipedia.py         Wikipedia EN via HuggingFace datasets
│   ├── code_search_net.py   CodeSearchNet via HuggingFace datasets
│   └── common_crawl.py      Common Crawl WARCs via HTTPS + trafilatura
├── filters/
│   ├── quality.py           Heuristic quality filters (FineWeb/Gopher-style)
│   └── dedup.py             Exact + datatrove disk-based MinHash deduplication
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

# Download the fasttext language ID model (~1MB) — required before first run
make download-fasttext-model
```

**Minimal run — validate the pipeline before committing to a full run**

```bash
python curator/scripts/curate.py --target mini --mini
```

This caps Wikipedia at 5k docs, CodeSearchNet at 10k samples (Python + JS only),
and Common Crawl at 2 WARC segments (~70k raw docs). Total runtime ~30–45 min.
Run each stage individually to inspect output between steps:

```bash
python curator/scripts/curate.py --target mini --mini --stage download
python curator/scripts/curate.py --target mini --mini --stage filter
python curator/scripts/curate.py --target mini --mini --stage dedup
python curator/scripts/curate.py --target mini --mini --stage blend
```

**Full pipeline**

```bash
# 125M dataset (~3B tokens)
python curator/scripts/curate.py --target 125m

# 350M dataset (~10B tokens) with 16 parallel workers
python curator/scripts/curate.py --target 350m --workers 16

# 1B dataset (~25B tokens) with 32 parallel workers
python curator/scripts/curate.py --target 1b --workers 32
```

**Individual stages**

```bash
python curator/scripts/curate.py --target 125m --stage download
python curator/scripts/curate.py --target 125m --stage filter
python curator/scripts/curate.py --target 125m --stage dedup --workers 8
python curator/scripts/curate.py --target 125m --stage blend
python curator/scripts/curate.py --target 125m --stage upload
```

**S3 upload**

Use `make curate-upload` or the stage directly — the upload automatically creates a
versioned path by target and date:

```bash
# Recommended — versioned upload
make curate-upload SIZE=125m

# Or directly
python curator/scripts/curate.py --target 125m --stage upload
```

For manual S3 operations (listing, downloading specific runs):

```bash
# List all uploads
python curator/scripts/upload_s3.py list

# Download a specific run
python curator/scripts/upload_s3.py download \
    --src 125m/2026-04-02/curated \
    --dst data/curated
```

---

## Data Directory Layout

```
data/
├── raw/
│   ├── wikipedia/              raw Wikipedia JSONL shards
│   ├── code/                   raw CodeSearchNet JSONL shards
│   └── common_crawl/           raw Common Crawl JSONL shards
│       └── cc_progress.json    segment-level resume tracker (delete to re-download)
├── filtered/
│   ├── wikipedia/              quality filtered
│   ├── wikipedia_deduped/      + deduplicated
│   ├── code/                   quality filtered
│   ├── code_deduped/           + deduplicated
│   ├── common_crawl/           quality filtered
│   └── common_crawl_deduped/   + deduplicated
├── dedup_scratch/              datatrove intermediate state (safe to delete after dedup)
│   ├── wikipedia/
│   │   ├── exact_deduped/      intermediate exact-dedup shards
│   │   ├── signatures/         MinHash signatures
│   │   ├── buckets/            LSH bucket pairs
│   │   ├── clusters/           duplicate cluster assignments
│   │   └── logs/               datatrove stage logs
│   ├── code/
│   └── common_crawl/
└── curated/
    ├── train.jsonl             final blended dataset
    └── blend_stats.json        source mix breakdown
```

---

## S3 Structure

Each upload is versioned by target and date, so multiple runs never overwrite each other:

```
s3://your-bucket/slm/data/
├── 125m/
│   ├── 2026-04-02/
│   │   ├── curated/train.jsonl
│   │   └── curated/blend_stats.json
│   └── 2026-04-15/
│       ├── curated/train.jsonl
│       └── curated/blend_stats.json
├── 350m/
│   └── 2026-04-20/
│       ├── curated/train.jsonl
│       └── curated/blend_stats.json
└── mini/
    └── 2026-04-01/
        ├── curated/train.jsonl
        └── curated/blend_stats.json
```

Re-uploading on the same day overwrites that day's run. Runs on different days
are preserved independently.

---

## Quality Filters

Heuristics adapted from FineWeb and Gopher. Filters marked ✗ are skipped for the
code source — symbol-heavy syntax, long identifiers, and absence of stop words are
normal properties of code, not quality signals.

| Filter | Threshold | Catches | Skipped for code |
|---|---|---|---|
| Min length | 200 chars | Stubs, empty pages | |
| Max length | 100k chars | Extremely long documents | |
| Mean word length | 3–10 chars | Gibberish, SEO spam | ✗ |
| Symbol ratio | < 10% symbols/words | Symbol-heavy spam | ✗ |
| Bullet ratio | < 90% bullet lines | Pure list content | |
| Ellipsis ratio | < 30% ellipsis lines | Truncated content | |
| Alpha ratio | > 70% alpha chars | Numeric/code spam | ✗ |
| Repeated lines | < 30% duplicates | Boilerplate, repeated content | |
| Stop words | ≥ 2 EN stop words | Non-English content | ✗ |

---

## Deduplication

Two-stage deduplication applied after quality filtering:

**Stage 1 — Exact dedup**
SHA-256 hash of normalized text. Zero false positives. Catches verbatim duplicates. Shared across all sources — a Wikipedia article that also appears in Common Crawl is caught. The only in-memory structure; grows ~70 bytes/doc (~560MB at 125m, ~2GB at 350m, ~5GB at 1b).

**Stage 2 — Fuzzy dedup (datatrove)**
4-stage disk-based MinHash LSH pipeline. Catches near-duplicates (Jaccard similarity > 0.8). Peak RAM is bounded by shard size, not corpus size — 125m, 350m, and 1b all run with the same memory footprint.

```
signatures  →  buckets  →  cluster  →  filter
(per-doc       (LSH          (union-     (stream +
 minhash)       grouping)     find)       drop dupes)
```

Intermediate state written to `data/dedup_scratch/` and safe to delete after the dedup stage completes.

---

## Blend

Streaming reservoir sampling — no source is loaded into memory in full:

1. Stream each source to a per-source staging file, stopping when the character target is hit
2. Merge staging files into a single file (interleaved)
3. Shuffle using a byte-offset index — builds an index of line start positions, shuffles the index, reads lines in shuffled order

Peak RAM during blend is the offset index (~8 bytes/line). At 100M documents that is ~800MB.

---

## Output Format

Each record in the final `train.jsonl`:

```json
{
  "text": "...",
  "source": "wikipedia | code | common_crawl",
  "title": "...",               // wikipedia only
  "url": "...",                 // wikipedia + common_crawl
  "language": "python",         // code only
  "crawl": "CC-MAIN-2024-10"   // common_crawl only
}
```

---

## Infrastructure

Recommended hardware for each model size:

| Target | Instance | RAM | Est. curation runtime |
|---|---|---|---|
| `mini` | Any | 4GB+ | ~30–45 min |
| `125m` | `c8g.4xlarge` (16 vCPU) | 32GB | ~12–16 hrs |
| `350m` | `c8g.4xlarge` (16 vCPU) | 32GB | ~40–50 hrs |
| `1b` | `c8g.8xlarge` (32 vCPU) | 64GB | ~96–120 hrs |

All runs on AWS spot in `us-east-1` to minimize Common Crawl egress latency. Attach an EBS volume (`gp3`, 500GB) for `data/` so it survives spot interruptions — the pipeline is fully resumable at every stage.

**Parallelism:** The `--workers` flag controls the number of parallel workers in the dedup stage. Set it to the number of available CPU cores. Filter and download are currently single-threaded per shard.

**Spot interruption:** The Common Crawl download stage tracks progress at the segment level in `data/raw/common_crawl/cc_progress.json`. Each fully completed WARC segment is recorded there — a spot interruption loses at most one segment of work (~35k docs). Restart the exact same command and it resumes from the next unprocessed segment. Filter, dedup, and blend stages skip shards that already exist on disk.

---

## Key Design Decisions

**Why these three sources?** Wikipedia provides factual, high-quality general text. CodeSearchNet adds coding capability from the start — without it the base model has poor code understanding, making code SFT harder. Common Crawl provides scale and diversity that neither Wikipedia nor CodeSearchNet can match alone.

**Why 70/20/10?** Following empirical findings from The Pile, RedPajama, and FineWeb — web data dominates for scale, Wikipedia provides quality signal, and 10% code is sufficient for the base model before code SFT.

**Why trafilatura over BeautifulSoup?** trafilatura is specifically designed for main content extraction from web pages. It handles boilerplate removal (navigation, ads, footers) significantly better than generic HTML parsers.

**Why HTTPS for Common Crawl instead of S3?** Direct S3 access to the `commoncrawl` bucket fails on EC2 instances with IAM roles attached — the instance role credentials are rejected by the bucket policy. HTTPS via `data.commoncrawl.org` works reliably regardless of instance credentials.

**Why datatrove for dedup instead of datasketch?** datasketch's `MinHashLSH` is an in-memory data structure. At 350m scale the index requires ~32GB RAM; at 1b it requires ~85GB and cannot fit on a single instance. datatrove's disk-based pipeline uses a sort-based approach (signatures → buckets → cluster → filter) where RAM usage is bounded by shard size, not corpus size. This is the same approach used by FineWeb and RedPajama at trillion-token scale.

**Why streaming blend?** The original implementation loaded all deduped records into RAM before sampling — at 350m scale this requires ~90GB and OOMs on any standard instance. The streaming approach hits the same token targets with constant memory by sampling per shard and shuffling via a byte-offset index.

**Why versioned S3 uploads?** Each run is uploaded to `{target}/{date}/curated/` so multiple runs never overwrite each other. This makes it safe to re-run curation with different parameters or data sources and compare results, and allows rolling back to a previous run if issues are found during training.

**Why per-stage resumability?** Curation runs take hours to days on spot instances that can be interrupted with 2 minutes notice. Each stage checks for existing output before processing — safe to interrupt and restart without reprocessing completed work.

**Why fasttext for language detection instead of langdetect?** Language detection runs on every document in the Common Crawl pipeline — tens of millions of HTML pages at 125m scale. `langdetect` is pure Python and adds ~5–10ms per document, compounding to 50–100+ hours of wall time on a single instance. fasttext's language identification model (`lid.176.ftz`) is C-backed, covers 176 languages, and runs ~1000x faster with equivalent accuracy. The model file is ~1MB and downloaded once via `make download-fasttext-model`.