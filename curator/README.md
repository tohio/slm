# Curator

Data quality determines model quality more than any other factor. This stage processes raw, imperfect web data into a clean, tokenized dataset ready for pre-training. Rather than using pre-cleaned HuggingFace datasets, we run the full pipeline against Common Crawl — the same approach used in production LLM training — to exercise every curation component against realistic noise.

## Pipeline

Each stage is independently resumable. If a spot instance is preempted mid-run, restart with `--start-stage <stage>` and the pipeline picks up where it left off.

```
Common Crawl WARCs
      │
      ▼
 1. extract          HTML → clean text (trafilatura), encoding normalization
      │
      ▼
 2. language_filter  fastText language ID, retain English (score ≥ 0.65)
      │
      ▼
 3. heuristic_filter Rule-based quality signals (Gopher methodology)
      │
      ▼
 4. quality_filter   fastText classifier — Wikipedia-like vs web noise
      │
      ▼
 5. exact_dedup      MD5 hash deduplication — byte-identical documents
      │
      ▼
 6. fuzzy_dedup      MinHash + LSH — near-duplicate documents (Jaccard ≥ 0.8)
      │
      ▼
 7. pii              Regex-based redaction: emails, phones, IPs
      │
      ▼
 8. tokenize         SentencePiece BPE → memory-mapped .bin/.idx files
```

## Design Decisions

**Why Common Crawl instead of pre-cleaned datasets?**
Pre-cleaned sources (Wikipedia, OpenWebText) are easier but skip the hardest part of real data curation. Common Crawl contains encoding errors, multilingual content, SEO spam, duplicate articles across thousands of domains, and boilerplate-heavy pages. Running the full pipeline against this teaches you what each component is actually doing and why it exists.

**Why two deduplication passes?**
Exact dedup (MD5) is O(n) and catches byte-identical documents cheaply. Fuzzy dedup (MinHash LSH) catches near-duplicates — the same article republished across hundreds of domains with minor edits. Web data without fuzzy dedup contains enormous amounts of redundant content that wastes training compute and biases the model toward overrepresented text.

**Why heuristics before the classifier?**
Heuristic filters are fast and cheap — they eliminate the majority of garbage (too short, too many symbols, high repetition) before the slower fastText classifier runs. Ordering matters for throughput.

**Retention rates to expect**

| Stage | Typical retention |
|---|---|
| Language filter | 60–70% |
| Heuristic filter | 40–60% |
| Quality filter | 30–50% |
| Exact dedup | 85–95% |
| Fuzzy dedup | 70–85% |
| **Combined** | **~15–20% of raw** |

This is normal. A 20GB WARC file yielding 3–4GB of clean text is a good outcome.

## Usage

```bash
# Download Common Crawl subset (20 WARC files ~20GB compressed)
make download-data

# Run full pipeline
make curate

# Resume after preemption
python curator/pipelines/pipeline.py \
    --config curator/configs/curator.yaml \
    --start-stage fuzzy_dedup

# Train tokenizer on curated output
make tokenizer

# Upload to S3 for GPU instance
make upload-data S3_BUCKET=my-bucket
```

## Infrastructure

Curation runs on AWS Spot instances (CPU-bound workload). Spot instances offer 70–90% cost savings over on-demand — and since each stage checkpoints to S3, preemption only loses the current stage's progress.

Recommended instance types:

| Instance | vCPU | RAM | Best for |
|---|---|---|---|
| `c5.4xlarge` | 16 | 32GB | Extraction, filtering |
| `r5.4xlarge` | 16 | 128GB | MinHash deduplication |

Estimated cost for a full curation run (20 WARC files): **$3–5**.

## Output

The tokenization stage produces NeMo-compatible memory-mapped files:

```
/data/curated/stages/tokenize/
    text_document.bin     ← raw uint16 token IDs
    text_document.idx     ← document offsets for O(1) random access
```

These files are consumed directly by the pre-training stage with no further conversion.
