# Curator

Data quality determines model quality more than any other factor. This stage processes raw, imperfect web data into a clean, tokenized dataset ready for pre-training. Rather than using pre-cleaned HuggingFace datasets, we run the full pipeline against Common Crawl — the same approach used in production LLM training — to exercise every curation component against realistic noise.

## Pipeline

Each stage is independently resumable. If an instance is preempted mid-run, restart with `--start-stage <stage>` and the pipeline picks up where it left off using `.complete` markers.

```
Common Crawl WARCs
      │
      ▼
 1. extract          HTML → clean text (trafilatura), encoding normalization
      │              Parallelized at record level across all available CPUs
      ▼
 2. language_filter  fastText language ID, retain English (score ≥ 0.65)
      │
      ▼
 3. heuristic_filter Rule-based quality signals (Gopher methodology)
      │
      ▼
 4. quality_filter   fastText classifier — Wikipedia-like vs web noise
      │              Auto-skipped if model not trained yet (pass 1)
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
                     Auto-skipped if tokenizer not trained yet (pass 1)
```

## Two-Pass Curation

The pipeline runs in two passes because the quality classifier and tokenizer depend on curated output that doesn't exist yet on the first run.

**Pass 1** — `quality_filter` and `tokenize` are auto-skipped (models not present):
```bash
make docker-curate    # extract → language_filter → heuristic_filter →
                      # exact_dedup → fuzzy_dedup → pii
make tokenizer        # train tokenizer on pass 1 output
```

**Pass 2** — both stages run automatically once models exist:
```bash
# train quality classifier (see .todo for automation)
make docker-curate    # pipeline.py detects models at runtime, runs all stages
```

No manual edits to `curator.yaml` required between passes — `pipeline.py` checks for model files at startup and logs `WILL RUN` or `WILL SKIP` for each optional stage.

## Design Decisions

**Why Common Crawl instead of pre-cleaned datasets?**
Pre-cleaned sources (Wikipedia, OpenWebText) are easier but skip the hardest part of real data curation. Common Crawl contains encoding errors, multilingual content, SEO spam, duplicate articles across thousands of domains, and boilerplate-heavy pages. Running the full pipeline against this teaches you what each component is actually doing and why it exists.

**Why two deduplication passes?**
Exact dedup (MD5) is O(n) and catches byte-identical documents cheaply. Fuzzy dedup (MinHash LSH) catches near-duplicates — the same article republished across hundreds of domains with minor edits. Web data without fuzzy dedup contains enormous amounts of redundant content that wastes training compute and biases the model toward overrepresented text.

**Why heuristics before the classifier?**
Heuristic filters are fast and cheap — they eliminate the majority of garbage (too short, too many symbols, high repetition) before the slower fastText classifier runs. Ordering matters for throughput.

**Why process WARCs one at a time during extraction?**
Loading all WARC records into memory simultaneously (~11GB for 20 WARCs) exhausts the Dask scheduler during task graph submission. Processing one WARC at a time (~600MB peak) keeps memory bounded while still utilizing all CPU cores via Dask parallelism within each WARC.

**Retention rates (observed on CC-MAIN-2024-10)**

| Stage | Typical | Observed (20 WARCs) |
|---|---|---|
| Language filter | 60–70% | ~22% |
| Heuristic filter | 40–60% | — |
| Quality filter | 30–50% | skipped (pass 1) |
| Exact dedup | 85–95% | ~99% |
| Fuzzy dedup | 70–85% | ~98.7% |
| **Combined** | **~15–20% of raw** | — |

> Language filter retention of 22% is normal for Common Crawl — the raw crawl is heavily non-English. Overall combined retention of 15–20% of raw records is expected.

## Infrastructure

Curation runs on CPU instances. Dask worker count and memory limits are detected automatically from available CPUs and RAM — no instance-specific config required.

All commands run inside the Docker container via `make docker-curate`. The `/data` directory is bind-mounted from the host so output persists across container restarts.

## Usage

```bash
# First time setup
make init-dirs
make docker-build
make download-models        # fastText language ID model (lid.176.bin)

# Download Common Crawl WARCs
make download-data N_WARC_FILES=20

# Run curation pipeline (pass 1)
make docker-curate

# Train tokenizer on curated output
make tokenizer

# Upload to S3 for GPU instance
make upload-data S3_BUCKET=my-bucket

# Resume after interruption
make docker-shell-cpu
# inside container:
python curator/pipelines/pipeline.py \
    --config curator/configs/curator.yaml \
    --start-stage fuzzy_dedup
```

## Output

The curation pipeline writes to `/data/curated/stages/<stage>/` at each step. The tokenization stage produces NeMo-compatible memory-mapped files:

```
/data/curated/tokenized/
    text_document.bin     ← raw uint16 token IDs
    text_document.idx     ← document offsets for O(1) random access
```

These files are uploaded to S3 by `make upload-data` and pulled onto the GPU instance by `make setup-instance`. The pre-training stage consumes them directly with no further conversion.