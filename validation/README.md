# validation

Additional quality validation on top of the curator's heuristic filters. The primary addition is perplexity-based filtering using a KenLM language model — the most impactful filter for removing low-quality web text that passes heuristic checks.

---

## What This Adds

The curator (`curator/filters/quality.py`) already applies heuristic filters — length, symbol ratio, bullet ratio, alpha ratio, stop words, repeated lines. Validation adds:

| Filter | What it catches |
|---|---|
| Terminal punctuation (C4) | Incomplete sentences, truncated content, navigation text |
| Repeated n-grams (Gopher) | Boilerplate, templated content, repeated paragraphs |
| Language detection (fastText) | Non-English content that passed earlier filters |
| Perplexity (KenLM) | Gibberish, SEO spam, malformed text |

---

## Getting Started

**Prerequisites**

```bash
# Install dependencies (included in requirements.txt)
make install

# Install KenLM Python bindings (not on PyPI — built from source)
make install-kenlm

# Download KenLM English language model (~4GB)
make download-kenlm-model
```

**Run validation**

```bash
make validate

# Upload validated data to S3
make validate-upload SIZE=125m
```

**Or directly**

```bash
# Default — manual pipeline with auto perplexity threshold
python validation/scripts/validate.py

# With datatrove pipeline
python validation/scripts/validate.py --use-datatrove

# Custom input/output
python validation/scripts/validate.py \
    --input data/curated/train.jsonl \
    --output data/validated/train.jsonl

# Fixed perplexity threshold
python validation/scripts/validate.py --perplexity-threshold 800

# Skip perplexity (no KenLM model)
python validation/scripts/validate.py --no-perplexity
```

---

## Perplexity Filter

KenLM scores how "natural" each document is according to a 5-gram language model trained on Wikipedia. Low perplexity = natural English text. High perplexity = gibberish, non-English, or malformed content.

The threshold is automatically computed at the **90th percentile** of the perplexity distribution — removing the bottom 10% of documents by quality. This is more principled than a fixed threshold as it adapts to the data distribution.

```
Perplexity distribution (typical web crawl):
    p10:  ~100   ← very high quality
    p50:  ~300   ← average quality
    p90:  ~1500  ← threshold (remove above this)
    p99:  ~5000  ← clearly low quality
```

---

## Two Modes

**Manual pipeline** (default) — pure Python, no datatrove required. Applies terminal punctuation check, repeated line check, and KenLM perplexity filter. Faster to set up.

**datatrove pipeline** (`--use-datatrove`) — uses datatrove's production-grade C4, Gopher, and language filters. More thorough, better tested, recommended for final production runs.

---

## Key Design Decisions

**Why perplexity filtering?** Heuristic filters catch obvious quality issues — too short, too many symbols, too many bullets. Perplexity filtering catches subtle quality issues — text that looks structurally fine but is semantically incoherent or templated. FineWeb found perplexity filtering to be one of the highest-impact filters for improving downstream model quality.

**Why 90th percentile threshold?** Fixed thresholds don't generalize across data sources. Web crawl and Wikipedia have very different perplexity distributions. The 90th percentile adapts to whatever data you feed it.

**Why skip perplexity for code?** Code has high perplexity relative to English prose by definition — it contains identifiers, syntax, and structure that don't follow natural language patterns. Applying a prose-trained perplexity filter to code would incorrectly remove valid code.

**Why KenLM over a neural LM?** KenLM is a count-based n-gram model — scoring a document takes microseconds vs seconds for a neural model. At 125M scale (millions of documents) a neural perplexity filter would take weeks. KenLM's Wikipedia-trained model is fast, robust, and has been validated by the cc_net and FineWeb pipelines.