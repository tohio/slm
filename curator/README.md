# curator

Data curation pipeline for SLM pretraining. Downloads raw data from eighteen concrete sources (13 non-code top-level sources + 5 code sub-sources), applies quality filters, deduplicates, blends to target token ratios with cap-and-redistribute overflow handling, and uploads to S3.

---

## Pipeline

```
common_crawl          ─┐
fineweb               ─┤
fineweb_edu           ─┤
wikipedia             ─┤
pg19                  ─┤
pes2o                 ─┼─► quality filter ─► dedup ─► blend ─► train.jsonl + val.jsonl ─► S3
nemotron_cc_math      ─┤
stackexchange         ─┤
synthetic_arithmetic  ─┤
synthetic_task_code   ─┤
educational_qa_mcq    ─┤
factual_restraint     ─┤
nemotron_specialized  ─┤
code × 5              ─┘
```

Each source runs independently through filtering and deduplication. The blend stage reads deduped shards from each source up to its character budget, with shortfalls routed through source-aware overflow chains.

---

## Data Sources

13 top-level non-code sources plus 5 code sub-sources sharing the 15% code budget = 18 concrete source loaders total.

| Source | Target Share | Supply | Notes |
|---|---:|---|---|
| Common Crawl | 5% | unlimited/time-bound | direct WARC download via HTTPS |
| FineWeb | 10% | very large | `HuggingFaceFW/fineweb`, broad web fallback |
| FineWeb-Edu | 26% | large | `HuggingFaceFW/fineweb-edu`, educational/explanatory web text |
| Wikipedia | 10% | finite | `wikimedia/wikipedia` EN |
| pg19 | 2.5% | finite | public-domain long-form books |
| peS2o | 5% | finite | academic/scientific prose; supply-bound at larger scales |
| Nemotron CC Math | 5% | very large | `nvidia/Nemotron-CC-Math-v1`, math/STEM supplement replacing OpenWebMath |
| StackExchange | 5% | finite/large | Q&A-style web text |
| Synthetic arithmetic | 3% | generated locally | arithmetic formats: QA, bare equations, equation completion, word problems, comparisons, and simple multi-step arithmetic |
| Synthetic task code | 5% | generated locally | targeted task-shaped code examples; Python 70%, Go 15%, Rust 10%, Bash 5% |
| Educational QA/MCQ | 3% | generated locally | targeted QA, MCQ, explanation, and cloze formats; benchmark datasets excluded |
| Factual restraint | 0.5% | generated locally | uncertainty, private/unverifiable facts, no fake search/tool claims |
| Nemotron Specialized | 5% | large | `nvidia/Nemotron-Pretraining-Specialized-v1.1`, scalable specialized/synthetic reservoir |
| **Code total** | **15%** | mixed | split across 5 code sub-sources |

### Code sub-mix (percentages of the 15% code share)

| Code source | Sub-share | Languages | Notes |
|---|---:|---|---|
| the-stack-dedup (v1) | 40% | python, go, rust, shell | `bigcode/the-stack-dedup` — raw-code diversity source |
| CodeSearchNet | 30% | Python, Java, JavaScript, PHP, Ruby, Go | `code_search_net` — curated function/docstring code signal |
| the-stack-smol | 15% | 30 languages | `bigcode/the-stack-smol` — small broad-language diversity slice |
| Jupyter notebooks | 10% | mostly Python | `bigcode/jupyter-parsed` — code+prose |
| CoNaLa | 5% | Python | `neulab/conala` mined — NL-to-code intent pairs |

### Scale-invariant percentages

Percentages are the same at every size. Scaling up changes `corpus_tokens`, not the mix. A reader adding a new size (e.g. `slm-500m`) gets correct per-source budgets without editing curator code.

### Cap-and-redistribute

Several sources are supply-bound at large scales: peS2o (abstracts only) and jupyter run out at 350m+; Wikipedia, pg19, nemotron_cc_math, and stack_smol all become supply-bound at 1b; codesearchnet and conala upstream is small enough to bind even sooner. Each source writes up to its budget or until its supply is exhausted, whichever is smaller. Deficits then follow source-aware overflow chains: local synthetic deficits route to Nemotron Specialized first, then FineWeb-Edu, then FineWeb; general deficits route to FineWeb-Edu first and FineWeb as the final fallback.

This behavior is load-bearing at 1b scale; partially load-bearing at 125m/350m for the always-supply-bound sources (peS2o, jupyter).

### NVIDIA/Nemotron sources are supplemental

FineWeb and FineWeb-Edu remain the broad web base for this from-scratch
training pipeline. NVIDIA/Nemotron datasets supplement specific high-signal
gaps in math and specialized synthetic data.

### Run-specific realized mix

The source mix above defines the pretraining target mix. The actual realized mix for each curation run is written to `data/curated/blend_stats.json` at the end of the blend stage.

Realized percentages can differ slightly from target percentages when a source is supply-bound or filtered/deduplicated more aggressively than expected. Local synthetic deficits route to Nemotron Specialized first, then FineWeb-Edu, then FineWeb. General source deficits route to FineWeb-Edu first and FineWeb as the final fallback.

Use `blend_stats.json` as the source of truth for a completed run. `export.py` reads this file when producing per-model cards.

## Token Targets

| Model | Curation target | Expected retained/tokenized | Epochs | Consumed target |
|---|---:|---:|---:|---:|
| `mini` | 1M | mini only | 1 | 1M |
| `slm-125m` | 10B | ~9B+ | 2 | 20B |
| `slm-350m` | 25B | ~23B+ | 2 | 50B |
| `slm-1b` | 75B | ~69B+ | 1 | 75B |

`corpus_tokens` / curation target is the curator-side target, not a guaranteed final retained token count. The targets include a retention buffer for filtering, validation, deduplication, source availability, and tokenization losses.

Why 1b uses 1 epoch: at a 75B curation target, one epoch already gives the 1B model a materially larger fresh-token budget than the smaller sizes while avoiding an immediate second pass over finite sources. 125m and 350m retain 2 epochs because their smaller corpus targets are intended for cheaper iteration and validation.

---

## Structure

```
curator/
├── constants.py             Re-exports CHARS_PER_TOKEN etc. from config/data_mix.py
├── sources/
│   ├── common_crawl.py        Common Crawl WARCs via HTTPS + trafilatura
│   ├── fineweb.py             HuggingFaceFW/fineweb (streaming, final fallback)
│   ├── fineweb_edu.py         HuggingFaceFW/fineweb-edu educational web text
│   ├── wikipedia.py           wikimedia/wikipedia EN
│   ├── pg19.py                pg19 public-domain books
│   ├── pes2o.py               allenai/peS2o academic papers (streaming)
│   ├── nemotron_cc_math.py    nvidia/Nemotron-CC-Math-v1 (streaming)
│   ├── stackexchange.py       HuggingFaceH4 stack-exchange Q+A (streaming)
│   ├── nemotron_specialized.py nvidia/Nemotron-Pretraining-Specialized-v1.1
│   ├── synthetic_arithmetic.py generated arithmetic pretraining source
│   ├── synthetic_task_code.py  generated task-shaped code examples
│   ├── educational_qa_mcq.py   generated QA/MCQ educational examples
│   ├── factual_restraint.py    generated factual-restraint examples
│   ├── code_search_net.py     CodeSearchNet — 6 languages
│   ├── stack_smol.py          bigcode/the-stack-smol — 30 languages
│   ├── stack_v1.py            bigcode/the-stack-dedup (inline content)
│   ├── stack_v2.py            DISABLED — see file header
│   ├── jupyter.py             bigcode/jupyter-parsed
│   └── conala.py              neulab/conala-mined
├── filters/
│   ├── quality.py             Heuristic quality filters (FineWeb/Gopher-style)
│   └── dedup.py               Exact + datatrove disk-based MinHash deduplication
└── scripts/
    ├── curate.py              Main pipeline entry point, mix layer, cap-and-redistribute
    ├── sample_source.py       Print actual source/stage records for human review
    └── upload_s3.py           S3 upload/download utilities
```

---

## Getting Started

**Prerequisites**

```bash
pip install -r requirements.txt
cp .env.sample .env
# Set S3_BUCKET, AWS credentials, DATA_DIR, HF_TOKEN, SWH_AUTH_TOKEN in .env

# Download the fasttext language ID model (~1MB) — required before first run
make download-fasttext-model
```

One environment variable is required beyond the existing ones:
- `HF_TOKEN` — required for gated datasets (BigCode and selected NVIDIA/Nemotron datasets). Accept Terms of Use on each dataset's Hugging Face page before first run.

**Accept dataset Terms of Use**

Before the first curation run, accept the terms for any gated Hugging Face
datasets used by the active mix. The active mix uses BigCode and selected
NVIDIA/Nemotron datasets.

Required access checks:
- https://huggingface.co/datasets/bigcode/the-stack-dedup
- https://huggingface.co/datasets/bigcode/the-stack-smol
- https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1
- https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.1

After accepting terms, make sure the token is available on the machine running
curation:

```bash
huggingface-cli login
# or
export HF_TOKEN=hf_...
```

If credentials are stored in `.env`, `curate.py` loads them. Standalone test
snippets should explicitly load `.env` with:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".env"), override=False)
```

Nemotron Specialized subset policy:

```text
Use:
  - Nemotron-Pretraining-Code-Concepts
  - Nemotron-Pretraining-Unconditional-Algorithmic
  - Nemotron-Pretraining-Formal-Logic
  - Nemotron-Pretraining-Economics

Exclude by default:
  - Nemotron-Pretraining-Multiple-Choice
```

The standalone Multiple-Choice config is excluded by default because it may
introduce downstream DeepSeek-license obligations for distributed or hosted
derivative models.

**Minimal run — validate the pipeline before committing to a full run**

```bash
python curator/scripts/curate.py --target mini --mini
```

Mini exercises every source at small scale (~100 docs to a few thousand per source) to validate end-to-end that all 18 active source loaders, filter logic, dedup, and the mix layer work correctly. Total runtime 30–60 min.

Run each stage individually to inspect output between steps:

```bash
python curator/scripts/curate.py --target mini --mini --stage download
python curator/scripts/curate.py --target mini --mini --stage filter
python curator/scripts/curate.py --target mini --mini --stage dedup
python curator/scripts/curate.py --target mini --mini --stage blend
```

**Full pipeline**

```bash
# 125m dataset (10B curation target; ~9B+ expected retained/tokenized)
python curator/scripts/curate.py --target 125m

# 350m dataset (25B curation target; ~23B+ expected retained/tokenized)
python curator/scripts/curate.py --target 350m --workers 32

# 1b dataset (75B curation target; ~69B+ expected retained/tokenized)
python curator/scripts/curate.py --target 1b --workers 64
```

**Individual stages**

```bash
python curator/scripts/curate.py --target 125m --stage download
python curator/scripts/curate.py --target 125m --stage filter
python curator/scripts/curate.py --target 125m --stage dedup --workers 8
python curator/scripts/curate.py --target 125m --stage blend
python curator/scripts/curate.py --target 125m --stage upload
```

**Human sample inspection**

After mini curation, inspect actual records from each source before scaling to a full dataset. Confirm that the data is readable, complete, useful, and aligned with the expected signal for that source.

```bash
python curator/scripts/sample_source.py --stage filtered --source wikipedia --limit 10 --max-chars 2500
python curator/scripts/sample_source.py --stage filtered --source fineweb --limit 10 --max-chars 2500
python curator/scripts/sample_source.py --stage filtered --source nemotron_cc_math --limit 10 --max-chars 2500
python curator/scripts/sample_source.py --stage filtered --source stackexchange --limit 10 --max-chars 2500
python curator/scripts/sample_source.py --stage filtered --source codesearchnet --limit 10 --max-chars 2500
```

Use `--random` when you want samples from across the source rather than the first matching records:

```bash
python curator/scripts/sample_source.py --stage filtered --source fineweb --limit 20 --max-chars 2500 --random --seed 13
```

Review samples manually:

- Is the text readable and complete?
- Does it look like real content or boilerplate/noise?
- Does the source provide the expected signal: factual prose, math, Q&A, code, or instruction-shaped examples?
- Did filtering or validation remove useful formatting?
- Are generated/template-like sources producing direct, verified examples rather than vague prose?

Quality filters can remove obvious junk, but they do not prove that a source teaches the model the behavior we want. Repeat sample inspection when sources, filters, validation, or extraction logic change.

**S3 upload**

Use `make curate-upload SIZE=125m` or the stage directly — the upload automatically creates a versioned path by target and date:

```bash
make curate-upload SIZE=125m
# or
python curator/scripts/curate.py --target 125m --stage upload
```

For manual S3 operations:

```bash
python curator/scripts/upload_s3.py list
python curator/scripts/upload_s3.py download --src 125m/2026-04-02/curated --dst data/curated
```

---

## Data Directory Layout

```
data/
├── raw/
│   ├── common_crawl/               raw CC JSONL shards + cc_progress.json
│   ├── fineweb/                    streamed FineWeb shards
│   ├── fineweb_edu/                streamed FineWeb-Edu shards
│   ├── wikipedia/                  raw Wikipedia shards
│   ├── pg19/                       pg19 book shards
│   ├── pes2o/                      streamed peS2o shards
│   ├── nemotron_cc_math/           streamed math web shards
│   ├── stackexchange/              streamed SE Q+A shards
│   ├── synthetic_arithmetic/       generated arithmetic shards
│   ├── synthetic_task_code/        generated task-code shards
│   ├── educational_qa_mcq/         generated QA/MCQ shards
│   ├── factual_restraint/          generated factual-restraint shards
│   ├── nemotron_specialized/       streamed specialized synthetic shards
│   ├── codesearchnet/              CSN 6-language shards
│   ├── stack_smol/                 stack-smol 30-language shards
│   ├── stack_v1/                   stack-v1 4-language shards (content inline)
│   ├── stack_v2/                   (disabled — present only if re-enabled)
│   ├── jupyter/                    jupyter notebook shards
│   └── conala/                     CoNaLa pair shards
├── filtered/
│   ├── <source>/                   quality-filtered shards
│   └── <source>deduped/           + deduplicated
├── dedup_scratch/                  datatrove intermediate state (cleaned per-source after success)
│   └── <source>/                   per-source exact + minhash state
└── curated/
├── blend<source>.jsonl        per-source staging (cleaned up after shuffle)
├── train.jsonl                 final blended train split
├── val.jsonl                   final blended val split (uniform sample)
└── blend_stats.json            per-source docs/chars/deficit/val_docs breakdown
```

---

## S3 Structure

Each upload is versioned by target and date, so multiple runs never overwrite each other:

```
s3://your-bucket/slm/data/
├── 125m/
│   └── YYYY-MM-DD/
│       └── curated/
│           ├── train.jsonl
│           ├── val.jsonl
│           └── blend_stats.json
├── 350m/
│   └── YYYY-MM-DD/curated/
├── 1b/
│   └── YYYY-MM-DD/curated/
└── mini/
    └── YYYY-MM-DD/curated/

```

Re-uploading on the same day overwrites that day's run. Runs on different days are preserved independently.

---

## Quality Filters

Heuristics adapted from FineWeb and Gopher. Filters marked ✗ are skipped for code-adjacent or symbol-heavy generated sources (`synthetic_arithmetic`, `synthetic_task_code`, `educational_qa_mcq`, `factual_restraint`, `nemotron_specialized`, `codesearchnet`, `stack_smol`, `stack_v1`, `jupyter`, `conala`) — symbol-heavy syntax, long identifiers, numeric expressions, and absence of stop words are normal properties of these sources, not quality signals.

The set of code-adjacent source tags lives in `curator/filters/quality.py` as `CODE_SOURCES`. Adding a new code-adjacent or symbol-heavy source is a single-line change.

| Filter | Threshold | Catches | Skipped for code |
|---|---|---|---|
| Min length | 500 chars | Stubs, empty pages | |
| Max length | 50k chars | Extremely long documents | |
| Mean word length | 3–10 chars | Gibberish, SEO spam | ✗ |
| Symbol ratio | < 8% symbols/words | Symbol-heavy spam | ✗ |
| Bullet ratio | < 90% bullet lines | Pure list content | |
| Ellipsis ratio | < 30% ellipsis lines | Truncated content | |
| Alpha ratio | > 75% alpha chars | Numeric/code spam | ✗ |
| Repeated lines | < 20% duplicates | Boilerplate, repeated content | |
| Boilerplate patterns | < 2 matches | Cookie banners, JS-required pages | ✗ |
| Language (fasttext) | EN score ≥ 0.65 | Non-English content | ✗ |
| Stop words (fallback) | ≥ 3 EN stop words | Non-English when fasttext missing | ✗ |

**Mixed-content sources (jupyter, conala) and generated/template-like sources are included in `CODE_SOURCES` or fuzzy-dedup skip handling as appropriate.** Their prose or numeric components bypass English-prose filters as a result. This is an accepted trade-off: per-chunk filter dispatch isn't feasible at the source level, and skipping prose filters on these is safer than rejecting valid code or dense arithmetic examples.

**StackExchange HTML stripping.** The HF `HuggingFaceH4/stack-exchange-preferences` dataset stores Q+A bodies as raw HTML (`<p>...</p>` etc.). Tags are stripped at extraction time in `curator/sources/stackexchange.py` — without this, the symbol-ratio filter would reject 99.93% of records. Block-level closing tags (`</p>`, `</div>`, etc.) are converted to paragraph breaks before stripping so structure survives.

---

## Deduplication

Two-stage deduplication applied after quality filtering, per source:

**Stage 1 — Exact dedup.** SHA-256 (8-byte prefix, binary) of normalized text. The hash index is shared across all sources within a run — a Wikipedia article that also appears in Common Crawl is caught. Grows at ~8 bytes/document; at 100M documents that's ~800MB.

**Stage 2 — Fuzzy dedup (datatrove).** 4-stage disk-based MinHash LSH pipeline: signatures → buckets → cluster → filter. Catches near-duplicates (Jaccard similarity > 0.8). Peak RAM is bounded by shard size, not corpus size — 125m, 350m, and 1b run with the same memory footprint.

Generated/template-like sources (`synthetic_arithmetic`, `synthetic_task_code`, `educational_qa_mcq`, `factual_restraint`) still run exact dedup, but bypass fuzzy MinHash dedup. MinHash collapses useful near-duplicate template variation too aggressively; exact dedup only removes true duplicate rows.

Per-source scratch (`data/dedup_scratch/<source>/`) is deleted automatically after each source's MinHash filter writes its output successfully. Without this, the 125m run accumulated 135 GB of scratch across the source set; at 1b it would scale to ~780 GB and not fit on a 2 TB disk alongside raw + filtered + curated.

---

## Blend

Three passes:

**Pass 1 (parallel).** Each source streams its deduped shards to a per-source staging file (`blend_<source>.jsonl`), stopping when the source's character target is reached or its supply is exhausted. Deficit (target minus actual) is recorded per source.

**Pass 2 (sequential).** If total deficit > 0, source-aware overflow chains append additional content. Local synthetic deficits route to Nemotron Specialized first, then FineWeb-Edu, then FineWeb. General source deficits route to FineWeb-Edu first and FineWeb as the final fallback.

**Pass 3 (shuffle + split).** Two shuffle strategies based on size, both producing globally-mixed train and a uniform-sample val:
- **In-memory** — when total staging (scaled by ~5× for Python object overhead) fits in `SHUFFLE_RAM_BUDGET_GB` (default 12 GB). Read everything, shuffle once, write split.
- **Weighted-interleave + reservoir sample** — when staging exceeds the RAM budget. Open all staging files at once; at each step pick a source weighted by remaining lines and read one line. Each line either enters a reservoir (val sample, uniform across the corpus by Vitter's Algorithm R) or a chunk buffer (train, written to disk in shuffled chunks). Train chunks are then concatenated in shuffled order. Reservoir sampling during blend keeps the validation split uniform across the corpus.

Characters-to-tokens conversion uses `CHARS_PER_TOKEN = 4.3` from `config/data_mix.py` — measured from the 32k-vocab tokenizer trained on the 125m corpus. Recalibrate there if the tokenizer is retrained on a substantially different corpus.

Per-source val doc counts are written to `blend_stats.json`'s `val_docs` field so the realized val mix can be inspected post-blend; at 125m val matches train within ±0.25pp per source.

---

## Output Format

Each record in the final `train.jsonl` and `val.jsonl`:

```json
{
  "text": "...",
  "source": "<source name>",
  "language": "en",
  "...": "source-specific metadata fields"
}
```

Per-source metadata varies (e.g. Wikipedia has `title` and `url`; CodeSearchNet has `repo` and `path`; peS2o has `paper_id` and `subset`). All records carry `text` and `source` at minimum.

---

## Infrastructure

### Hardware recommendations

These are recommendations, not floors. The pipeline streams everywhere, so RAM isn't strictly load-bearing — a reader with less RAM can run 1b, it just takes longer. vCPU count matters more than RAM for throughput (CC download + MinHash dedup are CPU-bound).

| Target | vCPUs | RAM | Curation runtime |
|---|---|---|---|
| `mini` | 4+ | 8 GB | 30–60 min |
| `slm-125m` | 16+ | 32 GB | ~16 hrs (measured: 11h25m download + 16m filter + 3h6m dedup + 3m blend) |
| `slm-350m` | 32+ | 64 GB | _TBD — pending 350m run_ |
| `slm-1b` | 64+ | 128 GB | _TBD — pending 1b run_ |

Runtimes are CPU-bound on MinHash dedup and I/O-bound on Common Crawl download. Download dominates wall time at 125m (~72%) and is unlikely to scale linearly to 350m/1b — supply-bound sources converge regardless of target size, while CC segments and FineWeb stream length grow with the budget.

Run close to `us-east-1` (AWS) or `us-east1` (GCP) to minimise Common Crawl egress latency. Attach a persistent disk (500GB+) for `DATA_DIR` — the pipeline is fully resumable at every stage.

### Preemptible interruption handling

- **Common Crawl** tracks progress per WARC segment in `cc_progress.json`.
- **FineWeb / FineWeb-Edu / peS2o / nemotron_cc_math / nemotron_specialized / StackExchange / the-stack-v1 / pg19** (streaming sources) track progress by counting completed shards and skipping that many records on restart.
- **Filter / dedup / blend** skip files that already exist on disk.

Restart the exact same command to resume. At most one segment or shard of work is lost per interruption.

### Use tmux for long runs

```bash
tmux new -s curate
make curate SIZE=125m WORKERS=62
# Ctrl+B, D to detach — tmux attach -t curate to reattach
```

### Open-file-descriptor limit

MinHash dedup of large sources (stack_v1 has ~2,103 shards at 125m) opens many files concurrently in stage 2 (LSH bucketing). The default `ulimit -n 1024` is insufficient — the curate Make targets prepend `ulimit -n 65536 && ` to lift the limit before invoking the pipeline.

---

## Key Design Decisions

**Why 18 concrete sources?** Distribution coverage. A model pretrained only on web scrape (even filtered) has characteristic weaknesses: poor factual recall on niche topics (→ Wikipedia), no long-range coherence over book-length spans (→ pg19), weak technical/academic prose (→ peS2o), weak math reasoning and math-page style (→ nemotron_cc_math), sparse clean elementary arithmetic mappings (→ synthetic_arithmetic), weak Q+A structure (→ StackExchange), weak educational/explanatory web signal (→ FineWeb-Edu), weak task-shaped code signal (→ synthetic_task_code), weak QA/MCQ answer-selection format (→ educational_qa_mcq), weak factual restraint/uncertainty behavior (→ factual_restraint), and weak code (→ 5 code sources covering raw bulk, curated functions, multi-language samples, notebook prose+code, and NL→code intent). Each source covers a specific gap.

**Why scale-invariant percentages?** A reader scaling from 125m to 1b should change one number (`corpus_tokens`) and get proportionally more of everything. Per-scale mix tuning is an axis of complexity that serves no one; the supply-constrained case is handled by cap-and-redistribute, not per-scale knobs.

**Why stack-v1 capped at 50% of code?** stack-v1 is raw code files with minimal metadata; CodeSearchNet has docstrings, Jupyter has prose-and-code, CoNaLa has NL-intent pairs. Those sources teach the model *how humans describe and explain code*, not just syntax. Letting stack-v1 dominate at 90%+ would trade the describe-and-explain signal away for more raw-completion data.

**Why sample-100BT for FineWeb instead of a specific CC snapshot?** Reproducibility. `sample-100BT` is a deterministic 100B-token subset of FineWeb — anyone who runs the same code gets the same data. Named snapshots also work but are subject to FineWeb re-releases and can drift.

**Why trafilatura over BeautifulSoup for Common Crawl?** trafilatura is specifically designed for main-content extraction from web pages. It handles boilerplate removal (navigation, ads, footers) significantly better than generic HTML parsers.

**Why HTTPS for Common Crawl instead of S3?** Direct S3 access to the `commoncrawl` bucket fails on EC2 instances with IAM roles attached — the role credentials are rejected by the bucket policy. HTTPS via `data.commoncrawl.org` works reliably regardless of instance credentials.

**Why streaming-first code?** At 1b scale with 75B curation targets, materializing any large source in memory is infeasible on reasonable hardware. FineWeb, stack-v1, and pg19 use streaming because their on-HF layouts (large volume or many small parquet files) make full-dataset downloads impractical. The other sources use streaming for consistency so the pipeline works uniformly across hardware sizes. RAM is not the load-bearing scaling axis here — vCPU count and network throughput are.

**Why datatrove for dedup instead of datasketch?** datasketch's `MinHashLSH` is in-memory. At 350m it requires ~32GB; at 1b it requires ~85GB and may not fit on a single instance. datatrove's disk-based pipeline keeps RAM bounded by shard size regardless of corpus size — the same pattern used by FineWeb and RedPajama at trillion-token scale.

**Why fasttext over langdetect?** Language detection runs on every Common Crawl document. fasttext's `lid.176.ftz` is C-backed and ~1000× faster than pure-Python `langdetect` at equivalent accuracy, covering 176 languages. The model is ~1MB, downloaded once via `make download-fasttext-model`.

**Why versioned S3 uploads?** Each run uploads to `{target}/{date}/curated/` so multiple runs never overwrite each other. Safe to re-run curation with different parameters and compare results; allows rolling back to a previous run if issues are found during training.

**Why per-stage resumability?** Curation runs take hours to days on spot instances that can be interrupted with 2 minutes notice. Each stage checks for existing output before processing — safe to interrupt and restart without reprocessing completed work.

**Why blend-time train/val split via reservoir sampling?** Splitting at training time silently drifts out of sync with the underlying tokenization. Splitting via tail-slice of the blend's chunked shuffle produces a non-uniform val sample (the tail inherits source-bias from whichever chunks land last). Reservoir sampling during the blend gives a uniform sample across the entire corpus by construction; the realized val mix matches train within ±0.25pp per source.

---

## Scaling Beyond 1b

The pipeline is designed to scale. Scale-invariant mix percentages, streaming-first code, and cap-and-redistribute all generalise to larger targets. To run at 3b or beyond:

1. Add an entry to `TARGET_CONFIGS` in `config/data_mix.py` with the new `corpus_tokens`, `epochs`, and `cc_crawls` list.
2. Review Wikipedia and pg19 supply: at token budgets approaching 40B × 1 epoch (equivalent to 1b × 2 epochs), Wikipedia repetition approaches 1.6×. Either drop Wikipedia's share to ~7% at that scale, or accept the repetition.
3. At 5B+ code tokens, consider adding a second bulk-code source to avoid stack-v1 over-epoching.
4. Consider upgrading FineWeb from `sample-100BT` to a larger sample or the full dataset, depending on how much of FineWeb's headroom the new target consumes.

No code changes are required for scaling — the target config, source mix, and cap-and-redistribute handle supply variance automatically.

---

## Contamination

The following eval benchmarks are **not** present in any training source:

- HumanEval (Python code completion)
- MBPP (Mostly Basic Python Problems)
- APPS (earlier `codeparrot/apps` was considered; dropped from the mix specifically to keep this clean)
- HellaSwag, ARC, MMLU, TruthfulQA (general-knowledge evals — not in code or academic sources)

Documented here so model cards can claim clean code-eval results without asterisks.

One source worth flagging: peS2o overlaps with academic papers. If future evals use paper-QA benchmarks (QASPER, SciQ, etc.), contamination analysis would be needed.

---

## Known Limitations

**stack-v1 near-duplicate coverage.** stack-v1 applies exact deduplication but not near-duplicate removal. Near-dups in code (forks, templates, auto-generated files with small variants) slip through. The downstream MinHash dedup stage catches some of this, but v2's dataset-level near-dup filtering was stronger. Acceptable tradeoff for avoiding SWH's rate-limit problems; re-evaluate if repetition shows up in eval.

**Jupyter and CoNaLa prose components are not language-filtered.** Labeling them as code-adjacent skips English-prose filters, which means non-English prose in these sources passes through. The prose volume is small and largely English-coded on GitHub/StackOverflow, so this is not a meaningful corpus contamination, but the model will see the occasional non-English notebook comment or SO intent.

**Char-to-token ratio is approximate.** `CHARS_PER_TOKEN = 4.3` in `config/data_mix.py` is the measured average for the trained tokenizer on the 125m corpus (excluding code sources). Real ratios vary by domain: English prose ~4.5, code ~3.5, math ~3. The approximation is fine for target sizing; filtering, validation, deduplication, source availability, and tokenization reduce the final retained/tokenized corpus. Recalibrate if the tokenizer is retrained on a substantially different mix, especially after adding symbol-heavy generated data.

**wikipedia cold-cache overhead.** wikipedia's `wikimedia/wikipedia` 20231101.en config loads the full ~19GB dataset (~6.4M articles) before iteration starts, even though mini only uses 5000 articles and 125m/1b only use a fraction. Cold-cache runs spend 10–15 minutes on wikipedia alone; once cached, subsequent runs are instant. Could be migrated to streaming like pg19 was, if this overhead becomes disruptive.

## Curation performance notes

The curation pipeline is parallel where the work is CPU-heavy or structurally safe to parallelize.

Common Crawl uses a dedicated parallel pipeline because WARC processing has two distinct bottlenecks:

- network-bound HTTPS WARC downloads
- CPU-bound WARC parsing, trafilatura extraction, and language detection

The Common Crawl source therefore uses separate download and extraction worker pools. Worker counts are derived from the global curation worker count so high-CPU machines can keep both download and extraction busy.

Most Hugging Face dataset sources are streamed sequentially during the download stage. This is intentional for now. Streaming one source at a time keeps stdout readable, avoids excessive concurrent Hugging Face Hub requests, and makes failures easier to debug. Data curation is an occasional batch process, not a continuously running service, so the pipeline prioritizes reproducibility and operational clarity over maximizing every possible download parallelism opportunity.

Potential future optimizations:

- Run non-Common-Crawl source downloads concurrently with per-source log files.
- Add clean parent-process progress summaries for parallel source downloads.
- Tune per-source download caps from measured filter/dedup retention.
- Parallelize naturally partitioned sources such as CodeSearchNet by language.
- Parallelize Stack sources by language or data directory.
