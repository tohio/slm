# curator

Data curation pipeline for SLM pretraining. Downloads raw data from ten sources, applies quality filters, deduplicates, blends to target token ratios with cap-and-redistribute overflow handling, and uploads to S3.

---

## Pipeline

```
common_crawl   ─┐
fineweb        ─┤
wikipedia      ─┤
pg19           ─┤
pes2o          ─┼─► quality filter ─► dedup ─► blend ─► train.jsonl ─► S3
open_web_math  ─┤
stackexchange  ─┤
code × 5       ─┘
```

Each source runs independently through filtering and deduplication. The blend stage reads deduped shards from each source up to its character budget, with any shortfall from supply-constrained sources routed to FineWeb as an overflow sink.

---

## Data Sources

| Source | Share | Supply | Notes |
|---|---|---|---|
| Common Crawl | 10% | unlimited (time-bound) | direct WARC download via HTTPS |
| FineWeb | 47.5% | 15T tokens | `HuggingFaceFW/fineweb` (`sample-100BT` subset), overflow sink |
| Wikipedia | 10% | ~3.7B tokens | `wikimedia/wikipedia` 20231101.en |
| pg19 | 2.5% | ~2.9B tokens | `pg19` — public-domain books pre-1919 |
| peS2o | 5% | ~42B tokens | `allenai/peS2o` v2 — academic papers |
| open-web-math | 10% | ~14.7B tokens | `open-web-math/open-web-math` |
| StackExchange | 5% | ~15B tokens | `HuggingFaceH4/stack-exchange-preferences` Q+A |
| **Code total** | **10%** | | split across 5 sub-sources |

### Code sub-mix (percentages of the 10% code share)

| Code source | Sub-share | Languages | Notes |
|---|---|---|---|
| the-stack-dedup (v1) | 50% | python, go, rust, shell | `bigcode/the-stack-dedup` — bulk raw code, content inline in parquet shards |
| CodeSearchNet | 35% | Python, Java, JavaScript, PHP, Ruby, Go | `code_search_net` — curated function-level with docstrings |
| the-stack-smol | 10% | 30 languages | `bigcode/the-stack-smol` — diverse small sample |
| Jupyter notebooks | 4% | mostly Python | `codeparrot/github-jupyter-code-to-text` — code+prose |
| CoNaLa | 1% | Python | `neulab/conala` mined — NL→code pairs |

### Scale-invariant percentages

Percentages are the same at every size. Scaling up changes `target_tokens`, not the mix. A reader adding a new size (e.g. `slm-500m`) gets correct per-source budgets without editing curator code.

### Cap-and-redistribute

Finite sources (Wikipedia, pg19, etc.) may supply less than their character budget at larger scales. Each source writes up to its budget or until its supply is exhausted, whichever is smaller. The total shortfall is added to FineWeb's budget at the end of staging. FineWeb has effectively unlimited supply (15T tokens) so this always closes the gap.

This behavior is load-bearing at 1b scale, where pg19 is supply-constrained; less visible at 125m and 350m where most sources have comfortable headroom.

---

## Token Targets

| Model | Total tokens | Epochs | Supply situation |
|---|---|---|---|
| `mini` | 1M | 1 | all sources comfortable |
| `slm-125m` | 5B | 2 | all sources comfortable |
| `slm-350m` | 15B | 2 | all sources comfortable |
| `slm-1b` | 30B | 1 | pg19 near supply limit; FineWeb overflow covers |

Why 1b uses 1 epoch: at 30B tokens with a single epoch, every source is below its supply ceiling, so no repetition. Modern small-model training (Llama, Phi) follows the same pattern — more fresh data outperforms fewer tokens seen multiple times. 125m and 350m retain 2 epochs because their smaller token budgets leave plenty of headroom.

---

## Structure

```
curator/
├── constants.py             Shared constants (CHARS_PER_TOKEN, CC_CHARS_PER_SEGMENT)
├── sources/
│   ├── common_crawl.py        Common Crawl WARCs via HTTPS + trafilatura
│   ├── fineweb.py             HuggingFaceFW/fineweb (streaming, overflow sink)
│   ├── wikipedia.py           wikimedia/wikipedia EN
│   ├── pg19.py                pg19 public-domain books
│   ├── pes2o.py               allenai/peS2o academic papers (streaming)
│   ├── open_web_math.py       open-web-math (streaming)
│   ├── stackexchange.py       HuggingFaceH4 stack-exchange Q+A (streaming)
│   ├── code_search_net.py     CodeSearchNet — 6 languages
│   ├── stack_smol.py          bigcode/the-stack-smol — 30 languages
│   ├── stack_v1.py            bigcode/the-stack-dedup (inline content)
│   ├── stack_v2.py            DISABLED — see file header
│   ├── jupyter.py             codeparrot jupyter-code-to-text
│   └── conala.py              neulab/conala-mined
├── filters/
│   ├── quality.py             Heuristic quality filters (FineWeb/Gopher-style)
│   └── dedup.py               Exact + datatrove disk-based MinHash deduplication
└── scripts/
    ├── curate.py              Main pipeline entry point, mix layer, cap-and-redistribute
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
- `HF_TOKEN` — required for gated datasets (FineWeb, the-stack-smol, the-stack-dedup). Accept Terms of Use on each dataset's HuggingFace page before first run.

**Minimal run — validate the pipeline before committing to a full run**

```bash
python curator/scripts/curate.py --target mini --mini
```

Mini exercises every source at small scale (~100 docs to a few thousand per source) to validate end-to-end that all 10 source loaders, filter logic, dedup, and the mix layer work correctly. Total runtime 30–60 min.

Run each stage individually to inspect output between steps:

```bash
python curator/scripts/curate.py --target mini --mini --stage download
python curator/scripts/curate.py --target mini --mini --stage filter
python curator/scripts/curate.py --target mini --mini --stage dedup
python curator/scripts/curate.py --target mini --mini --stage blend
```

**Full pipeline**

```bash
# 125m dataset (~5B tokens)
python curator/scripts/curate.py --target 125m

# 350m dataset (~15B tokens)
python curator/scripts/curate.py --target 350m --workers 32

# 1b dataset (~30B tokens)
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
│   ├── wikipedia/                  raw Wikipedia shards
│   ├── pg19/                       pg19 book shards
│   ├── pes2o/                      streamed peS2o shards
│   ├── open_web_math/              streamed math web shards
│   ├── stackexchange/              streamed SE Q+A shards
│   ├── codesearchnet/              CSN 6-language shards
│   ├── stack_smol/                 stack-smol 30-language shards
│   ├── stack_v1/                   stack-v1 4-language shards (content inline)
│   ├── stack_v2/                   (disabled — present only if re-enabled)
│   ├── jupyter/                    jupyter notebook shards
│   └── conala/                     CoNaLa pair shards
├── filtered/
│   ├── <source>/                   quality-filtered shards
│   └── <source>_deduped/           + deduplicated
├── dedup_scratch/                  datatrove intermediate state
│   └── <source>/                   per-source exact + minhash state
└── curated/
    ├── blend_<source>.jsonl        per-source staging (cleaned up after shuffle)
    ├── train.jsonl                 final blended dataset
    └── blend_stats.json            per-source docs/chars/deficit breakdown
```

---

## S3 Structure

Each upload is versioned by target and date, so multiple runs never overwrite each other:

```
s3://your-bucket/slm/data/
├── 125m/
│   ├── 2026-04-02/
│   │   └── curated/
│   │       ├── train.jsonl
│   │       └── blend_stats.json
│   └── 2026-04-15/
│       └── curated/
│           ├── train.jsonl
│           └── blend_stats.json
├── 350m/
│   └── 2026-04-20/curated/
├── 1b/
│   └── 2026-05-05/curated/
└── mini/
    └── 2026-04-01/curated/
```

Re-uploading on the same day overwrites that day's run. Runs on different days are preserved independently.

---

## Quality Filters

Heuristics adapted from FineWeb and Gopher. Filters marked ✗ are skipped for code-adjacent sources (`codesearchnet`, `stack_smol`, `stack_v2`, `jupyter`, `conala`) — symbol-heavy syntax, long identifiers, and absence of stop words are normal properties of code, not quality signals.

The set of code-adjacent source tags lives in `curator/filters/quality.py` as `CODE_SOURCES`. Adding a new code-adjacent source is a single-line change.

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

**Mixed-content sources (jupyter, conala) are included in `CODE_SOURCES`.** Their prose components bypass English-prose filters as a result. This is an accepted trade-off: per-chunk filter dispatch isn't feasible at the source level, and skipping prose filters on these is safer than rejecting valid code.

---

## Deduplication

Two-stage deduplication applied after quality filtering, per source:

**Stage 1 — Exact dedup.** SHA-256 (8-byte prefix, binary) of normalized text. The hash index is shared across all sources within a run — a Wikipedia article that also appears in Common Crawl is caught. Grows at ~8 bytes/document; at 100M documents that's ~800MB.

**Stage 2 — Fuzzy dedup (datatrove).** 4-stage disk-based MinHash LSH pipeline: signatures → buckets → cluster → filter. Catches near-duplicates (Jaccard similarity > 0.8). Peak RAM is bounded by shard size, not corpus size — 125m, 350m, and 1b run with the same memory footprint.

Intermediate state written to `data/dedup_scratch/` and safe to delete after the dedup stage completes.

---

## Blend

Three passes:

**Pass 1 (parallel).** Each source streams its deduped shards to a per-source staging file (`blend_<source>.jsonl`), stopping when the source's character target is reached or its supply is exhausted. Deficit (target minus actual) is recorded per source.

**Pass 2 (sequential).** If total deficit > 0, FineWeb appends additional content to its staging file to cover the shortfall. FineWeb is the overflow sink because its supply (15T tokens) exceeds any deficit we could realistically produce.

**Pass 3 (shuffle).** Two shuffle strategies based on size:
- **In-memory** — when total staging (scaled by ~5× for Python object overhead) fits in `SHUFFLE_RAM_BUDGET_GB` (default 12 GB). One pass: read everything, shuffle once, write.
- **Chunked disk** — read staging files into shuffled chunks, then shuffle chunk order and concatenate. Peak RAM bounded by `chunk_lines × avg_line_size`.

Characters-to-tokens conversion uses `CHARS_PER_TOKEN = 5` from `curator/constants.py`. Recalibrate there if the empirical ratio shifts by more than ~10% after a 125m tokenized run.

---

## Output Format

Each record in the final `train.jsonl`:

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

| Target | vCPUs | RAM | Est. curation runtime |
|---|---|---|---|
| `mini` | 4+ | 8 GB | 30–60 min |
| `slm-125m` | 16+ | 32 GB | _TBD — pending 125m rerun_ |
| `slm-350m` | 32+ | 64 GB | _TBD — pending 350m run_ |
| `slm-1b` | 64+ | 128 GB | _TBD — pending 1b run_ |

Runtimes are CPU-bound on MinHash dedup and I/O-bound on Common Crawl download.

Run close to `us-east-1` (AWS) or `us-east1` (GCP) to minimise Common Crawl egress latency. Attach a persistent disk (500GB+) for `DATA_DIR` — the pipeline is fully resumable at every stage.

### Preemptible interruption handling

- **Common Crawl** tracks progress per WARC segment in `cc_progress.json`.
- **FineWeb / peS2o / open-web-math / StackExchange / the-stack-v1** (streaming sources) track progress by counting completed shards and skipping that many records on restart.
- **Filter / dedup / blend** skip files that already exist on disk.

Restart the exact same command to resume. At most one segment or shard of work is lost per interruption.

### Use tmux for long runs

```bash
tmux new -s curate
make curate SIZE=125m WORKERS=62
# Ctrl+B, D to detach — tmux attach -t curate to reattach
```

---

## Key Design Decisions

**Why 10 sources?** Distribution coverage. A model pretrained only on web scrape (even filtered) has characteristic weaknesses: poor factual recall on niche topics (→ Wikipedia), no long-range coherence over book-length spans (→ pg19), weak technical/academic prose (→ peS2o), weak math reasoning (→ open-web-math), weak Q+A structure (→ StackExchange), weak code (→ 5 code sources). Each source covers a specific gap.

**Why scale-invariant percentages?** A reader scaling from 125m to 1b should change one number (`target_tokens`) and get proportionally more of everything. Per-scale mix tuning is an axis of complexity that serves no one; the supply-constrained case is handled by cap-and-redistribute, not per-scale knobs.

**Why FineWeb as overflow sink?** It has the largest supply (15T tokens, ~500× our largest target) and is the most web-representative of the non-CC sources. Routing deficit there preserves the mix shape while guaranteeing token targets are hit.

**Why stack-v1 capped at 50% of code?** stack-v1 is raw code files with minimal metadata; CodeSearchNet has docstrings, Jupyter has prose-and-code, CoNaLa has NL-intent pairs. Those sources teach the model *how humans describe and explain code*, not just syntax. Letting stack-v1 dominate at 90%+ would trade the describe-and-explain signal away for more raw-completion data.

**Why sample-100BT for FineWeb instead of a specific CC snapshot?** Reproducibility. `sample-100BT` is a deterministic 100B-token subset of FineWeb — anyone who runs the same code gets the same data. Named snapshots also work but are subject to FineWeb re-releases and can drift.

**Why trafilatura over BeautifulSoup for Common Crawl?** trafilatura is specifically designed for main-content extraction from web pages. It handles boilerplate removal (navigation, ads, footers) significantly better than generic HTML parsers.

**Why HTTPS for Common Crawl instead of S3?** Direct S3 access to the `commoncrawl` bucket fails on EC2 instances with IAM roles attached — the role credentials are rejected by the bucket policy. HTTPS via `data.commoncrawl.org` works reliably regardless of instance credentials.

**Why streaming-first code?** At 1b scale with 30B+ tokens, materializing any large source in memory is infeasible on reasonable hardware. FineWeb and stack-v2 require streaming; the other sources use streaming for consistency so the pipeline works uniformly across hardware sizes. RAM is not the load-bearing scaling axis here — vCPU count and network throughput are.

**Why datatrove for dedup instead of datasketch?** datasketch's `MinHashLSH` is in-memory. At 350m it requires ~32GB; at 1b it requires ~85GB and may not fit on a single instance. datatrove's disk-based pipeline keeps RAM bounded by shard size regardless of corpus size — the same pattern used by FineWeb and RedPajama at trillion-token scale.

**Why fasttext over langdetect?** Language detection runs on every Common Crawl document. fasttext's `lid.176.ftz` is C-backed and ~1000× faster than pure-Python `langdetect` at equivalent accuracy, covering 176 languages. The model is ~1MB, downloaded once via `make download-fasttext-model`.

**Why versioned S3 uploads?** Each run uploads to `{target}/{date}/curated/` so multiple runs never overwrite each other. Safe to re-run curation with different parameters and compare results; allows rolling back to a previous run if issues are found during training.

**Why per-stage resumability?** Curation runs take hours to days on spot instances that can be interrupted with 2 minutes notice. Each stage checks for existing output before processing — safe to interrupt and restart without reprocessing completed work.

---

## Scaling Beyond 1b

The pipeline is designed to scale. Scale-invariant mix percentages, streaming-first code, and cap-and-redistribute all generalise to larger targets. To run at 3b or beyond:

1. Add an entry to `TARGET_CONFIGS` in `curate.py` with the new `total_tokens` and `cc_crawls` list.
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

**Char-to-token ratio is approximate.** `CHARS_PER_TOKEN = 5` is a fleet-wide default in `curator/constants.py`. Real ratios vary by domain: English prose ~4.5, code ~3.5, math ~3. The approximation is fine for target sizing but will produce slightly under-target tokens if the corpus skews code-heavy. Recalibrate from a tokenized 125m run if needed.