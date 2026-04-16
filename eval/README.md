# eval

Benchmark evaluation using `lm-evaluation-harness`. Evaluates trained SLM checkpoints on standard LLM benchmarks and writes results to JSON.

---

## Benchmarks

| Benchmark | Task | Metric | Few-shot | Measures |
|---|---|---|---|---|
| HellaSwag | `hellaswag` | `acc_norm` | 10 | Commonsense reasoning |
| ARC-Easy | `arc_easy` | `acc_norm` | 25 | Science QA (easy) |
| ARC-Challenge | `arc_challenge` | `acc_norm` | 25 | Science QA (hard) |
| MMLU | `mmlu` | `acc` | 5 | Broad knowledge (57 subjects) |
| TruthfulQA | `truthfulqa_mc2` | `acc` | 0 | Factual accuracy |
| HumanEval | `humaneval` | `pass@1` | 0 | Python code generation |

---

## Usage

```bash
# Evaluate all benchmarks
make eval SIZE=125m

# Or directly
python eval/eval.py --model results/slm-125m-dpo/final

# Quick subset (hellaswag + arc only — faster)
python eval/eval.py --model results/slm-125m-dpo/final --tasks quick

# Specific tasks
python eval/eval.py --model results/slm-125m-dpo/final --tasks hellaswag,arc_easy,mmlu

# Limit examples (smoke test)
python eval/eval.py --model results/slm-125m-dpo/final --tasks quick --limit 100

# Override few-shot count for all tasks
python eval/eval.py --model results/slm-125m-dpo/final --num-fewshot 0

# Compare base vs aligned
python eval/eval.py --model results/slm-125m/final       --tasks all
python eval/eval.py --model results/slm-125m-dpo/final   --tasks all
```

---

## Results

Results are saved to `results/eval/<model_name>/eval_<timestamp>.json`:

```
results/eval/
├── slm-125m/
│   └── eval_20260415_120000.json
├── slm-125m-chat-code/
│   └── eval_20260415_130000.json
└── slm-125m-dpo/
    └── eval_20260415_140000.json
```

The `export.py` script reads the most recent result file for the `slm-{size}-dpo` model and embeds the scores in the Hub model card automatically. Run `make eval` before `make export-chat`.

---

## Tokenizer

The evaluator looks for the tokenizer at `<model_path>/tokenizer/` — the subdirectory written by `train.py`, `train_sft.py`, and `train_dpo.py`. If that directory is missing, it falls back to the model root. A clear error is raised if `tokenizer_config.json` cannot be found in either location.

---

## Expected Performance

Approximate expected scores at convergence — **rough targets, not guarantees**:

| Benchmark | Random | GPT-2 (117M) | slm-125m target |
|---|---|---|---|
| HellaSwag | 25% | ~31% | ~35–40% |
| ARC-Easy | 25% | ~44% | ~45–55% |
| ARC-Challenge | 25% | ~26% | ~28–35% |
| MMLU | 25% | ~26% | ~27–32% |
| TruthfulQA | — | ~40% | ~35–45% |

Performance scales with model size — 350M and 1B should outperform 125M significantly on MMLU and ARC-Challenge.