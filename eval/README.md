# eval

Benchmark evaluation using `lm-evaluation-harness`. Evaluates trained SLM checkpoints on standard LLM benchmarks and writes results to JSON.

Target library version: **lm-eval >= 0.4.2**.

---

## Benchmarks

| Benchmark | Task | Metric | Filter | Few-shot | Measures |
|---|---|---|---|---|---|
| HellaSwag | `hellaswag` | `acc_norm` | `none` | 10 | Commonsense reasoning |
| ARC-Easy | `arc_easy` | `acc_norm` | `none` | 25 | Science QA (easy) |
| ARC-Challenge | `arc_challenge` | `acc_norm` | `none` | 25 | Science QA (hard) |
| MMLU | `mmlu` | `acc` | `none` | 5 | Broad knowledge (57 subjects) |
| TruthfulQA | `truthfulqa_mc2` | `acc` | `none` | 0 | Factual accuracy |
| HumanEval | `humaneval` | `pass@1` | `create_test` | 0 | Python code generation |

lm-eval 0.4.x keys result dicts as `"{metric},{filter}"`. `eval.py` handles all current filter variants (`none`, `create_test`, `strict-match`, `flexible-extract`) plus a catch-all for unknowns, so adding a new benchmark doesn't require touching the result-parsing code.

---

## Sanity Evaluation

In addition to benchmark evaluation, this directory includes a lightweight sanity eval for post-training behavior:

| File | Purpose |
|---|---|
| `sanity_prompts.jsonl` | Fixed prompts covering factual QA, AI/ML concepts, code generation, function completion, code explanation, factual restraint, and stop behavior |
| `sanity_eval.py` | Deterministic generation runner that checks required text, forbidden text, repetition, answer length for simple prompts, and clean task formatting |

The sanity eval is not a replacement for benchmark evaluation. It is a regression gate for assistant behavior that benchmark scores may not capture.

It checks whether the model can:
- Answer simple factual questions without unsupported elaboration.
- Choose the correct domain for ambiguous terms.
- Provide appropriate-depth explanations.
- Produce code when code is requested.
- Explain code when explanation is requested.
- Avoid obvious repetition.
- Stop once the answer is complete.

Run against a local checkpoint:

```bash
python eval/sanity_eval.py \
  --model results/slm-125m-dpo/final \
  --json-out results/eval/sanity/slm-125m-dpo.json
```

---

## Sanity Checks

The sanity eval is a deterministic behavior regression check. It is not a benchmark score and does not replace lm-eval. It catches failures that benchmark aggregates may hide.

| Category | Checks |
|---|---|
| Simple factual QA | Direct answer, no unsupported elaboration |
| AI/ML concepts | Correct domain interpretation, no electrical-transformer confusion |
| Code generation | Produces code when code is requested |
| Function completion | Returns function body when requested |
| Code explanation | Explains code instead of rewriting it |
| Factual restraint | Avoids inventing private or unverifiable facts |
| Stop behavior | Stops after a complete answer |
| Repetition | Rejects obvious repeated-token or repeated-word runs |

---

## Usage

```bash
# Evaluate all standard benchmarks
make eval SIZE=125m

# Run behavior sanity eval
make eval-sanity SIZE=125m

# Or directly: standard benchmark eval
python eval/eval.py --model results/slm-125m-dpo/final

# Or directly: behavior sanity eval
python eval/sanity_eval.py --model results/slm-125m-dpo/final

# Quick benchmark subset (hellaswag + arc_easy + arc_challenge — faster, no code exec)
python eval/eval.py --model results/slm-125m-dpo/final --tasks quick

# Specific benchmark tasks
python eval/eval.py --model results/slm-125m-dpo/final --tasks hellaswag,arc_easy,mmlu

# Limit benchmark examples (smoke test)
python eval/eval.py --model results/slm-125m-dpo/final --tasks quick --limit 100

# Override benchmark few-shot count for all tasks
python eval/eval.py --model results/slm-125m-dpo/final --num-fewshot 0

# Override benchmark precision (default bfloat16)
python eval/eval.py --model results/slm-125m-dpo/final --dtype float16

# Debug a weird benchmark result by saving per-example inputs/outputs
python eval/eval.py --model results/slm-125m-dpo/final --tasks mmlu --log-samples

# Save sanity eval results
python eval/sanity_eval.py \
  --model results/slm-125m-dpo/final \
  --json-out results/eval/sanity/slm-125m-dpo.json

# Run sanity eval against the exported Hub model
python eval/sanity_eval.py \
  --model tohio/slm-125m-chat \
  --trust-remote-code

# Compare base vs aligned on standard benchmarks
python eval/eval.py --model results/slm-125m/final       --tasks all
python eval/eval.py --model results/slm-125m-dpo/final   --tasks all

# Compare base vs aligned on sanity behavior
python eval/sanity_eval.py --model results/slm-125m/final
python eval/sanity_eval.py --model results/slm-125m-dpo/final
```

`eval.py` exits with status 1 if any task fails, so it composes correctly with `make` and CI chains.

---

## Precision

Eval loads the model in **bfloat16** by default, matching training precision. Override with `--dtype float16` or `--dtype float32` if needed. Running eval in float32 on a bf16-trained model roughly doubles memory and can OOM at the 1B size with default batch size.

---

## Code execution (HumanEval)

HumanEval runs model-generated Python code locally to test correctness. `eval.py` sets `HF_ALLOW_CODE_EVAL=1` and passes `confirm_run_unsafe_code=True` to lm-eval **only when a code-executing task is in the run** — so `--tasks quick` or `--tasks mmlu` will not enable code execution.

If your environment prohibits executing untrusted code, skip HumanEval with `--tasks hellaswag,arc_easy,arc_challenge,mmlu,truthfulqa`.

---

## Results

Results are saved to `results/eval/<model_name>/eval_<UTC_timestamp>Z.json`:

```
results/eval/
├── slm-125m/
│   └── eval_20260415_120000Z.json
├── slm-125m-chat-code/
│   └── eval_20260415_130000Z.json
└── slm-125m-dpo/
    └── eval_20260415_140000Z.json
```

Sanity eval results can be written to `results/eval/sanity/`:

```text
results/eval/sanity/
├── slm-125m-dpo.json
└── tohio-slm-125m-chat.json
```

Timestamps are UTC (suffix `Z`) so standard benchmark filenames sort consistently across machines in different timezones.

Each standard benchmark JSON file includes:
- `model`, `model_name`, `tasks`, `failed_tasks` — what was run
- `dtype` — precision used for the run
- `timestamp` — UTC timestamp
- `results` — per-task scores from lm-eval
- `groups` — aggregated scores for group-tasks like MMLU
- `config` — lm-eval's run config

Each sanity eval JSON file includes:
- `model` — local checkpoint path or Hub model ID
- `passed`, `total` — aggregate pass count
- `results` — per-prompt output and failure reasons

The `export.py` script reads the most recent result file for the `slm-{size}-dpo` model and embeds the scores in the Hub model card automatically. Run `make eval` before `make export-chat`.

---

## Tokenizer

The evaluator looks for the tokenizer at `<model_path>/tokenizer/` — the subdirectory written by `train.py`, `train_sft.py`, and `train_dpo.py`. If that directory is missing, it falls back to the model root. A clear error is raised if `tokenizer_config.json` cannot be found in either location.

---

## Expected Performance

Approximate expected scores at convergence — **rough targets, not guarantees**, and highly dependent on the exact eval harness version and prompt formatting:

| Benchmark | Random | slm-125m target |
|---|---|---|
| HellaSwag | 25% | ~35–40% |
| ARC-Easy | 25% | ~45–55% |
| ARC-Challenge | 25% | ~28–35% |
| MMLU | 25% | ~27–32% |
| TruthfulQA | — | ~35–45% |

Performance scales with model size — 350M and 1B should outperform 125M significantly on MMLU and ARC-Challenge. Baseline numbers for comparable open models are best pulled fresh from the Open LLM Leaderboard rather than memorised here.
