# Evaluation

This directory provides benchmark evaluation for any completed model branch
and deterministic behavior checks for chat-formatted post-training branches.

## Contents

| File | Purpose |
|---|---|
| `eval.py` | Run lm-evaluation-harness tasks and save structured results |
| `sanity_eval.py` | Check factual, format, repetition, code, and stop behavior |
| `sanity_prompts.jsonl` | Versioned behavior cases and pass/fail rules |

## Checkpoint paths

| Variant | Default checkpoint |
|---|---|
| Base | `$RESULTS_DIR/runs/<size>/pretrain/final` |
| Instruct | `$RESULTS_DIR/runs/<size>/sft_instruct/final` |
| Chat | `$RESULTS_DIR/runs/<size>/dpo_chat/final` |
| Code | `$RESULTS_DIR/runs/<size>/sft_code/final` |

Evaluation reads the tokenizer packaged in the checkpoint's `tokenizer/`
subdirectory, falling back to tokenizer files at the checkpoint root.

## Benchmark evaluation

The benchmark wrapper supports:

| Name | Harness task | Default shots | Reported metric |
|---|---|---:|---|
| `hellaswag` | `hellaswag` | 10 | normalized accuracy |
| `arc_easy` | `arc_easy` | 25 | normalized accuracy |
| `arc_challenge` | `arc_challenge` | 25 | normalized accuracy |
| `mmlu` | `mmlu` | 5 | accuracy |
| `truthfulqa` | `truthfulqa_mc2` | 0 | accuracy |
| `humaneval` | `humaneval` | 0 | pass@1 |

Evaluation and curation share the version-locked benchmark contract in
`config/benchmarks.py`. It pins lm-eval v0.4.9 task definitions and immutable
Hugging Face dataset commits used by exact and 13-word benchmark
decontamination.

Run a complete branch evaluation:

```bash
make eval-base SIZE=125m
make eval-instruct SIZE=125m
make eval-chat SIZE=125m
make eval-code SIZE=125m
```

`make eval` is an alias for `make eval-chat`.

Use a bounded benchmark pass before committing to the full suite:

```bash
python eval/eval.py \
  --model results/runs/125m/dpo_chat/final \
  --tasks quick \
  --limit 50 \
  --batch-size 4
```

Select tasks and precision directly:

```bash
python eval/eval.py \
  --model results/runs/125m/pretrain/final \
  --tasks hellaswag,arc_easy \
  --num-fewshot 0 \
  --device cuda \
  --dtype bfloat16 \
  --log-samples
```

Supported CLI options are `--model`, `--tasks`, `--batch-size`,
`--num-fewshot`, `--device`, `--dtype`, `--limit`, and `--log-samples`.

HumanEval executes generated code in the evaluation process. Use it only in an
isolated environment. The evaluator enables the harness code-execution switch
only when HumanEval is selected.

## Behavior sanity checks

Sanity cases are rendered with the tokenizer's chat template, so use them for
instruct, chat, and code branches:

```bash
make eval-sanity-instruct SIZE=125m
make eval-sanity-chat SIZE=125m
make eval-sanity-code SIZE=125m
```

`make eval-sanity` is an alias for the chat check.

Direct invocation:

```bash
python eval/sanity_eval.py \
  --model results/runs/125m/dpo_chat/final \
  --device cuda \
  --repetition-penalty 1.1 \
  --json-out results/runs/125m/eval/sanity/chat.json
```

Use benchmark evaluation, not chat-formatted sanity prompts, to assess the raw
base checkpoint.

## Outputs and failure behavior

For run-scoped checkpoints, benchmark JSON is saved to:

```text
$RESULTS_DIR/runs/<size>/eval/<variant>/eval_<UTC timestamp>.json
```

Other local model paths use:

```text
$RESULTS_DIR/eval/<model name>/eval_<UTC timestamp>.json
```

Benchmark evaluation exits nonzero if any selected task fails. Sanity
evaluation prints each failed rule, writes JSON when `--json-out` is supplied,
and exits nonzero unless every case passes. These outputs are evidence for a
release decision; export does not run evaluation automatically.
