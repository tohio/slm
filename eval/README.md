# eval

Evaluation stage for SLM. This folder runs benchmark evaluation and behavior sanity checks against the base, instruct, chat, and code variants.

---

## Responsibility

`eval/` owns:

- lm-evaluation-harness wrapper
- behavior sanity evaluation
- per-variant eval output files
- sanity prompt definitions

---

## Files

```text
eval/
├── eval.py
├── sanity_eval.py
├── sanity_prompts.jsonl
└── README.md
```

---

## Model paths

```text
base      results/runs/<size>/pretrain/final
instruct  results/runs/<size>/sft_instruct/final
chat      results/runs/<size>/dpo_chat/final
code      results/runs/<size>/sft_code/final
```

---

## Commands

Benchmark eval:

```bash
make eval-base     SIZE=125m
make eval-instruct SIZE=125m
make eval-chat     SIZE=125m
make eval-code     SIZE=125m
make eval          SIZE=125m
```

Sanity eval:

```bash
make eval-sanity-base     SIZE=125m
make eval-sanity-instruct SIZE=125m
make eval-sanity-chat     SIZE=125m
make eval-sanity-code     SIZE=125m
make eval-sanity          SIZE=125m
```

Mini eval:

```bash
make eval-mini SIZE=mini
```

Direct benchmark call:

```bash
python eval/eval.py --model results/runs/125m/dpo_chat/final
python eval/eval.py --model results/runs/125m/pretrain/final --tasks hellaswag,arc_easy --limit 50
```

Direct sanity call:

```bash
python eval/sanity_eval.py   --model results/runs/125m/dpo_chat/final   --json-out results/runs/125m/eval/sanity/chat.json
```

---

## Benchmarks

Configured benchmark set:

| Benchmark | Measures |
|---|---|
| HellaSwag | commonsense reasoning |
| ARC-Easy / ARC-Challenge | science QA |
| MMLU | broad knowledge |
| TruthfulQA | factual accuracy |
| HumanEval | Python code generation |
| MBPP | basic Python programming |

---

## Sanity evaluation

Sanity prompts test behavior that benchmark scores may miss:

- direct arithmetic
- exact-answer questions
- factual restraint
- current-information restraint
- code generation
- repetition avoidance
- clean stopping

Sanity eval is deterministic and writes JSON for later inspection.

---

## Outputs

Evaluation outputs are written under the run-specific results directory:

```text
results/runs/<size>/eval/
```

Export reads these outputs when building model cards.
