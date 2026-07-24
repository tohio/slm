# eval

Optional benchmark and sanity evaluation for SLM checkpoints.

Evaluation is not required for export. Exported model cards do not include evaluation results.

---

## Owns

- `eval/eval.py` — lm-evaluation-harness wrapper
- `eval/sanity_eval.py` — behavior sanity checks
- `eval/sanity_prompts.jsonl` — sanity prompt set

---

## Model paths

```text
base      results/runs/<size>/pretrain/final
instruct  results/runs/<size>/sft_instruct/final
chat      results/runs/<size>/dpo_chat/final
code      results/runs/<size>/sft_code/final
```

---

## Benchmark eval

Make targets:

```bash
make eval-base SIZE=125m
make eval-instruct SIZE=125m
make eval-chat SIZE=125m
make eval-code SIZE=125m
make eval SIZE=125m
make eval-mini SIZE=mini
```

`make eval` defaults to the chat variant.

Direct calls:

```bash
python eval/eval.py --model results/runs/125m/dpo_chat/final
python eval/eval.py --model results/runs/125m/pretrain/final --tasks hellaswag,arc_easy --limit 50
```

Common options:

```text
--tasks
--batch-size
--num-fewshot
--device
--limit
--output
```

---

## Sanity eval

Make targets:

```bash
make eval-sanity-base SIZE=125m
make eval-sanity-instruct SIZE=125m
make eval-sanity-chat SIZE=125m
make eval-sanity-code SIZE=125m
make eval-sanity SIZE=125m
```

Direct call:

```bash
python eval/sanity_eval.py   --model results/runs/125m/dpo_chat/final   --json-out results/runs/125m/eval/sanity/chat.json
```

Common options:

```text
--model
--prompts
--device
--repetition-penalty
--json-out
```

---

## Outputs

Eval outputs are written under the run-specific results directory:

```text
results/runs/<size>/eval/
```

Sanity eval writes JSON when `--json-out` is supplied.

---

## Notes

- Benchmark eval is optional and can be noisy for small models.
- Sanity eval is useful for quick behavior checks after SFT/DPO.
- Export does not read eval results into model cards.
