# scripts

Utility and diagnostic scripts that sit outside a single pipeline stage.

---

## Files

```text
scripts/
├── pretrain_hf_125m.py
├── reinit_special_embeds.py
├── run_lm_eval.py
├── sanity_train.py
└── README.md
```

---

## `sanity_train.py`

Known-good training diagnostic using a simple Hugging Face data path. Use it to separate model/trainer issues from curation-data issues.

Examples:

```bash
make sanity-train
make sanity-train-small
make sanity-train-tiny
make sanity-train-save SANITY_SIZE=tiny
```

Direct:

```bash
python scripts/sanity_train.py --arch 125m --target-tokens 2500000000
python scripts/sanity_train.py --arch mini --target-tokens 50000000 --save
```

---

## `run_lm_eval.py`

Wrapper for lm-evaluation-harness with the custom SLM architecture.

Example:

```bash
python scripts/run_lm_eval.py   --model results/runs/125m/dpo_chat/final   --tasks hellaswag,arc_easy   --batch-size 4
```

Prefer the Makefile eval targets for normal use:

```bash
make eval-base SIZE=125m
make eval-instruct SIZE=125m
make eval-chat SIZE=125m
make eval-code SIZE=125m
```

---

## `pretrain_hf_125m.py`

Diagnostic pretraining path that uses a Hugging Face dataset with the 125m architecture. It is useful for validating trainer behavior independently from the full curation pipeline.

---

## `reinit_special_embeds.py`

Reinitializes chat/special-token embeddings before SFT.

Normal use:

```bash
make reinit-embeds SIZE=125m
```

Direct use:

```bash
python scripts/reinit_special_embeds.py --size 125m
```

Expected input/output:

```text
input   results/runs/<size>/pretrain/final
output  patched checkpoint ready for SFT
```

---

## Adding scripts

Scripts in this folder should be:

- stage-neutral diagnostics, or
- helpers used across multiple stages, or
- one-off recovery utilities with clear comments

Stage-owned scripts should stay inside their stage folder.
