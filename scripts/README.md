# scripts/

Standalone utilities and diagnostics that complement the main pipeline.

Unlike the staged pipeline in `curator/`, `pretrain/`, `finetune/`, and
`alignment/`, scripts here are independent — they don't feed into or
depend on pipeline state, and can be run at any time.

## Contents

### `sanity_train.py`

A self-contained diagnostic that validates the model architecture and
training loop against a known-good reference setup. It trains either the
mini or the 125m architecture on
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
tokenized with
[Mistral's tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.1),
bypassing the SLM curator and tokenizer entirely. This lets you confirm
that training works end-to-end before attributing any issues to data or
tokenization.

Typical use: run before a full pretraining run when you've made changes to
the model, training loop, or dependencies, and want confidence the core
components still learn as expected.

#### Sizes (timings on H200)

| Target                 | Architecture       | Tokens | Time    | Use case                                |
|------------------------|--------------------|--------|---------|-----------------------------------------|
| `sanity-train`         | 125m (~125M params)| 2.5B   | ~90 min | Full diagnostic before a real run       |
| `sanity-train-small`   | mini (~22M params) | 500M   | ~12 min | Quick check; matches what 125m needs    |
| `sanity-train-tiny`    | mini (~22M params) | 50M    | ~2 min  | Smoke test; "did I just break the loop" |

#### Running

```bash
make sanity-train          # 125m, 2.5B tokens
make sanity-train-small    # mini, 500M tokens
make sanity-train-tiny     # mini, 50M tokens

# Save the trained model — defaults to 125m
make sanity-train-save
make sanity-train-save SANITY_SIZE=small
make sanity-train-save SANITY_SIZE=tiny
```

By default, no model is saved — only the training log, final eval loss,
and QA probe results are printed. With `--save` (or via
`sanity-train-save`), the model is written to `results/sanity-<arch>/`.

#### What success looks like

A healthy run produces:

- **Training loss** that decreases steadily (e.g., 10.5 → 3–4 over the run).
- **Eval loss** that tracks training loss (gap < 0.5 means the model is
  generalizing, not memorizing).
- **QA probes** (capital of France, 2+2, etc.) where the model assigns
  lower loss to the correct continuation than to a wrong one. A score of
  ≥3/5 is meaningful at this scale.
- **Generation samples** that produce real English fragments — varied
  vocabulary, grammatical structure — even if sentence-level meaning is
  weak.

If training loss and eval loss diverge sharply, or the QA probes score
0–1, the issue is in the model code or training loop. If they pass,
those components are fine and any failure on the curated pipeline is
elsewhere (data, tokenizer, or downstream stages).

#### GPU sizing

Defaults are tuned for an H200 (141GB HBM3e), where batch size 16 fits
comfortably for both architectures. For smaller GPUs:

- **H100 (80GB)**: drop `--batch-size` to 8 for the 125m arch
- **A100 (40GB)**: drop to 4 for 125m, 8 for mini
- **Smaller**: pass `--batch-size` explicitly to the script

```bash
.venv/bin/python scripts/sanity_train.py --arch 125m --batch-size 8
```

## Adding new scripts

Scripts in this directory should be:

- **Self-contained** — minimal assumptions about pipeline state
- **Documented** — include a short description in this README
- **Stable** — if a script is experimental or temporary, keep it in a
  personal branch rather than committing to `main`

When a script becomes a stable part of a pipeline stage, migrate it into
the relevant stage directory (for example, `curator/scripts/` or
`pretrain/data/`).