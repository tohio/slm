# scripts/

Standalone utilities and diagnostics that complement the main pipeline.
Unlike the staged pipeline in `curator/`, `pretrain/`, `finetune/`, and
`alignment/`, scripts here are independent — they don't feed into or
depend on pipeline state, and can be run at any time.

> **Note:** `config_gen/config_gen.py` and `config_gen/accel_gen.py` are also standalone
> utilities but live alongside the model code rather than here, because
> they produce artifacts the pipeline consumes (training configs and
> accelerate launch configs). Scripts in this directory don't produce
> artifacts other stages rely on.

## Contents

The sections below follow a consistent template:

- **What it does** — one-line summary
- **When to use it**
- **How to run** — Make target + direct invocation
- **What success looks like** — how to know the script worked
- **GPU sizing notes** (where applicable)

---

### `sanity_train.py`

**What it does.** Trains either a mini (~22M params) or 125m (~125M params)
architecture on
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
tokenized with
[Mistral's tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.1).
Bypasses the SLM curator and tokenizer entirely.

**When to use it.** Run before a full pretraining run when you've made
changes to the model code, training loop, or dependencies, and want
confidence the core components still learn as expected against a
known-good reference setup. If `sanity_train.py` learns and the curated
pipeline doesn't, the issue is in your data or tokenizer — not the model.

**How to run.**

```bash
make sanity-train          # 125m, 2.5B tokens
make sanity-train-small    # mini, 500M tokens
make sanity-train-tiny     # mini, 50M tokens

# Save the trained model — defaults to 125m
make sanity-train-save
make sanity-train-save SANITY_SIZE=small
make sanity-train-save SANITY_SIZE=tiny

# Direct invocation (custom batch size for non-H200 GPUs)
.venv/bin/python scripts/sanity_train.py --arch 125m --batch-size 8
```

By default, no model is saved — only the training log, final eval loss,
and QA probe results are printed. With `--save` (or via
`sanity-train-save`), the model is written to `results/sanity-<arch>/`.

**Sizes (timings on H200):**

| Target               | Architecture       | Tokens | Time    | Use case                                |
|----------------------|--------------------|--------|---------|-----------------------------------------|
| `sanity-train`       | 125m (~125M params)| 2.5B   | ~90 min | Full diagnostic before a real run       |
| `sanity-train-small` | mini (~22M params) | 500M   | ~12 min | Quick check; matches what 125m needs    |
| `sanity-train-tiny`  | mini (~22M params) | 50M    | ~2 min  | Smoke test; "did I just break the loop" |

**What success looks like.** A healthy run produces:

- **Training loss** that decreases steadily (e.g., 10.5 → 3–4 over the run).
- **Eval loss** that tracks training loss (gap < 0.5 means the model is
  generalizing, not memorizing).
- **QA probes** (capital of France, 2+2, etc.) where the model assigns
  lower loss to the correct continuation than to a wrong one. A score of
  ≥3/5 is meaningful at this scale.
- **Generation samples** that produce real English fragments — varied
  vocabulary, grammatical structure — even if sentence-level meaning is
  weak.

If training and eval loss diverge sharply, or the QA probes score 0–1,
the issue is in the model code or training loop. If they pass, those
components are fine and any failure on the curated pipeline is elsewhere
(data, tokenizer, or downstream stages).

**GPU sizing notes.** Defaults are tuned for an H200 (141GB HBM3e), where
batch size 16 fits comfortably for both architectures. For smaller GPUs:

- **H100 (80GB):** drop `--batch-size` to 8 for the 125m arch
- **A100 (40GB):** drop to 4 for 125m, 8 for mini
- **Smaller:** pass `--batch-size` explicitly to the script

Note that `sanity_train.py` does **not** use `config_gen/config_gen.py`. The
sanity script ships its own minimal config tuned for a fixed reference
setup — the whole point is to be a known-good baseline, so its config
is held constant rather than auto-tuned per GPU. For real pretraining
runs, use `make config-gen-pretrain` to size the config to your
hardware.

---

### `run_lm_eval.py`

**What it does.** Wraps the `lm-evaluation-harness` CLI with the SLM
architecture pre-registered via `AutoConfig.register("slm", SLMConfig)`
and `AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)`. Hands
off to `lm_eval.__main__.cli_evaluate` so all standard `lm_eval` flags
work unchanged.

**When to use it.** Use this for ad-hoc `lm_eval` CLI invocations against
SLM checkpoints — running a single task with `--limit`, inspecting
generations with `--log_samples`, testing chat-formatted prompts with
`--apply_chat_template`, or any other lm_eval flag combination not
exposed by `eval/eval.py`.

For standard benchmark runs, prefer `make eval-base`, `make eval-instruct`,
or `make eval-chat` — those use `eval/eval.py` and are sufficient for
producing the canonical eval JSONs. Reach for this wrapper when you need
the full surface area of the lm_eval CLI directly.

> **Why a wrapper is needed.** SLM is a custom architecture. lm_eval calls
> `AutoConfig.from_pretrained` internally before model construction, which
> raises `KeyError: 'slm'` unless SLM has been registered with the Auto*
> registries first. Running `python -m lm_eval` directly against an SLM
> checkpoint will fail; this wrapper handles the registration so the rest
> of the CLI works transparently.

**How to run.**

```bash
# Standard invocation — all lm_eval CLI flags work
.venv/bin/python scripts/run_lm_eval.py \
  --model hf \
  --model_args pretrained=results/slm-125m-chat-code/final,dtype=bfloat16 \
  --tasks humaneval \
  --num_fewshot 0 \
  --batch_size 1 \
  --apply_chat_template \
  --output_path results/eval/debug_humaneval \
  --log_samples \
  --limit 5

# Full eval suite against the DPO checkpoint
.venv/bin/python scripts/run_lm_eval.py \
  --model hf \
  --model_args pretrained=results/slm-125m-dpo/final,dtype=bfloat16 \
  --tasks hellaswag,arc_easy,arc_challenge,mmlu,truthfulqa,humaneval \
  --batch_size 1 \
  --apply_chat_template \
  --output_path results/eval/slm-125m-dpo-cli \
  --log_samples
```

The wrapper accepts every flag `python -m lm_eval` accepts. Anything you
would pass to lm_eval directly, pass to this script the same way.

**What success looks like.** The script loads the model without
`KeyError: 'slm'`, runs the requested tasks to completion, and writes
results to `--output_path`. With `--log_samples`, per-example generations
are saved alongside results — useful for debugging tasks like HumanEval
where the score alone doesn't tell you whether the model produced
plausible code or chat-formatted prose.

A pass@1 of 0.0 on HumanEval is not a wrapper failure — it's the model.
The wrapper succeeds when the CLI runs to completion and produces
inspectable output, regardless of the score.

**GPU sizing notes.** Same as your training/eval defaults. The wrapper
adds no memory overhead beyond what lm_eval itself uses. Pass
`--batch_size` smaller if you OOM on a smaller GPU.

---

## Adding new scripts

Each new script gets a section in this README following the template at
the top — what / when / how / success / GPU sizing.

Scripts in this directory should be:

- **Self-contained** — minimal assumptions about pipeline state
- **Documented** — section in this README
- **Stable** — if a script is experimental or temporary, keep it in a
  personal branch rather than committing to `main`

When a script becomes a stable part of a pipeline stage, migrate it into
the relevant stage directory (for example, `curator/scripts/` or
`pretrain/data/`).