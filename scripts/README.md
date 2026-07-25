# scripts

## Purpose

Stage-neutral utility and diagnostic scripts.

Stage-owned scripts should stay in their stage folder. This folder is for helpers used across stages or diagnostics that intentionally sit outside the main pipeline.

## How It Fits In

These utilities diagnose or support multiple stages without owning a pipeline
artifact; see [Architecture](../docs/ARCHITECTURE.md).

## Files

```text
scripts/
├── pretrain_hf_125m.py
├── reinit_special_embeds.py
├── run_lm_eval.py
├── sanity_train.py
├── sft_model_comparison.py
└── README.md
```

---

## `sanity_train.py`

Known-good training diagnostic using a simple Hugging Face data path. Use it to separate model/trainer issues from curation-data issues.

Make targets:

```bash
make sanity-train
make sanity-train-small
make sanity-train-tiny
make sanity-train-save SANITY_SIZE=tiny
```

Direct examples:

```bash
python scripts/sanity_train.py --arch 125m --target-tokens 2500000000
python scripts/sanity_train.py --arch mini --target-tokens 50000000 --save
```

Useful options:

```text
--arch
--target-tokens
--batch-size
--lr
--warmup-steps
--log-every
--eval-every
--scratch-dir
--save
--reuse-tokens
```

---

## `reinit_special_embeds.py`

Reinitializes chat-template special-token embeddings before SFT.

Normal use:

```bash
make reinit-embeds SIZE=125m
```

Direct use:

```bash
python scripts/reinit_special_embeds.py --size 125m
python scripts/reinit_special_embeds.py \
  --src results/runs/125m/pretrain/checkpoint-152000 \
  --dst results/runs/125m/pretrain/final
```

Options:

```text
--size
--src
--dst
--no-backup
```

Default input/output:

```text
input/output  results/runs/<size>/pretrain/final
```

The script uses direct safetensors I/O and avoids `SLMForCausalLM.from_pretrained()` / `save_pretrained()` for checkpoint safety.

---

## `run_lm_eval.py`

Wrapper for lm-evaluation-harness with the local SLM architecture pre-registered.

Example:

```bash
python scripts/run_lm_eval.py \
  --model hf \
  --model_args pretrained=results/runs/125m/sft_code/final,dtype=bfloat16 \
  --tasks humaneval \
  --num_fewshot 0 \
  --batch_size 1 \
  --apply_chat_template \
  --output_path results/eval/debug_humaneval \
  --log_samples \
  --limit 5
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

## `sft_model_comparison.py`

Fail-fast comparison of SmolLM2-135M and a native SLM-125M export. It checks
checkpoint integrity, prompt sensitivity, and cached/uncached parity before an
optional controlled completion-only SFT run.

```bash
make export-base-local SIZE=125m
make compare-sft-preflight
make compare-sft
```

The harness uses `trust_remote_code=False` for both models. It owns tokenization
and labels so both trainers receive the same selected record identities without
silent preprocessing drops. Tokenizer-specific token totals are reported.

---

## Adding scripts

Scripts in this folder should be:

- stage-neutral diagnostics
- helpers used across multiple stages
- one-off recovery utilities with clear comments

Stage-specific production scripts belong in the owning stage folder.
