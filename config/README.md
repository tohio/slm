# Shared Configuration

This directory defines repository-wide data, path, and PyTorch runtime
contracts. Stage code should import these values instead of duplicating them.

## Contents

| File | Purpose |
|---|---|
| `data_mix.py` | Pretraining source mix, code sub-mix, curation limits, split policy, and per-size token targets |
| `paths.py` | Run-scoped data, result, evaluation, and export path builders |
| `holdout.py` | Split identity and train/val/test integrity |
| `chat.py` | No-tool/tool-aware templates, rendering fingerprints, web_search contract |
| `checkpoints.py` | Bundled tokenizer resolution, checkpoint hashes, immutable run inputs and recovery-file checks |
| `provenance.py` | Portable final-checkpoint ancestry and measured source-corpus history |
| `schedule.py` | Shared SFT/DPO fractional-epoch update/warmup calculations |
| `benchmarks.py` | Immutable benchmark dataset revisions shared by curation and evaluation |
| `runtime.py` | Safe CUDA matmul and SDPA dispatcher configuration |
| `__init__.py` | Public configuration exports |

## Data configuration

`data_mix.py` is the source of truth for:

- top-level pretraining source percentages;
- allocation within the code share;
- fixed and supplemental source caps;
- cross-source deduplication priority;
- the overflow source;
- mini/125M/350M/1B corpus targets and epochs;
- curation limits and validation split fraction; and
- mini-profile source caps.

Validate the module's internal invariants:

```bash
python -m config.data_mix
```

Print the active targets without copying values into another file:

```bash
python - <<'PY'
from config.data_mix import TARGET_CONFIGS, consumed_tokens

for size, settings in TARGET_CONFIGS.items():
    print(
        size,
        "corpus_tokens=", settings["corpus_tokens"],
        "epochs=", settings["epochs"],
        "consumed_tokens=", consumed_tokens(size),
    )
PY
```

Changing this contract invalidates affected curation manifests and may change
generated pretraining step counts. Start a new run ID rather than mixing
artifacts from different data contracts.

## Paths

`paths.py` loads `.env` and resolves three roots:

| Variable | Default | Owns |
|---|---|---|
| `DATA_DIR` | `data` | corpora, tokenizer, prepared SFT/DPO data, metadata |
| `RESULTS_DIR` | `results` | training checkpoints and evaluation results |
| `EXPORTS_DIR` | `$RESULTS_DIR/exports` | native model packages |

All model-size data is scoped under `$DATA_DIR/runs/<size>/`; all training
results are scoped under `$RESULTS_DIR/runs/<size>/`.

Set path environment variables before importing `config.paths`. Prefer its
helper functions over string concatenation so local, mounted, and restored
runs share one layout.

## Checkpoint and run identity

`checkpoints.py` is shared by pretraining, instruct/code SFT, DPO, and export.
It never repairs learned weights or substitutes a tokenizer by model-size label.
Post-training binds config/weights, data, tokenizer/rendering, and process count
before accepting a resume; DPO also binds the fixed reference policy. Existing
audits are compared before any write and remain immutable. See
[training identity](../docs/TRAIN.md#post-training-identity-and-resume).

## CUDA runtime policy

`configure_torch_runtime()` runs only when CUDA is available. It enables local
Inductor FX-graph and AOTAutograd caches by default and stores compiler artifacts
under `$TORCHINDUCTOR_CACHE_DIR` when explicitly set, otherwise
`$DATA_DIR/cache/torchinductor`. Keeping `DATA_DIR` on persistent storage lets
compatible compile artifacts survive Python-process and instance restarts; PyTorch
validates cache compatibility before reuse. Explicit cache environment variables
are never overwritten. It also enables high
float32 matmul precision and sets automatic SDPA dispatch as the default for
reference/evaluation paths. Its log reports policy, not the selected kernel.

`GroupedQueryAttention` overrides that default locally for SLM CUDA BF16/FP16
forwards in training mode: only Flash, memory-efficient, and cuDNN SDPA are
allowed. Unsupported fused inputs raise instead of silently using math. The
scope is inside the compiled attention call and restores previous backend
flags on exit, including failure. CPU, FP32 diagnostics, and eval-mode
inference keep normal dispatch. Native Llama is not modified by this policy.

Training entry points call the runtime helper; utilities that construct a
standalone CUDA model should do the same.

## Validation

```bash
python -m pytest \
  tests/test_data_config.py \
  tests/test_misc_contract.py \
  -q
```
