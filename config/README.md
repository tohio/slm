# Shared Configuration

This directory defines repository-wide data, path, and PyTorch runtime
contracts. Stage code should import these values instead of duplicating them.

## Contents

| File | Purpose |
|---|---|
| `data_mix.py` | Pretraining source mix, code sub-mix, curation limits, split policy, and per-size token targets |
| `paths.py` | Run-scoped data, result, evaluation, and export path builders |
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

## CUDA runtime policy

`configure_torch_runtime()` runs only when CUDA is available. It enables high
float32 matmul precision and leaves Flash, memory-efficient, cuDNN, and math
SDPA implementations available to PyTorch's dispatcher. Keeping the math
fallback enabled prevents an unsupported fast-kernel shape from becoming a
hard failure.

Training entry points call this helper; utilities that construct a standalone
CUDA model should do the same.

## Validation

```bash
python -m pytest \
  tests/test_data_config.py \
  tests/test_misc_contract.py \
  -q
```
