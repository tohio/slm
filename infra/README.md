# Infrastructure

This directory bootstraps supported Ubuntu hosts and validates the pinned CPU
or NVIDIA training environment.

## Contents

| File | Purpose |
|---|---|
| `setup.sh` | Install a data-processing/development host and create persistent paths |
| `setup_gpu_instance.sh` | Install the GPU stack, restore one run, and activate its tokenizer |
| `verify_environment.py` | Check pinned package versions and optional CUDA requirements |
| `gpu_smoke.py` | Dataset-free eager/compiled training and generation acceptance test |

The setup scripts install packages, create a virtual environment, update
`.env`, and modify shell configuration. Read them before running on an
existing multi-purpose host.

## Data-processing host

On a fresh Ubuntu 22.04 host:

```bash
cp .env.sample .env
make setup-data-dir DATA_DIR=/data/slm/data
source .venv/bin/activate
```

Download the models used by language identification and perplexity validation:

```bash
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model DATA_DIR=/data/slm/data
```

`setup.sh` installs the full development/curation dependency set, KenLM
bindings, and the spaCy English model. It creates run-scoped directories but
does not accept dataset terms or supply credentials.

## GPU training host

Populate `.env` with the artifact-store and Hugging Face credentials, then
restore one complete curation/tokenizer run:

```bash
make setup-gpu \
  DATA_DIR=/data/slm/data \
  SIZE=125m \
  RUN_ID=125m-YYYYMMDD-abcdef

source .venv/bin/activate
```

The GPU setup:

1. requires an NVIDIA driver new enough for the pinned CUDA 13.0 runtime;
2. installs `requirements-gpu.txt`;
3. verifies package versions, CUDA, native SM support, and BF16;
4. restores `tokenized`, `tokenizer`, and `metadata` artifacts for the exact
   run ID; and
5. synchronizes that tokenizer into the runtime tokenizer directory.

It does not restore curated/validated JSONL by default because pretraining
reads the tokenized arrays.

## Environment verification

Check pinned Python packages on any installed host:

```bash
.venv/bin/python infra/verify_environment.py
```

Require the complete NVIDIA contract:

```bash
.venv/bin/python infra/verify_environment.py --require-cuda
```

The CUDA check requires runtime 13.0, driver `580.65.06` or newer, a PyTorch
wheel containing the detected compute capability, and BF16 support.

## Dataset-free GPU acceptance

Run after every GPU-image or dependency upgrade:

```bash
make test-gpu-gate
```

The smoke test creates a tiny SLM model and verifies eager BF16 optimization,
compiled optimization, finite gradients/losses, peak-memory reporting, and
cached/uncached greedy-generation parity. It does not download data or load a
trained checkpoint.

Passing this gate establishes environment compatibility, not throughput or
full-run convergence. Generate recipes for the actual GPU next:

```bash
make config-gen SIZE=125m GPUS=1
```

## Operational notes

- Use persistent storage for `$DATA_DIR`, `$RESULTS_DIR`, and `$EXPORTS_DIR`.
- Keep a curation `RUN_ID` with the artifacts restored to a training host.
- Re-run environment and GPU acceptance after changing any pinned framework,
  CUDA wheel, driver, or GPU type.
- vLLM uses its own runtime/image contract; validate serving separately.
