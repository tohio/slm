# Infrastructure

Bootstrap the existing Ubuntu curation/training host roles and verify their
separate dependency stacks. Setup installs system packages, updates `.env` and
shell configuration, and requires appropriate privileges; inspect scripts before
running on an existing multi-purpose host.

## Contents

| File | Purpose |
|---|---|
| `setup_curate.sh` | Curation bootstrap, KenLM build, spaCy model, paths, and verification |
| `setup_train.sh` | GPU bootstrap, training stack, optional source-run restoration |
| `setup_environment.sh` | Internal pip/uv/conda installer shared by both roles |
| `verify_environment.py` | Pinned role versions and optional CUDA checks |
| `gpu_smoke.py` | Dataset-free eager/compiled training and generation acceptance |

## Setup by role

Prepare `.env` from `.env.sample`, selecting the data/results/export paths and
credentials required by the chosen workflow. W&B is required: populate
`WANDB_API_KEY` and `WANDB_PROJECT`, along with `HF_TOKEN` and the common paths.
Storage settings are validated when selected transfers execute; `HF_USERNAME`
is needed only for model publication. Use separate checkouts/environments
for the two host roles; curation and training require different Transformers/Hub
versions.

```bash
make setup-curate DATA_DIR=/data/slm/data
make setup-train DATA_DIR=/data/slm/data
```

`setup-curate` installs `requirements-curation.txt`, including KenLM build and
orjson/FastText handling; `setup-train` installs the complete GPU and evaluation
stack from `requirements-training.txt`. Both include the common
`requirements.txt`. No separate GPU, evaluation, or HF-transfer requirements
file is needed.

### Installer selection

Both commands accept `INSTALLER=pip|uv|conda`; pip/venv is the default. Install
uv or conda first when selecting it. All three place the environment in the
checkout's `.venv`, so existing pipeline commands use the same Python path.
Conda creates a Python 3.12 prefix and installs the role requirements with pip;
it does not select a separate conda CUDA build. uv uses the same pinned CUDA
wheel/index as pip.

```bash
make setup-curate INSTALLER=uv DATA_DIR=/data/slm/data
make setup-train INSTALLER=conda DATA_DIR=/data/slm/data
```

These illustrate alternative hosts, not two roles to layer into one `.venv`.
Activate a pip/uv environment with `source .venv/bin/activate`; for conda use
`conda activate "$PWD/.venv"`. Switching between conda and venv requires moving
an existing environment aside explicitly; setup does not silently delete it.

## Curation assets

Setup installs Python dependencies, `wget` for asset downloads, and KenLM Python
bindings from pinned revision `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5`.
It verifies the curation package contract; a pinned source does not substitute
for testing the build on the target host. The language-ID and perplexity model
**assets** remain explicit downloads:

```bash
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model DATA_DIR=/data/slm/data
```

Curation's internal prerequisite gate checks those three model files, `.env`,
and pinned curation versions before source processing. Missing assets produce
instructions and stop curation. Dataset access terms and credentials remain the
operator's responsibility.

## Training and optional restoration

Training setup requires an already working NVIDIA driver (`nvidia-smi`), then
checks the NVIDIA/CUDA/BF16 contract and installs the pinned GPU stack. It does
not install a driver or provide a CPU-only training installation. Model tests
can execute on CPU after this environment has been installed. With no source
run selected it performs setup only. With a source run selected it restores
the requested artifacts; missing credentials fail the transfer rather than
silently skipping it:

```bash
make setup-train SIZE=mini DATASET_SIZE=350m DATASET_RUN_ID=350m-YYYYMMDD-abcdef \
  DATA_DIR=/data/slm/data ARTIFACT_BACKEND=hf
```

The setup restore default is `tokenized,tokenizer,metadata`. Select additional
stages explicitly, for example `ARTIFACT_STAGES=validated,tokenized,tokenizer,metadata`
for final test-document completions. The source tokenizer stays under
`runs/<dataset-size>/tokenizer`; model outputs remain under the model size.

The pinned stack requires CUDA runtime 13.0, driver 580.65.06 or newer, native
GPU architecture support in the installed PyTorch wheel, and BF16 support.
Internal training checks run again at relevant entry points. For diagnosis:

```bash
.venv/bin/python infra/verify_environment.py --profile training --require-cuda
make test-gpu-gate
```

The GPU gate uses a tiny model, not a production corpus or checkpoint. Passing it
is not proof of model convergence or production throughput.

`train-all` already invokes this setup/restore flow. Use it directly for a new
end-to-end training run, or use setup followed by individual stages—not both.

## Configure before launching

```bash
make config-gen SIZE=mini GPUS=1
```

For multiple GPUs substitute the chosen count in `GPUS=N`. Configuration includes
the required DDP launch file; no interactive/global Accelerate setup is needed.
See [config generation](../config_gen/README.md), [artifact routing](../docs/PRETRAINING_DATA.md),
and [training](../docs/TRAIN.md). Serving retains its separate vLLM runtime contract.
