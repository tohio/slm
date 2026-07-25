# Pretraining

This directory converts the validated corpus to memory-mapped token arrays and
trains a base SLM model from scratch.

## Contents

| Path | Purpose |
|---|---|
| `data/tokenize_data.py` | Encode validated JSONL with the run tokenizer |
| `data/dataset.py` | Read fixed-length windows from memory-mapped token arrays |
| `train.py` | Build the model, run causal-language-model training, resume, and save |
| `configs/gpt_*.yaml` | Size-specific model and optimization recipes |

Post-training branches are owned by `finetune/` and `alignment/`.

## Data contract

Tokenization reads:

```text
$DATA_DIR/runs/<size>/validated/train.jsonl
$DATA_DIR/runs/<size>/validated/val.jsonl
$DATA_DIR/runs/<size>/tokenizer/slm_tokenizer.json
```

It writes:

```text
$DATA_DIR/runs/<size>/tokenized/train.bin
$DATA_DIR/runs/<size>/tokenized/train.json
$DATA_DIR/runs/<size>/tokenized/val.bin
$DATA_DIR/runs/<size>/tokenized/val.json
```

The JSON sidecars record the input digest, tokenizer fingerprint, binary
format, and realized source/document/token counts. Training requires those
sidecars and rejects a tokenizer that does not match the tokenized corpus.
The train/validation split is created by curation; pretraining does not split
documents again.

`dataset.py` memory maps the flat token arrays and returns fixed-length causal
language-model windows without loading the entire corpus into memory.

## Configuration

Available profiles:

```text
pretrain/configs/gpt_smoke.yaml
pretrain/configs/gpt_mini.yaml
pretrain/configs/gpt_125m.yaml
pretrain/configs/gpt_350m.yaml
pretrain/configs/gpt_1b.yaml
```

Generate the selected recipe on the target training host:

```bash
make config-gen-pretrain SIZE=125m GPUS=1
```

To generate pretraining, SFT, and DPO recipes together:

```bash
make config-gen SIZE=125m GPUS=1
```

Configuration generation uses the selected model size, GPU type/count, VRAM
policy, and effective-batch targets. Inspect the resulting YAML before
starting an expensive run.

## Usage

Encode and verify the corpus:

```bash
make tokenize SIZE=125m
```

Equivalent direct command:

```bash
python pretrain/data/tokenize_data.py \
  --size 125m \
  --chunk-size 256 \
  --verify
```

Before training, activate the tokenizer associated with this size and data
run:

```bash
make restore-size-tokenizer SIZE=125m
```

Start or resume pretraining:

```bash
make pretrain SIZE=125m GPUS=1
make pretrain-resume SIZE=125m GPUS=1
```

The bounded architecture/data rehearsal uses the mini recipe:

```bash
make pretrain-mini SIZE=mini GPUS=1
```

Direct invocation:

```bash
accelerate launch \
  pretrain/train.py \
  --config pretrain/configs/gpt_125m.yaml

accelerate launch \
  pretrain/train.py \
  --config pretrain/configs/gpt_125m.yaml \
  --resume
```

## Outputs and resume

Checkpoints and the promoted base model are stored at:

```text
$RESULTS_DIR/runs/<size>/pretrain/checkpoints/
$RESULTS_DIR/runs/<size>/pretrain/final/
```

`--resume` selects the latest compatible checkpoint in the run directory.
The final checkpoint is the parent of instruct SFT.

Before using chat/control tokens in post-training, reinitialize their embedding
rows in the final base checkpoint:

```bash
make reinit-embeds SIZE=125m
```

Run the base generation smoke check after a completed pretraining run:

```bash
make smoke-gen SIZE=125m
```

## Validation

Run the architecture and training-contract tests before a paid run:

```bash
make test-model
make test-training-args
```

Validate a completed pretraining artifact:

```bash
make test-training SIZE=125m
```

The artifact test expects an existing checkpoint; it is not a substitute for
pretraining and does not launch another full run.
