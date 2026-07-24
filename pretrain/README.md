# pretrain

Base-model pretraining for SLM. This folder owns corpus tokenization, memory-mapped datasets, base training, checkpoint resume, and smoke generation.

---

## Owns

- `pretrain/data/tokenize_data.py` — validated JSONL to tokenized `.bin`
- `pretrain/data/dataset.py` — memory-mapped train/val dataset
- `pretrain/train.py` — Hugging Face Trainer pretraining entry point
- `pretrain/configs/` — size-specific base training configs

Post-training is handled by `finetune/` and `alignment/`.

---

## Configs

```text
pretrain/configs/gpt_smoke.yaml
pretrain/configs/gpt_mini.yaml
pretrain/configs/gpt_125m.yaml
pretrain/configs/gpt_350m.yaml
pretrain/configs/gpt_1b.yaml
```

Generate/update configs for the current hardware:

```bash
make config-gen-pretrain SIZE=125m GPUS=1
make config-gen SIZE=125m GPUS=1
```

---

## Inputs

Tokenization reads:

```text
data/runs/<size>/validated/train.jsonl
data/runs/<size>/validated/val.jsonl
data/runs/<size>/tokenizer/
```

Training reads:

```text
data/runs/<size>/tokenized/train.bin
data/runs/<size>/tokenized/val.bin
data/runs/<size>/tokenized/train.json
data/runs/<size>/tokenized/val.json
```

`train.json` and `val.json` store input SHA-256, tokenizer fingerprint, binary
format version, and realized per-source document/token counts. Tokenization is
ordered and reproducible; stale derived binaries are rebuilt automatically.
The dataset reader requires these sidecars.

---

## Outputs

```text
data/runs/<size>/tokenized/train.bin
data/runs/<size>/tokenized/val.bin
results/runs/<size>/pretrain/checkpoints/
results/runs/<size>/pretrain/final
```

---

## Commands

Tokenize:

```bash
make tokenize SIZE=125m
```

Train:

```bash
make pretrain-mini SIZE=mini GPUS=1
make pretrain SIZE=125m GPUS=1
make pretrain-resume SIZE=125m GPUS=1
```

Smoke generation:

```bash
make pretrain-smoke SIZE=125m
make smoke-gen SIZE=125m
```

Before SFT:

```bash
make reinit-embeds SIZE=125m
```

Direct calls:

```bash
python pretrain/data/tokenize_data.py --size 125m --verify
accelerate launch pretrain/train.py --config pretrain/configs/gpt_125m.yaml
python pretrain/train.py --config pretrain/configs/gpt_125m.yaml --resume
```

---

## Dataset behavior

`dataset.py` memory maps flat token arrays and slices fixed-length windows for causal LM training.

The train/val split is created before tokenization by the curator blend stage. Pretraining does not create its own runtime split.

---

## Checkpoint contract

The final base checkpoint is:

```text
results/runs/<size>/pretrain/final
```

This checkpoint is the parent for instruct SFT.

---

## Tests

```bash
make test-training SIZE=mini
make test-training SIZE=125m
```
