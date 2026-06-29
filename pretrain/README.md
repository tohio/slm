# pretrain

Pretraining stage for SLM. This folder tokenizes validated text into binary train/val files and trains the base decoder-only model from scratch.

---

## Responsibility

`pretrain/` owns:

- tokenized dataset construction
- pretraining configs
- base model training
- checkpoint resume
- smoke generation after pretraining

Post-training lives in `finetune/` and `alignment/`.

---

## Files

```text
pretrain/
├── configs/
│   ├── gpt_mini.yaml
│   ├── gpt_smoke.yaml
│   ├── gpt_125m.yaml
│   ├── gpt_350m.yaml
│   └── gpt_1b.yaml
├── data/
│   ├── dataset.py
│   ├── tokenize_data.py
│   └── upload_tokenized.py
├── train.py
└── README.md
```

---

## Inputs

Tokenization reads validated JSONL splits and the trained tokenizer:

```text
data/runs/<size>/validated/train.jsonl
data/runs/<size>/validated/val.jsonl
data/runs/<size>/tokenizer/tokenizer.json
data/runs/<size>/tokenizer/tokenizer_config.json
```

Pretraining reads tokenized binaries:

```text
data/runs/<size>/tokenized/train.bin
data/runs/<size>/tokenized/val.bin
```

---

## Commands

Tokenize:

```bash
make tokenize SIZE=125m
```

Generate configs:

```bash
make config-gen-pretrain SIZE=125m GPUS=1
make config-gen SIZE=125m GPUS=1
```

Train:

```bash
make pretrain-mini SIZE=mini GPUS=1
make pretrain SIZE=125m GPUS=1
make pretrain-resume SIZE=125m GPUS=1
```

Smoke generation:

```bash
make smoke-gen SIZE=125m
```

Before post-training:

```bash
make reinit-embeds SIZE=125m
```

---

## Outputs

```text
results/runs/<size>/pretrain/checkpoints/
results/runs/<size>/pretrain/final
```

The base export target publishes:

```text
tohio/slm-<size>
```

---

## Config generation

`make config-gen-pretrain` tunes pretraining config fields for the current GPU and GPU count.

```bash
make config-gen-pretrain SIZE=125m GPUS=4 GPU=h200
make config-gen-pretrain SIZE=1b GPUS=8 GPU=b200 MODE=aggressive
```

Accelerate config must match the GPU count:

```bash
make accelerate-config-single
make accelerate-config-multi GPUS=4
make accel-gen-fsdp GPUS=8
```

---

## Tokenized data format

`tokenize_data.py` writes contiguous token ID arrays to `.bin` files. Training uses memory-mapped access so large datasets do not need to fit in RAM.

Expected files:

```text
train.bin
val.bin
```

---

## Val split

The train/val split is produced before tokenization by the curator blend stage. Tokenization processes both splits independently.

---

## Tests

```bash
make test-training SIZE=mini
make test-training SIZE=125m
```

`test-training` validates that the model loads, loss is finite, dataset indexing works, and the checkpoint path is usable.
