# finetune

Supervised fine-tuning stages for SLM. This folder trains the `instruct`, `code`, and optional raw code-completion variants.

---

## Responsibility

`finetune/` owns:

- SFT dataset preparation
- response-control data generation
- instruct SFT
- code SFT
- optional raw code-completion SFT

DPO chat alignment lives in `alignment/`.

---

## Lineage

```text
results/runs/<size>/pretrain/final
  ↓
results/runs/<size>/sft_instruct/final
  ├── results/runs/<size>/dpo_chat/final
  └── results/runs/<size>/sft_code/final
```

`dpo_chat` is trained in `alignment/`. `sft_code` is trained here.

---

## Files

```text
finetune/
├── configs/
│   ├── sft_instruct_mini.yaml
│   ├── sft_instruct_125m.yaml
│   ├── sft_instruct_350m.yaml
│   ├── sft_instruct_1b.yaml
│   ├── sft_code_mini.yaml
│   ├── sft_code_125m.yaml
│   ├── sft_code_350m.yaml
│   ├── sft_code_1b.yaml
│   ├── code_completion_125m.yaml
│   ├── code_completion_350m.yaml
│   └── code_completion_1b.yaml
├── data/
│   ├── prepare_sft.py
│   ├── prepare_code_completion.py
│   └── response_control.py
├── train_sft.py
├── train_code_completion.py
└── README.md
```

---

## Commands

Prepare SFT data:

```bash
make prepare-sft SIZE=125m
```

Train instruct SFT:

```bash
make sft-instruct SIZE=125m GPUS=1
make sft-instruct-mini SIZE=mini GPUS=1
make sft-instruct-resume SIZE=125m GPUS=1
```

Train code SFT:

```bash
make sft-code SIZE=125m GPUS=1
make sft-code-mini SIZE=mini GPUS=1
make sft-code-resume SIZE=125m GPUS=1
```

Compatibility aliases:

```bash
make sft SIZE=125m GPUS=1
make sft-mini SIZE=mini GPUS=1
```

Raw code-completion path:

```bash
make prepare-code-completion SIZE=125m
make sft-code-completion SIZE=125m
make eval-code-completion SIZE=125m
```

---

## Inputs

Instruct SFT starts from:

```text
results/runs/<size>/pretrain/final
```

Code SFT starts from:

```text
results/runs/<size>/sft_instruct/final
```

SFT data preparation uses the tokenizer under:

```text
data/runs/<size>/tokenizer/
```

---

## Outputs

```text
results/runs/<size>/sft_instruct/final
results/runs/<size>/sft_code/final
results/runs/<size>/sft_code_completion/final
```

---

## Instruct SFT

Instruct SFT combines a SmolTalk backbone with the local `response_control` data.

`response_control` reinforces:

- concise direct answers
- arithmetic
- simple factual answers
- factual restraint
- concept definitions
- response-format control
- clean stopping behavior

---

## Code SFT

Code SFT uses the Magicoder-style instruction data path plus local code examples. It reinforces:

- simple Python generation
- function completion
- write-code versus explain-code distinction
- code-specific instruction following

---

## Chat template

SFT examples are formatted using the project tokenizer chat template and special tokens. The template must remain consistent across SFT, DPO, inference, and serving.

---

## Tests

```bash
make test-sft-instruct SIZE=mini
make test-sft-code SIZE=mini

make test-sft-instruct SIZE=125m
make test-sft-code SIZE=125m
```

Compatibility alias:

```bash
make test-sft-chat SIZE=125m
```
