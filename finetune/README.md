# finetune

Supervised fine-tuning stages for SLM. This folder owns instruct SFT, code SFT, response-control examples, and raw code-completion SFT.

---

## Owns

- `finetune/data/prepare_sft.py` — SmolTalk/instruct and Magicoder/code SFT datasets
- `finetune/data/response_control.py` — local response-control examples
- `finetune/data/prepare_code_completion.py` — raw body-completion examples
- `finetune/train_sft.py` — instruct and code SFT trainer
- `finetune/train_code_completion.py` — raw code-completion trainer
- `finetune/configs/` — SFT and code-completion configs

DPO chat alignment lives in `alignment/`.

---

## Configs

Instruct SFT:

```text
finetune/configs/sft_instruct_mini.yaml
finetune/configs/sft_instruct_125m.yaml
finetune/configs/sft_instruct_350m.yaml
finetune/configs/sft_instruct_1b.yaml
```

Code SFT:

```text
finetune/configs/sft_code_mini.yaml
finetune/configs/sft_code_125m.yaml
finetune/configs/sft_code_350m.yaml
finetune/configs/sft_code_1b.yaml
```

Raw code-completion SFT:

```text
finetune/configs/code_completion_125m.yaml
finetune/configs/code_completion_350m.yaml
finetune/configs/code_completion_1b.yaml
```

Generate/update SFT configs:

```bash
make config-gen-sft SIZE=125m GPUS=1
```

---

## Inputs

Instruct SFT starts from:

```text
results/runs/<size>/pretrain/final
data/runs/<size>/tokenizer/
```

Code SFT starts from:

```text
results/runs/<size>/sft_instruct/final
data/runs/<size>/tokenizer/
```

Code-completion SFT starts from:

```text
results/runs/<size>/sft_code/final
data/runs/<size>/code_completion/
```

---

## Outputs

Prepared data:

```text
data/runs/<size>/sft_instruct/train.jsonl
data/runs/<size>/sft_instruct/val.jsonl
data/runs/<size>/sft_code/train.jsonl
data/runs/<size>/sft_code/val.jsonl
data/runs/<size>/code_completion/train.jsonl
data/runs/<size>/code_completion/val.jsonl
```

Checkpoints:

```text
results/runs/<size>/sft_instruct/final
results/runs/<size>/sft_code/final
results/runs/<size>/sft_code_completion/final
```

---

## Commands

Prepare instruct and code SFT data:

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

Raw code-completion path:

```bash
make prepare-code-completion SIZE=125m
make sft-code-completion SIZE=125m
make eval-code-completion SIZE=125m
```

Direct calls:

```bash
python finetune/data/prepare_sft.py --size 125m --stage both
python finetune/data/prepare_sft.py --size 125m --stage instruct
python finetune/data/prepare_sft.py --size 125m --stage code
accelerate launch finetune/train_sft.py --config finetune/configs/sft_instruct_125m.yaml
accelerate launch finetune/train_sft.py --config finetune/configs/sft_code_125m.yaml
```

Compatibility aliases:

```bash
make sft SIZE=125m GPUS=1
make sft-mini SIZE=mini GPUS=1
```

---

## Instruct SFT

`prepare_sft.py` uses a size-aware SmolTalk policy:

```text
mini   tiny smol-smoltalk subset
125m   50% of HuggingFaceTB/smol-smoltalk
350m   full HuggingFaceTB/smol-smoltalk
1b     full HuggingFaceTB/smoltalk
```

Local response-control examples are appended to the instruct data.

---

## Code SFT

Code SFT uses `ise-uiuc/Magicoder-OSS-Instruct-75K` and keeps examples whose assistant response contains real code. Prose-only and explanation-only examples are filtered or normalized.

---

## Chat template and loss masking

`train_sft.py` expects conversational records. TRL applies the tokenizer chat template and uses assistant-only loss masking. The tokenizer chat template must include generation markers around assistant responses.

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
