# alignment

Preference-alignment stage for SLM. This folder prepares DPO preference data and trains the `chat` variant from the `instruct` checkpoint.

---

## Responsibility

`alignment/` owns:

- DPO preference dataset preparation
- chat-alignment configs
- DPO training
- chat checkpoint output

It does not train the base model, instruct model, or code model.

---

## Lineage

```text
results/runs/<size>/sft_instruct/final
  ↓
DPO training
  ↓
results/runs/<size>/dpo_chat/final
```

The `chat` model is a sibling of the `code` model. Both start from `sft_instruct/final`.

---

## Files

```text
alignment/
├── configs/
│   ├── dpo_chat_mini.yaml
│   ├── dpo_chat_125m.yaml
│   ├── dpo_chat_350m.yaml
│   └── dpo_chat_1b.yaml
├── data/
│   └── prepare_dpo.py
├── train_dpo.py
└── README.md
```

---

## Commands

Prepare DPO data:

```bash
make prepare-dpo SIZE=125m
```

Train chat DPO:

```bash
make dpo-chat SIZE=125m GPUS=1
make dpo-chat-mini SIZE=mini GPUS=1
make dpo-chat-resume SIZE=125m GPUS=1
```

Compatibility aliases are still available:

```bash
make dpo SIZE=125m GPUS=1
make dpo-mini SIZE=mini GPUS=1
make dpo-resume SIZE=125m GPUS=1
```

Direct calls:

```bash
python alignment/data/prepare_dpo.py --size 125m
accelerate launch alignment/train_dpo.py --config alignment/configs/dpo_chat_125m.yaml
```

---

## Inputs

Prepared DPO data expects:

```text
data/runs/<size>/tokenizer/
results/runs/<size>/sft_instruct/final
```

The tokenizer is used for length filtering and formatting. The instruct checkpoint is the policy initialization for DPO.

---

## Outputs

Prepared data and stats are written under the run-specific data directory.

The trained chat checkpoint is written to:

```text
results/runs/<size>/dpo_chat/final
```

---

## Data format

DPO records use the standard preference-pair shape:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "source": "..."
}
```

`chosen` and `rejected` must be different responses to the same prompt.

---

## Configs

DPO configs are size-specific:

```text
alignment/configs/dpo_chat_mini.yaml
alignment/configs/dpo_chat_125m.yaml
alignment/configs/dpo_chat_350m.yaml
alignment/configs/dpo_chat_1b.yaml
```

`make config-gen-dpo SIZE=<size> GPUS=<n>` regenerates the active DPO config for the current GPU profile.

---

## Tests

```bash
make test-dpo-chat SIZE=mini
make test-dpo-chat SIZE=125m
```

Compatibility alias:

```bash
make test-dpo SIZE=125m
```

The test validates preference data shape and that the chat DPO checkpoint loads and generates.
