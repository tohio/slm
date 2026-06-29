# alignment

DPO preference alignment for the SLM chat variant.

---

## Owns

- `alignment/data/prepare_dpo.py` — DPO preference dataset preparation
- `alignment/train_dpo.py` — DPO training with TRL
- `alignment/configs/` — size-specific chat DPO configs

This folder trains `dpo_chat` only. Base, instruct, and code training live in other folders.

---

## Configs

```text
alignment/configs/dpo_chat_mini.yaml
alignment/configs/dpo_chat_125m.yaml
alignment/configs/dpo_chat_350m.yaml
alignment/configs/dpo_chat_1b.yaml
```

Generate/update DPO configs:

```bash
make config-gen-dpo SIZE=125m GPUS=1
```

---

## Inputs

```text
results/runs/<size>/sft_instruct/final
data/runs/<size>/tokenizer/
```

Prepared DPO data:

```text
data/runs/<size>/dpo_chat/train.jsonl
data/runs/<size>/dpo_chat/val.jsonl
```

---

## Outputs

```text
results/runs/<size>/dpo_chat/final
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

Direct calls:

```bash
python alignment/data/prepare_dpo.py --size 125m --source all
python alignment/data/prepare_dpo.py --size 125m --source ultrafeedback
python alignment/data/prepare_dpo.py --size 125m --source handcrafted
accelerate launch alignment/train_dpo.py --config alignment/configs/dpo_chat_125m.yaml
python alignment/train_dpo.py --config alignment/configs/dpo_chat_125m.yaml --resume
```

Compatibility aliases:

```bash
make dpo SIZE=125m GPUS=1
make dpo-mini SIZE=mini GPUS=1
make dpo-resume SIZE=125m GPUS=1
```

---

## Preference data

`prepare_dpo.py` writes conversational preference records for TRL `DPOTrainer`:

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "chosen": [
    {"role": "assistant", "content": "preferred response"}
  ],
  "rejected": [
    {"role": "assistant", "content": "rejected response"}
  ],
  "source": "ultrafeedback_binarized"
}
```

Sources:

```text
HuggingFaceH4/ultrafeedback_binarized
handcrafted_behavior
targeted_behavior
```

The CLI exposes `--source all`, `--source ultrafeedback`, and `--source handcrafted`.

---

## Length filtering

`prepare_dpo.py` filters pairs with the actual SLM tokenizer before training. The default ceiling is based on the smallest DPO context budget so the same prepared dataset can be reused across model sizes.

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
