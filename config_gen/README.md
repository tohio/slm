# Configuration Generation

This directory generates hardware-specific training recipes and Accelerate
process-launch configurations.

These are separate configuration layers:

- `config_gen.py` writes model/trainer YAML: batch size, accumulation,
  checkpointing, steps, warmup, and stage recipe fields.
- `accel_gen.py` writes process topology: DDP or FSDP, process count, precision,
  and sharding settings.

## Training recipes

Generate every recipe for a size:

```bash
make config-gen SIZE=125m GPUS=1
```

Generate one stage:

```bash
make config-gen-pretrain SIZE=125m GPUS=1
make config-gen-sft SIZE=125m GPUS=1
make config-gen-dpo SIZE=125m GPUS=1
```

Automatic GPU detection uses `nvidia-smi`. Select a known profile explicitly
when generating off-host or when detection is ambiguous:

```bash
make config-gen SIZE=350m GPUS=4 GPU=h200 MODE=conservative
```

Supported planning modes:

| Mode | Planned VRAM use | Intent |
|---|---:|---|
| `conservative` | 70% | maximize headroom |
| `balanced` | 80% | default |
| `aggressive` | 90% | maximize micro-batch after validation |

The generator computes hardware-dependent fields while preserving
stage-specific objective and optimizer fields from its profiles. For
pretraining, `max_steps` and warmup are derived from the configured corpus
target, epochs, sequence length, and effective global batch.

Outputs:

```text
pretrain/configs/gpt_<size>.yaml
finetune/configs/sft_instruct_<size>.yaml
finetune/configs/sft_code_<size>.yaml
alignment/configs/dpo_chat_<size>.yaml
```

Inspect generated YAML before starting a paid run. GPU-memory estimates are a
planning model, not a measured guarantee.

## Accelerate topology

Generate a DDP or FSDP configuration:

```bash
make accel-gen-ddp GPUS=8
make accel-gen-fsdp GPUS=8
```

Outputs:

```text
accelerate_configs/multi_gpu.yaml
accelerate_configs/fsdp.yaml
```

The standard stage Make targets launch Accelerate with explicit process and
precision flags; they do not select a generated FSDP file. To use a generated
topology, pass it explicitly:

```bash
.venv/bin/accelerate launch \
  --config_file accelerate_configs/fsdp.yaml \
  pretrain/train.py \
  --config pretrain/configs/gpt_1b.yaml
```

Use DDP when each GPU can hold a full model/optimizer replica. Use FSDP only
after validating sharded checkpoint save/resume and export on the target
cluster.

## Direct usage

```bash
python -m config_gen.config_gen \
  --stage pretrain \
  --gpu h200 \
  --size 125m \
  --gpus 1 \
  --mode balanced \
  --output pretrain/configs/gpt_125m.yaml

python -m config_gen.accel_gen \
  --strategy fsdp \
  --gpus 8 \
  --sharding-strategy FULL_SHARD
```

Use `--help` on either module for all supported GPU identifiers and output
options.

## Validation

```bash
make test-config-gen
make test-accel-gen
```
