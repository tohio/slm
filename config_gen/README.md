# Configuration Generation

This directory generates hardware-specific training recipes and Accelerate
process-launch configurations.

These are separate configuration layers:

- `config_gen.py` writes model/trainer YAML: batch size, accumulation,
  checkpointing, steps, warmup, and stage recipe fields.
- `accel_gen.py` writes the internal DDP launch configuration when multiple GPUs are selected.

## Training recipes

Generate every recipe for a size:

```bash
make config-gen SIZE=125m GPUS=1
```

Mini uses the same generator flow as 125M/350M/1B, including instruct SFT, code
SFT, and DPO. Its bounded post-training sample/step budgets are preserved while
micro-batch and accumulation adapt to hardware. Smoke remains pretraining-only.

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
pretraining, the generated `max_steps` and warmup are planning values derived
from the configured corpus target, epochs, sequence length, and effective
global batch. Functional mini and production pretraining replace them from
verified tokenized train tokens while preserving the generated warmup ratio.

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

`config-gen` and the stage-specific generators invoke DDP config generation
internally when `GPUS > 1`. Training explicitly selects
`accelerate_configs/multi_gpu.yaml`; single-GPU launches use `single_gpu.yaml`.
No interactive/global Accelerate configuration or second sharded-training path
is needed. Use the same `GPUS` value for configuration and launch. Each DDP GPU
must hold a full model/optimizer replica.

## Direct usage

```bash
python -m config_gen.config_gen \
  --stage pretrain --gpu h200 --size 125m --gpus 1 --mode balanced \
  --output pretrain/configs/gpt_125m.yaml
```

Direct model-YAML generation does not generate the launch config; use the Make
flow for the normal complete configuration. Run the module's `--help` for GPU
identifiers and output options.

## Validation

```bash
make test-config-gen
make test-accel-gen
```

In generic examples, replace `N` with the GPU count you choose to use. Mini
uses the existing training config-generation flow; Smoke stays separate.
