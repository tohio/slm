# Accelerate Configurations

This directory stores process-launch configuration for Hugging Face
Accelerate.

## Checked-in files

| File | Purpose |
|---|---|
| `single_gpu.yaml` | One-process, one-GPU BF16 launch |
| `multi_gpu.yaml` | Multi-process DDP launch |

`config_gen/accel_gen.py` can regenerate `multi_gpu.yaml` and create
`fsdp.yaml`:

```bash
make accel-gen-ddp GPUS=8
make accel-gen-fsdp GPUS=8
```

The normal training Make targets pass process count and precision directly to
`accelerate launch`. A generated topology file affects a run only when it is
selected explicitly:

```bash
.venv/bin/accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  pretrain/train.py \
  --config pretrain/configs/gpt_350m.yaml
```

For interactive Accelerate defaults, copy a checked-in configuration with:

```bash
make accelerate-config-single
make accelerate-config-multi GPUS=8
```

Before using FSDP for an expensive run, verify checkpoint save, resume,
promotion, and export with the same topology. Training hyperparameters remain
in the stage YAML files; these files control process orchestration only.
