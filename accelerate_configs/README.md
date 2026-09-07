# Accelerate configurations

Repository-local process-launch configuration used by the training Make targets.
Model/trainer hyperparameters belong in the stage YAML files, not here.

| File | Purpose |
|---|---|
| `single_gpu.yaml` | One-process, one-GPU BF16 launch |
| `multi_gpu.yaml` | DDP, regenerated internally for the requested process count |

Run the regular config generator before launching training:

```bash
make config-gen SIZE=mini GPUS=1
```

For multiple GPUs use `GPUS=N`, replacing `N` with the number you choose. The
stage-specific config generators also generate the DDP file when `GPUS > 1`.
Training explicitly selects the appropriate repository config and passes the
same process count. No global Accelerate config, interactive setup, sed-based
configuration, or sharded training path is used.

Each DDP process holds a complete model/optimizer replica. See
[configuration generation](../config_gen/README.md) for hardware planning and
[training](../docs/TRAIN.md) for the launch workflow. Smoke remains separate.
