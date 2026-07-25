# Hardware recommendations

Practical hardware guidance for running the SLM pipeline. These are operational recommendations, not minimum theoretical requirements.

---

## CPU curation

| Size | Recommended CPU | Recommended RAM | Notes |
|---|---:|---:|---|
| `mini` | 4+ vCPU | 16 GB+ | pipeline validation |
| `125m` | 64+ vCPU | 256–384 GB | practical full run |
| `350m` | 64–96+ vCPU | 384 GB+ | larger curation run |
| `1b` | 96+ vCPU | 512 GB+ | largest supported run |

Use persistent storage for `DATA_DIR` and run long jobs inside `tmux`.

---

## Storage

Use a separate persistent disk for full runs.

Recommended starting points:

| Run | Storage |
|---|---:|
| `mini` | boot disk is fine |
| `125m` | 1 TB+ |
| `350m` | 1–2 TB+ |
| `1b` | 2 TB+ |

See [Disk setup](DISK_SETUP.md) for mounting a secondary disk at `/data`.

---

## GPU training

| Size | Practical GPU recommendation | Notes |
|---|---|---|
| `mini` | 1× 16 GB+ GPU | loop validation |
| `125m` | 1× A100 80GB / H100 / H200, or better | practical full run |
| `350m` | 1–4× A100 80GB / H100 / H200, or better | use multi-GPU when available |
| `1b` | 4–8× A100 80GB / H100 / H200, or better | prefer FSDP/multi-GPU |

---

## Config generation

Generate configs for the actual GPU count before training:

```bash
make accelerate-config-single
make config-gen SIZE=125m GPUS=1
```

For multi-GPU:

```bash
make accelerate-config-multi GPUS=4
make config-gen SIZE=125m GPUS=4
```

For larger runs with FSDP:

```bash
make accel-gen-fsdp GPUS=8
make config-gen SIZE=1b GPUS=8
```

Use the same `GPUS` value for Accelerate setup, config generation, and training.

---

## Batch scaling

The training pipeline uses data parallelism. Each GPU holds a model replica and the batch is split across GPUs.

Preserve global batch when changing GPU count:

```text
global_batch = micro_batch_size × gradient_accumulation_steps × num_gpus
```

Use `make config-gen` instead of hand-editing configs whenever possible.

---

## Resume targets

```bash
make pretrain-resume SIZE=125m GPUS=1
make sft-instruct-resume SIZE=125m GPUS=1
make sft-code-resume SIZE=125m GPUS=1
make dpo-chat-resume SIZE=125m GPUS=1
```

---

## Monitoring

```bash
watch -n 2 nvidia-smi
nvtop
```

Detailed command sequences are in the [command reference](COMMANDS.md).

## See Also

- [Architecture](ARCHITECTURE.md)
- [Disk setup](DISK_SETUP.md)
- [Command reference](COMMANDS.md)
