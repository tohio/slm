# Hardware Recommendations

Practical hardware guidance for running the SLM pipeline. These are operational recommendations, not minimum theoretical requirements.

---

## Token targets

| Size | Curation target | Epochs | Consumed target |
|---|---:|---:|---:|
| `125m` | 10B | 2 | 20B |
| `350m` | 25B | 2 | 50B |
| `1b` | 75B | 1 | 75B |

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

## GPU training

| Size | Practical GPU recommendation | Notes |
|---|---|---|
| `mini` | 1× 16 GB+ GPU | loop validation |
| `125m` | 1× A100 80GB / H100 / H200, or better | practical full run |
| `350m` | 1–4× A100 80GB / H100 / H200, or better | use multi-GPU when available |
| `1b` | 4–8× A100 80GB / H100 / H200, or better | prefer FSDP/multi-GPU |

---

## Pipeline validation run

```bash
make setup-gpu DATA_DIR=/data/slm/data SIZE=mini RUN_ID=<mini-run-id>
make accelerate-config-single

make pretrain-mini SIZE=mini GPUS=1
make reinit-embeds SIZE=mini
make prepare-sft SIZE=mini
make sft-instruct-mini SIZE=mini GPUS=1
make sft-code-mini SIZE=mini GPUS=1
make prepare-dpo SIZE=mini
make dpo-chat-mini SIZE=mini GPUS=1
make eval-mini SIZE=mini
```

---

## Full 125m run

```bash
make setup-gpu DATA_DIR=/data/slm/data SIZE=125m RUN_ID=<125m-run-id>
make accelerate-config-single
make config-gen SIZE=125m GPUS=1

make pretrain SIZE=125m GPUS=1
make reinit-embeds SIZE=125m
make prepare-sft SIZE=125m
make sft-instruct SIZE=125m GPUS=1
make sft-code SIZE=125m GPUS=1
make prepare-dpo SIZE=125m
make dpo-chat SIZE=125m GPUS=1
```

For multi-GPU:

```bash
make accelerate-config-multi GPUS=4
make config-gen SIZE=125m GPUS=4
make pretrain SIZE=125m GPUS=4
```

---

## Data parallelism

The training pipeline uses data parallelism. Each GPU holds a model replica and the batch is split across GPUs.

Preserve global batch when changing GPU count:

```text
global_batch = micro_batch_size × gradient_accumulation_steps × num_gpus
```

Use `make config-gen` instead of hand-editing configs whenever possible.

---

## FSDP

For larger runs, especially `1b`, use FSDP config generation:

```bash
make accel-gen-fsdp GPUS=8
make config-gen SIZE=1b GPUS=8
```

---

## Resume

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
