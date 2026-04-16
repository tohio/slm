# Hardware Recommendations

This guide helps you choose the right GPU configuration for each pipeline stage. The goal is **efficient resource use** — picking the smallest hardware that completes the job reasonably, not the largest hardware available.

The SLM pipeline uses pure data parallelism throughout — the model is replicated on each GPU and the batch is split across GPUs. No tensor parallelism or model parallelism is needed at any model size. You can adjust `GPUS=N` to match your budget and timeline without any code changes.

---

## Token Budgets

Token targets are set in `curator/scripts/curate.py` and drive everything downstream:

| Model | Curated tokens | Training epochs | Rationale |
|---|---|---|---|
| `slm-125m` | 5B | 2 | ~2× Chinchilla optimal |
| `slm-350m` | 15B | 2 | ~2× Chinchilla optimal |
| `slm-1b` | 30B | 2 | ~1.5× Chinchilla optimal |

---

## Three-Run Training Plan

We recommend validating the pipeline in three progressively larger runs before committing to the full 1B training.

---

### **Run 1: Pipeline Validation**

Prove the entire pipeline works end-to-end before committing to long runs.

| Aspect | Value |
|---|---|
| **Model** | 125M parameters |
| **Stage** | Mini config — pipeline validation only |
| **Hardware** | 1× any GPU (8GB+ VRAM) |
| **Pre-training duration** | ~5–10 min |
| **Full pipeline duration** | ~15–30 min |
| **Estimated cost** | <$1 |

```bash
make pretrain-mini  GPUS=1
make prepare-sft
make sft-mini       GPUS=1
make sft-code-mini  GPUS=1
make prepare-dpo
make dpo-mini       GPUS=1
make eval-mini
```

This uses the mini configs (500 steps each) — enough to confirm loss decreases and the full pipeline runs without errors. Not for producing a usable model.

---

### **Run 2: 125M Full Training**

Train a complete 125M model on the full 5B token dataset.

**Option A: 4× A100 40GB (Cost-conscious)**

| Aspect | Value |
|---|---|
| **Model** | 125M parameters |
| **Tokens** | 5B (2 epochs) |
| **Hardware** | 4× A100 40GB |
| **Pre-training duration** | ~2–3 hours |
| **Full pipeline duration** | ~4–5 hours |
| **Estimated cost** | ~$15–25 |

```bash
make accelerate-config-multi GPUS=4
make pretrain  SIZE=125m GPUS=4
make sft       SIZE=125m GPUS=4
make sft-code  SIZE=125m GPUS=4
make dpo       SIZE=125m GPUS=4
make eval      SIZE=125m
make export    SIZE=125m
```

Remember to adjust `gradient_accumulation_steps` in the configs for 4 GPUs — see the scaling comment in `pretrain/configs/gpt_125m.yaml`.

**Option B: 2× H100 80GB (Speed-focused)**

| Aspect | Value |
|---|---|
| **Model** | 125M parameters |
| **Tokens** | 5B (2 epochs) |
| **Hardware** | 2× H100 80GB |
| **Pre-training duration** | ~1–1.5 hours |
| **Full pipeline duration** | ~2–3 hours |
| **Estimated cost** | ~$15–20 |

```bash
make accelerate-config-multi GPUS=2
make pretrain  SIZE=125m GPUS=2
make sft       SIZE=125m GPUS=2
make sft-code  SIZE=125m GPUS=2
make dpo       SIZE=125m GPUS=2
make eval      SIZE=125m
make export    SIZE=125m
```

**Which to choose?**
- **4× A100:** Better if you want to demonstrate multi-GPU scaling on standard hardware
- **2× H100:** Better if you want faster iteration at similar cost

---

### **Run 3: 350M Full Training**

Scale to a larger model on the full 15B token dataset.

**Option A: 4× H100 80GB (Balanced)**

| Aspect | Value |
|---|---|
| **Model** | 350M parameters |
| **Tokens** | 15B (2 epochs) |
| **Hardware** | 4× H100 80GB |
| **Pre-training duration** | ~6–8 hours |
| **Full pipeline duration** | ~8–10 hours |
| **Estimated cost** | ~$80–120 |

```bash
make accelerate-config-multi GPUS=4
make pretrain  SIZE=350m GPUS=4
make sft       SIZE=350m GPUS=4
make sft-code  SIZE=350m GPUS=4
make dpo       SIZE=350m GPUS=4
make eval      SIZE=350m
make export    SIZE=350m
```

**Option B: 8× H100 80GB (Faster)**

| Aspect | Value |
|---|---|
| **Model** | 350M parameters |
| **Tokens** | 15B (2 epochs) |
| **Hardware** | 8× H100 80GB |
| **Pre-training duration** | ~3–4 hours |
| **Full pipeline duration** | ~4–6 hours |
| **Estimated cost** | ~$80–110 |

```bash
make accelerate-config-multi GPUS=8
make pretrain  SIZE=350m GPUS=8
make sft       SIZE=350m GPUS=8
make sft-code  SIZE=350m GPUS=8
make dpo       SIZE=350m GPUS=8
make eval      SIZE=350m
make export    SIZE=350m
```

**Which to choose?**
- **4× H100:** Good balance — near-linear speedup, lower cost
- **8× H100:** ~1.8× faster than 4× at ~2× the cost — diminishing returns start here

---

### **Run 4: 1B Full Training**

The flagship model — 30B tokens, full pipeline.

**Option A: 8× H100 80GB (Recommended)**

| Aspect | Value |
|---|---|
| **Model** | 1B parameters |
| **Tokens** | 30B (2 epochs) |
| **Hardware** | 8× H100 80GB |
| **Pre-training duration** | ~20–28 hours |
| **Full pipeline duration** | ~24–32 hours |
| **Estimated cost** | ~$250–350 |

```bash
make accelerate-config-multi GPUS=8
make pretrain  SIZE=1b GPUS=8
make sft       SIZE=1b GPUS=8
make sft-code  SIZE=1b GPUS=8
make dpo       SIZE=1b GPUS=8
make eval      SIZE=1b
make export    SIZE=1b
```

**Note on memory:** The 1B model in bfloat16 is ~2GB of weights. With optimizer states, gradients, and activations it fits comfortably on a single 80GB H100. Gradient checkpointing is already enabled in `gpt_1b.yaml` to reduce activation memory at 4096 sequence length. No tensor parallelism or model parallelism is needed.

**Option B: Nebius AMD Epyc Genoa (Alternative)**

If you're running on the Nebius instance (64 vCPU, 256GiB RAM) for a CPU baseline or curation:

| Aspect | Value |
|---|---|
| **Use case** | Data curation only — not GPU training |
| **Cost** | $1.72/hr |

The Nebius instance is for curation (`make curate SIZE=1b`), not pretraining. GPU training always runs on a separate GPU instance.

---

## Summary Table

| Run | Goal | Model | Tokens | Hardware | Duration | Cost |
|---|---|---|---|---|---|---|
| **1** | Validate pipeline | mini | ~4M | 1× any GPU | 15–30 min | <$1 |
| **2a** | Full 125M | 125M | 5B | 4× A100 | 4–5 hrs | $15–25 |
| **2b** | Full 125M | 125M | 5B | 2× H100 | 2–3 hrs | $15–20 |
| **3a** | Full 350M | 350M | 15B | 4× H100 | 8–10 hrs | $80–120 |
| **3b** | Full 350M | 350M | 15B | 8× H100 | 4–6 hrs | $80–110 |
| **4** | Full 1B | 1B | 30B | 8× H100 | 24–32 hrs | $250–350 |

**Total cost for all four runs (option A path):** ~$345–496
**Total wall-clock time (sequential):** ~37–48 hours

---

## When to Use Each GPU Type

### **Use A100s when:**
- Budget-constrained
- Running 125M or 350M models
- Timeline is flexible (4–10 hours acceptable)

### **Use H100s when:**
- Time-constrained
- Running 1B model (gradient checkpointing at 4096 seq len benefits from H100 memory bandwidth)
- Running 350M on a tight timeline

### **Avoid 8 GPUs for 125M:**
- Diminishing returns — 4→8 GPU gives ~1.8× speedup at 2× cost
- Better to run more experiments on 2–4 GPUs than one fast run on 8

---

## Cost Breakdown by Provider

Prices are approximate spot rates (April 2026):

| Provider | A100 40GB | H100 80GB |
|---|---|---|
| Lambda Labs | ~$1.10/hr | ~$3.50/hr |
| RunPod | ~$0.80/hr (spot) | ~$2.40/hr (spot) |
| Crusoe Energy | ~$0.90/hr | ~$3.00/hr |
| Nebius | — | ~$2.10/hr (H100) |

Use spot instances for all runs — training is resumable from checkpoints, so occasional interruptions are acceptable.

---

## Data Parallelism

The pipeline uses pure data parallelism throughout all model sizes. The model is replicated on each GPU and the batch is split:

```
GPUS=1: Full model on 1 GPU, full batch processed serially
GPUS=2: Model replicated on 2 GPUs, ~1.9x speedup
GPUS=4: Model replicated on 4 GPUs, ~3.7x speedup
GPUS=8: Model replicated on 8 GPUs, ~6.5x speedup
```

Near-linear scaling up to 4 GPUs. Diminishing returns after that due to communication overhead, but 8× H100 is still ~6.5× faster than 1× H100.

**Adjusting configs for multi-GPU:**

Each pretrain config includes a scaling comment. Example for 125M:

```yaml
# 1 GPU:  grad_accum=8,  max_steps=152000
# 4 GPU:  grad_accum=2,  max_steps=38000
# 8 GPU:  grad_accum=1,  max_steps=19000
```

Update `gradient_accumulation_steps` and `max_steps` before launching to keep the global batch size and token budget constant.

SFT and DPO configs similarly include scaling comments — update `gradient_accumulation_steps` before running multi-GPU.

---

## Resuming from Checkpoints

All stages checkpoint periodically. If a run is interrupted:

```bash
# Resume pretraining from latest checkpoint
make pretrain-resume SIZE=125m GPUS=4

# Resume SFT
make sft-resume      SIZE=125m GPUS=4

# Resume DPO
make dpo-resume      SIZE=125m GPUS=2
```

Spot instances are safe for all runs — you lose at most one checkpoint interval of work.

---

## Monitoring GPU Utilization

```bash
# Monitor every 2 seconds
watch -n 2 nvidia-smi

# Or use nvtop for a more detailed view
nvtop
```

**Expected during training:**
- GPU utilization: 90–98%
- GPU memory: 70–90% full

If utilization is below 80%, increase `micro_batch_size` or `gradient_accumulation_steps`. If you're hitting OOM, reduce `micro_batch_size` or enable `gradient_checkpointing: true`.