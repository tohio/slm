# Hardware Recommendations

This guide helps you choose the right GPU configuration for each pipeline stage. The goal is **efficient resource use** — picking the smallest hardware that completes the job reasonably, not the largest hardware available.

The SLM pipeline scales linearly with data parallelism (no code changes needed), so you can adjust `GPUS=N` to match your budget and timeline.

---

## Three-Run Training Plan

We recommend validating the pipeline in three progressively larger runs:

### **Run 1: Pipeline Validation (Sanity Check)**

Prove the entire pipeline works end-to-end before committing to long training runs.

| Aspect | Value |
|---|---|
| **Model** | 125M parameters |
| **Tokens** | 250M (10% of baseline) |
| **Hardware** | 2x A100 40GB |
| **Pre-training duration** | 8-10 minutes |
| **Full pipeline duration** | 30-45 minutes |
| **Estimated cost** | $3-5 |

**Why 2x A100?**
- Validates multi-GPU setup from the start
- Quick feedback loop (under 1 hour)
- Cheap enough to re-run if there are bugs
- Demonstrates data parallelism works

**Command:**
```bash
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=2
make sft CONFIG=finetune/configs/sft_chat_125m.yaml GPUS=2
make sft CONFIG=finetune/configs/sft_code_125m.yaml GPUS=2
make dpo CONFIG=alignment/configs/dpo_125m.yaml GPUS=2
```

---

### **Run 2: 125M Full Training**

Train a complete 125M baseline model with full pre-training data.

**Option A: 4x A100 (Cost-conscious)**

| Aspect | Value |
|---|---|
| **Model** | 125M parameters |
| **Tokens** | 2.5B (full) |
| **Hardware** | 4x A100 40GB |
| **Pre-training duration** | 3-4 hours |
| **Full pipeline duration** | 4-5 hours |
| **Estimated cost** | $12-18 |

**Command:**
```bash
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=4
make sft CONFIG=finetune/configs/sft_chat_125m.yaml GPUS=4
make sft CONFIG=finetune/configs/sft_code_125m.yaml GPUS=4
make dpo CONFIG=alignment/configs/dpo_125m.yaml GPUS=4
```

**Option B: 2x H100 (Speed-focused)**

| Aspect | Value |
|---|---|
| **Model** | 125M parameters |
| **Tokens** | 2.5B (full) |
| **Hardware** | 2x H100 80GB |
| **Pre-training duration** | 1.5-2 hours |
| **Full pipeline duration** | 2-3 hours |
| **Estimated cost** | $15-22 |

**Command:**
```bash
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=2
make sft CONFIG=finetune/configs/sft_chat_125m.yaml GPUS=2
make sft CONFIG=finetune/configs/sft_code_125m.yaml GPUS=2
make dpo CONFIG=alignment/configs/dpo_125m.yaml GPUS=2
```

**Which to choose?**
- **4x A100:** Better for portfolios (shows you understand scaling on standard hardware)
- **2x H100:** Better for time-constrained projects (half the duration at similar cost)

---

### **Run 3: 350M Full Training**

Scale to a larger model with matching token budget (Chinchilla scaling laws).

**Option A: 6x H100 (Balanced)**

| Aspect | Value |
|---|---|
| **Model** | 350M parameters |
| **Tokens** | 7B (Chinchilla-scaled) |
| **Hardware** | 6x H100 80GB |
| **Pre-training duration** | 4-5 hours |
| **Full pipeline duration** | 5-7 hours |
| **Estimated cost** | $60-85 |

**Command:**
```bash
make pretrain CONFIG=pretrain/configs/gpt_350m.yaml GPUS=6
make sft CONFIG=finetune/configs/sft_chat_350m.yaml GPUS=6
make sft CONFIG=finetune/configs/sft_code_350m.yaml GPUS=6
make dpo CONFIG=alignment/configs/dpo_350m.yaml GPUS=6
```

**Option B: 8x H100 (Maximum Efficiency)**

| Aspect | Value |
|---|---|
| **Model** | 350M parameters |
| **Tokens** | 7B (Chinchilla-scaled) |
| **Hardware** | 8x H100 80GB |
| **Pre-training duration** | 3-4 hours |
| **Full pipeline duration** | 4-5 hours |
| **Estimated cost** | $75-100 |

**Command:**
```bash
make pretrain CONFIG=pretrain/configs/gpt_350m.yaml GPUS=8
make sft CONFIG=finetune/configs/sft_chat_350m.yaml GPUS=8
make sft CONFIG=finetune/configs/sft_code_350m.yaml GPUS=8
make dpo CONFIG=alignment/configs/dpo_350m.yaml GPUS=8
```

**Which to choose?**
- **6x H100:** Sweet spot for cost/efficiency
- **8x H100:** Marginal speedup (10-15%) for ~15% more cost

---

---

### **Run 4: 1B Full Training (Production Scale)**

Demonstrate scaling to a production-size model with full Chinchilla token budget.

**Option A: 8x H100 (Recommended)**

| Aspect | Value |
|---|---|
| **Model** | 1B parameters |
| **Tokens** | 20B (Chinchilla-scaled) |
| **Hardware** | 8x H100 80GB with tensor parallelism (TP=2) |
| **Pre-training duration** | 18-24 hours |
| **Full pipeline duration** | 20-26 hours |
| **Estimated cost** | $240-330 |

**Command:**
```bash
make pretrain CONFIG=pretrain/configs/gpt_1b.yaml GPUS=8
make sft CONFIG=finetune/configs/sft_chat_1b.yaml GPUS=8
make sft CONFIG=finetune/configs/sft_code_1b.yaml GPUS=8
make dpo CONFIG=alignment/configs/dpo_1b.yaml GPUS=8
```

**Option B: 16x H100 (Maximum Speed)**

| Aspect | Value |
|---|---|
| **Model** | 1B parameters |
| **Tokens** | 20B (Chinchilla-scaled) |
| **Hardware** | 16x H100 80GB with tensor parallelism (TP=4, DP=4) |
| **Pre-training duration** | 10-12 hours |
| **Full pipeline duration** | 12-14 hours |
| **Estimated cost** | $240-280 |

**Command:**
```bash
make pretrain CONFIG=pretrain/configs/gpt_1b.yaml GPUS=16
make sft CONFIG=finetune/configs/sft_chat_1b.yaml GPUS=16
make sft CONFIG=finetune/configs/sft_code_1b.yaml GPUS=16
make dpo CONFIG=alignment/configs/dpo_1b.yaml GPUS=16
```

**Which to choose?**
- **8x H100:** Good for portfolio (shows understanding of tensor parallelism, reasonable cost)
- **16x H100:** Only if you need results in <14 hours (similar cost but adds complexity)

**Note on 1B scaling:**
- Tensor parallelism (TP) is now required — the model no longer fits on a single GPU
- 8x H100 uses TP=2 (model split across 2 GPUs, 4 data parallel groups)
- 16x H100 uses TP=4 (model split across 4 GPUs, 4 data parallel groups)
- Communication overhead increases, but still near-linear scaling
- This run demonstrates when to move from pure data parallelism to tensor parallelism

---

## Summary Table

| Run | Goal | Model | Tokens | Hardware | Duration | Cost | Key Lesson |
|---|---|---|---|---|---|---|---|
| **1** | Validate | 125M | 250M | 2x A100 | 30-45 min | $3-5 | Pipeline works |
| **2a** | Full baseline | 125M | 2.5B | 4x A100 | 4-5 hrs | $12-18 | Scaling on standard GPUs |
| **2b** | Full baseline | 125M | 2.5B | 2x H100 | 2-3 hrs | $15-22 | When to use expensive GPUs |
| **3a** | Prove scaling | 350M | 7B | 6x H100 | 5-7 hrs | $60-85 | Efficient large runs |
| **3b** | Prove scaling | 350M | 7B | 8x H100 | 4-5 hrs | $75-100 | Maximum utilization |
| **4a** | Production scale | 1B | 20B | 8x H100 | 20-26 hrs | $240-330 | Tensor parallelism required |
| **4b** | Production scale | 1B | 20B | 16x H100 | 12-14 hrs | $240-280 | Multi-node complexity |

**Total cost for all four runs:** $365-555
**Total wall-clock time (sequential):** ~40-60 hours
**Total wall-clock time (parallelized):** Could run runs 2, 3 & 4 simultaneously if resources available, ~26-36 hours

---

## Summary Table

### **When to use A100s**
- Budget-constrained projects
- Learning/portfolio projects (shows you understand efficient scaling)
- Small models (125M-350M)
- Long-running but not urgent (3-8 hour timeline acceptable)

### **When to use H100s**
- Time-constrained (need results in hours)
- Production runs
- Larger models (350M+)
- Complex training scenarios (long sequences, large batches)

### **When NOT to use 8 GPUs for 125M**
- Wasteful and expensive
- Diminishing returns: 4x speedup from 1→4 GPUs, only 2x speedup from 4→8
- Better to run smaller experiments faster on 2x GPUs

---

## Cost Breakdown by Provider

Prices vary by provider and spot vs. on-demand. Current ballpark (March 2026):

### **Lambda Labs**
- A100 40GB: ~$1.10/hr
- H100 80GB: ~$3.50/hr

### **RunPod**
- A100 40GB: ~$0.80/hr (spot)
- H100 80GB: ~$2.40/hr (spot)

### **Crusoe Energy**
- A100 40GB: ~$0.90/hr
- H100 80GB: ~$3.00/hr

**Tip:** Use spot instances for all three runs — training is resumable from checkpoints, so occasional interruptions are acceptable.

---

## Data Parallelism & Tensor Parallelism

**For 125M and 350M (Pure Data Parallelism):**

The pipeline uses pure data parallelism — model is replicated on each GPU, batch is split:

```
GPUS=1: Full model on 1 GPU, full batch processed serially
GPUS=2: Model replicated on 2 GPUs, batch split and processed in parallel
GPUS=4: Model replicated on 4 GPUs, 4x speedup (near-linear)
GPUS=6+: Speedup ~5-5.5x for 6, ~6-6.5x for 8 (communication overhead grows)
```

**Scaling efficiency (data parallel only):**
- 1→2 GPU: ~1.9x speedup
- 1→4 GPU: ~3.8x speedup
- 1→6 GPU: ~5.5x speedup
- 1→8 GPU: ~7x speedup (not linear, but still good)

Diminishing returns kick in after 4 GPUs, so 6-8 H100s is the practical ceiling for 350M.

**For 1B (Tensor + Data Parallelism):**

The 1B model requires **tensor parallelism** — the model itself is split across multiple GPUs:

```
GPUS=8 with TP=2, DP=4:
  - Model split across 2 GPUs (tensor parallel)
  - Replicated across 4 groups (data parallel)
  - Each group has full model copy split in half

GPUS=16 with TP=4, DP=4:
  - Model split across 4 GPUs (tensor parallel)
  - Replicated across 4 groups (data parallel)
  - Finer model parallelism, higher communication overhead
```

**Scaling efficiency (tensor + data parallel):**
- 8x H100: ~6.5-7x speedup vs. 1 GPU
- 16x H100: ~13-14x speedup vs. 1 GPU (nearly linear)

Tensor parallelism communication is more expensive than data parallelism, so efficiency drops slightly. For 1B, 8x H100 is a good balance — 16x is faster but the communication overhead grows.

---

## Resuming from Checkpoints

All stages checkpoint every 1,000 steps. If a run is interrupted:

```bash
# Training resumes automatically from the latest checkpoint
make pretrain CONFIG=pretrain/configs/gpt_125m.yaml GPUS=4
```

This makes spot instances safe — you can absorb interruptions without losing progress.

---

## Quick Reference: Command Template

```bash
# Run any stage with custom GPU count
make pretrain CONFIG=pretrain/configs/gpt_MODEL_SIZE.yaml GPUS=N
make sft CONFIG=finetune/configs/sft_STAGE_MODEL_SIZE.yaml GPUS=N
make dpo CONFIG=alignment/configs/dpo_MODEL_SIZE.yaml GPUS=N

# Example: 350M with 8 H100s
make pretrain CONFIG=pretrain/configs/gpt_350m.yaml GPUS=8
```

---

## Monitoring GPU Utilization

During training, monitor GPU usage to confirm you're fully utilizing the hardware:

```bash
# On the training instance (every 2 seconds)
watch -n 2 nvidia-smi

# Expected: GPUs should be 95%+ utilized, memory 80%+ full
```

If GPU utilization is low (<80%), you may benefit from increasing batch size or sequence length in the config.

If GPU memory is maxed out and you're getting OOM errors, reduce batch size or use gradient checkpointing (already enabled by default).