# Pre-Training

Training a GPT-style language model from scratch on the curated dataset. The model learns to predict the next token across a diverse corpus of general text and code — building the foundational representations that all downstream fine-tuning stages depend on.

## Architecture

The baseline model targets ~125M parameters — small enough to complete a full training run and iterate quickly, large enough to produce coherent language. The architecture is a standard GPT decoder with learned positional embeddings.

| | 125M (baseline) | 350M | 1B |
|---|---|---|---|
| Layers | 12 | 24 | 32 |
| Hidden size | 768 | 1024 | 2048 |
| Attention heads | 12 | 16 | 16 |
| Context length | 2048 | 2048 | 2048 |
| Tokens (Chinchilla) | ~2.5B | ~7B | ~20B |
| Parallelism | data only | data only | TP=2, DP=2 |

Scaling to 350M or 1B requires only the `SIZE` flag — no code or config changes.

```bash
make pretrain SIZE=350m GPUS=4
make pretrain SIZE=1b   GPUS=4
```

## Framework

Pre-training uses the **NeMo 2.x API** (`nemo.collections.llm.GPTModel`) running inside `nvcr.io/nvidia/nemo:25.02`. Configuration is Python-native — no YAML/Hydra required. All hyperparameters are passed as CLI arguments by `pretrain/scripts/train.sh`.

After pretraining, a conversion step produces `mcore_gpt.nemo` — the format required by NeMo-Aligner for SFT and DPO:

```
make pretrain         →  NeMo 2.x distributed checkpoint (/results/slm_gpt_125m/)
make convert-pretrain →  /results/slm_gpt_125m/mcore_gpt.nemo
make sft              →  loads mcore_gpt.nemo via NeMo-Aligner
```

## Design Decisions

**Why from scratch instead of continued pre-training?**
Starting from an existing checkpoint (LLaMA, Mistral) is the right production choice — it's faster and cheaper. We start from scratch deliberately: it exercises every stage of the pipeline, exposes how data quality and tokenizer design interact with training dynamics, and makes the pre-training stage non-trivial to showcase.

**Why BF16 over FP16?**
BF16 has the same range as FP32 (8 exponent bits) with reduced precision (7 mantissa bits). FP16's smaller range (5 exponent bits) requires loss scaling to prevent gradient underflow — BF16 doesn't. On A100/H100, BF16 is natively supported. Simpler training loop, no loss scaling to tune.

**Gradient checkpointing**
Enabled by default. Trades compute for memory — recomputes activations during the backward pass rather than storing them. At 125M on a single H100 this isn't strictly necessary, but it's good practice and required when scaling to 1B.

**Data blend: 70% general, 30% code**
The model needs strong general language understanding to be useful for chat. The 70/30 split is a starting point; adjust based on what your SFT data looks like.

## Training Dynamics

Key metrics to watch:

- **Training loss** should decrease smoothly. Spikes followed by recovery are normal.
- **Validation loss** should track training loss. A widening gap indicates overfitting.
- **Gradient norm** should stay below `gradient_clip_val=1.0`.
- **Throughput (tokens/sec)** should be stable.

Expected validation perplexity at convergence: **80–150** for 125M trained on ~2.5B tokens.

## Prerequisites

```bash
# 1. Host directories and data
make init-dirs
make setup-instance S3_BUCKET=my-bucket   # pulls tokenized data + tokenizer from S3

# 2. Verify dataset
ls /data/curated/tokenized/               # should show text_document.bin + text_document.idx

# 3. Build the Docker image (NGC auth handled by setup_gpu_instance.sh)
make docker-build
```

## Usage

```bash
# Single GPU, 125M (default)
make pretrain

# Multi-GPU, 350M
make pretrain SIZE=350m GPUS=4

# Multi-GPU, 1B (TP=2 required)
make pretrain SIZE=1b GPUS=4

# Enable W&B logging
docker run --gpus all --rm ... bash pretrain/scripts/train.sh --size 125m --wandb

# Resume from latest checkpoint
docker run --gpus all --rm ... bash pretrain/scripts/train.sh --size 125m --resume

# Convert checkpoint for NeMo-Aligner (required before make sft)
make convert-pretrain

# Evaluate
make eval-pretrain

# Export to HuggingFace
make convert-hf
```

## Infrastructure

For 125M on ~2.5B tokens:

| GPU | Est. Time | Est. Cost |
|---|---|---|
| 1x H100 80GB | ~8–12hrs | ~$25–36 |
| 1x A100 40GB | ~12–18hrs | ~$18–27 |
| 4x A100 40GB | ~4–6hrs | ~$24–36 |

Checkpoints are saved to `/results/slm_gpt_125m/` every 1,000 steps and bind-mounted to the host. Training resumes automatically from the latest checkpoint on the next run.