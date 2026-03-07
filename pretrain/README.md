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

Scaling to 350M or 1B requires only a config change — no code modifications.

```bash
make pretrain CONFIG=pretrain/configs/gpt_350m.yaml GPUS=4
```

## Design Decisions

**Why from scratch instead of continued pre-training?**
Starting from an existing checkpoint (LLaMA, Mistral) is the right production choice — it's faster and cheaper. We start from scratch here deliberately: it exercises every stage of the pipeline, exposes how data quality and tokenizer design interact with training dynamics, and makes the pre-training stage non-trivial to showcase. Switching to continued pre-training later is a one-line config change.

**Why BF16 over FP16?**
BF16 has the same range as FP32 (8 exponent bits) with reduced precision (7 mantissa bits). FP16's smaller range (5 exponent bits) requires loss scaling to prevent gradient underflow — BF16 doesn't. On A100/A6000, BF16 is natively supported with the same throughput as FP16. Simpler training loop, no loss scaling to tune.

**Gradient checkpointing**
Enabled by default (`activations_checkpoint_method: uniform`). Trades compute for memory — recomputes activations during the backward pass rather than storing them. At 125M on a single A6000 this isn't strictly necessary, but it's good practice and required when scaling to 1B.

**Data blend: 70% general, 30% code**
The model needs strong general language understanding to be useful for chat — code-heavy pre-training produces models that are technically capable but poor at natural conversation. The 70/30 split is a starting point; adjust based on what your SFT data looks like.

## Training Dynamics

Key metrics to watch during pre-training:

- **Training loss** should decrease smoothly. Spikes followed by recovery are normal. A spike that doesn't recover suggests a bad batch or a learning rate issue.
- **Validation loss** should track training loss. A widening gap indicates overfitting (unlikely at this data scale, but worth watching).
- **Gradient norm** should stay below `gradient_clip_val=1.0`. Sustained high norms suggest the LR is too high.
- **Throughput (tokens/sec)** should be stable. Sudden drops often indicate GPU memory pressure or NCCL communication issues on multi-GPU.

Expected validation perplexity at convergence: **80–150** for 125M trained on ~2.5B tokens. This is the baseline; SFT and DPO don't optimize perplexity directly so it may increase slightly in later stages.

## Usage

```bash
# Single GPU, 125M (default)
make pretrain

# Multi-GPU, 350M
make pretrain CONFIG=pretrain/configs/gpt_350m.yaml GPUS=4

# Resume from latest checkpoint (auto-detected)
bash pretrain/scripts/train.sh --config pretrain/configs/gpt_125m.yaml

# Evaluate after training
make eval-pretrain

# Export to HuggingFace format for external evaluation
bash pretrain/scripts/convert_ckpt.sh \
    --direction nemo_to_hf \
    --input /results/slm_gpt_125m/checkpoints/last.nemo \
    --output /results/slm_gpt_125m_hf/
```

## Infrastructure

Pre-training is the most GPU-intensive stage. For 125M on ~2.5B tokens:

| GPU | Est. Time | Est. Cost |
|---|---|---|
| 1x A6000 48GB | ~20–30hrs | ~$20–28 |
| 1x A100 40GB | ~12–18hrs | ~$18–27 |
| 4x A6000 48GB | ~6–8hrs | ~$22–30 |

Checkpoints are saved every 1,000 steps. If the instance is terminated, training resumes automatically from the latest checkpoint on the next run.
