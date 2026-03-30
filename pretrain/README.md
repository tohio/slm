# Pre-Training

Training a GPT-style language model from scratch on the curated dataset. The model learns to predict the next token across a diverse corpus of general text and code — building the foundational representations that all downstream fine-tuning stages depend on.

## Architecture

The baseline model targets ~125M parameters — small enough to complete a full training run and iterate quickly, large enough to produce coherent language.

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

Pre-training uses **NeMo 1.x** via the built-in script at `/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py` inside `nvcr.io/nvidia/nemo:25.02`. All hyperparameters live in the YAML configs (`pretrain/configs/gpt_<size>.yaml`), passed via Hydra CLI.

The key config fields are `mcore_gpt: true` (uses megatron-core under the hood) and `save_nemo_on_train_end: true` (saves a `.nemo` tarball at the end of training).

NeMo-Aligner SFT/DPO loads the `.nemo` checkpoint directly — no conversion step needed:

```
make pretrain   →  /results/slm_gpt_125m/last.nemo
make sft        →  loads last.nemo via GPTSFTModel.restore_from()
```

## Design Decisions

**Why NeMo 1.x instead of NeMo 2.x?**
NeMo 2.x (`nemo.collections.llm.GPTModel`) saves distributed checkpoints in a format that NeMo-Aligner 0.7.0 cannot load — there is no HF or `.nemo` export connector registered for the generic `GPTModel` in `nemo:25.02`. NeMo 1.x saves `.nemo` tarballs that NeMo-Aligner loads natively, giving a clean end-to-end pipeline with no conversion step. This tradeoff becomes irrelevant in later containers (`25.04+`) where NeMo-RL replaces NeMo-Aligner.

**Why from scratch instead of continued pre-training?**
Starting from scratch exercises every stage of the pipeline and exposes how data quality and tokenizer design interact with training dynamics.

**Why BF16?**
BF16 has the same range as FP32 with reduced precision. FP16 requires loss scaling to prevent gradient underflow — BF16 doesn't. Simpler training loop, no loss scaling to tune.

**Gradient checkpointing**
Enabled by default (`activations_checkpoint_method: uniform`). Trades compute for memory — required when scaling to 1B.

**Data blend: 70% general, 30% code**
The model needs strong general language understanding to be useful for chat. Adjust based on your SFT data mix.

## Training Dynamics

- **Training loss** should decrease smoothly. Spikes followed by recovery are normal.
- **Validation loss** should track training loss. A widening gap indicates overfitting.
- **Gradient norm** should stay below `gradient_clip_val=1.0`.
- **Throughput (tokens/sec)** should be stable.

Expected validation perplexity at convergence: **80–150** for 125M trained on ~2.5B tokens.

## Prerequisites

```bash
# 1. Host directories and data
make init-dirs
make setup-instance S3_BUCKET=my-bucket

# 2. Verify dataset
ls /data/curated/tokenized/   # text_document.bin + text_document.idx

# 3. Docker image
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
make pretrain SIZE=125m -- --wandb

# Resume from latest checkpoint (auto-detected via exp_manager)
make pretrain SIZE=125m -- --resume

# Evaluate after training
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

Checkpoints are saved every 1,000 steps to `/results/slm_gpt_125m/`. Training resumes automatically from the latest checkpoint on the next run via `exp_manager.resume_if_exists: true`.