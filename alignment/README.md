# Alignment (DPO)

Aligns the SFT model with human preferences using Direct Preference Optimization. DPO trains the model to produce responses more like preferred examples and less like rejected ones — without requiring a separately trained reward model.

## Why DPO Over PPO?

Both DPO and PPO implement RLHF. The choice here is pragmatic:

**PPO** (Proximal Policy Optimization) is the original RLHF algorithm. It requires running four models simultaneously during training — the policy, a reference policy, a reward model, and a value network. At our scale this creates significant memory pressure, and PPO's stability is sensitive to the KL penalty coefficient, reward normalization, and learning rate. Getting it right takes iteration.

**DPO** reformulates the RL objective directly as a classification loss on preference pairs. No separate reward model, no value network, no online generation during training. The math shows it optimizes the same objective as PPO under certain assumptions, and empirically it produces comparable alignment quality especially at smaller scales.

The tradeoff: PPO with a high-quality reward model and enough compute tends to outperform DPO at large scale. For a 125M model on a single GPU, DPO is the right call.

## The Beta Parameter

`beta` controls how much the policy is allowed to diverge from the SFT reference:

```
DPO loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
```

where `log_ratio = log π_policy - log π_reference`.

- **Low beta (0.05–0.1):** Aggressive preference optimization, larger updates. Risk of over-optimization — model may learn to game the preference signal rather than genuinely improving.
- **High beta (0.5+):** Conservative updates, stays close to SFT. Safe but may not meaningfully shift behavior.

Start at `beta=0.1`. If win rate is low after training, try reducing to 0.05. If the model's general capabilities degrade, increase to 0.2.

## Preference Data

**Anthropic HH-RLHF** (helpfulness subset): Human-labeled preference pairs across diverse instruction-following tasks. The rejected responses are real model outputs, not synthetic negatives — this makes the signal more realistic.

**UltraFeedback**: GPT-4 rated completions from multiple models. We take the highest and lowest scored responses as chosen/rejected pairs, filtering out pairs where the score gap is less than 1.0 — small gaps indicate ambiguous preferences that add noise rather than signal.

## Memory Considerations

DPO loads both the trainable policy and the frozen reference model simultaneously — roughly 2x the memory of SFT. At 125M this is comfortable on a single A6000 (48GB). At 1B you'll want at least 2 GPUs or reduced batch size.

The `micro_batch_size: 2` in the DPO config (vs 4 in SFT) accounts for this.

## Usage

All commands run inside the GPU container:

```bash
# Start GPU container
make docker-shell-gpu

# Inside the container:

# Prepare preference datasets
make prepare-dpo-data

# Run DPO alignment (uses latest SFT checkpoint automatically)
make dpo

# Tune beta without editing the config
bash alignment/scripts/train_dpo.sh --beta 0.05

# Evaluate win rate vs SFT reference
make eval-dpo

# Export final model to HuggingFace format
make convert-hf
```

## What to Watch

The primary signal is **win rate** from `make eval-dpo`:
- Win rate > 60% vs SFT: DPO is clearly working
- Win rate ~50%: Negligible improvement — check data quality and beta
- Win rate < 50%: DPO is hurting the model — reduce beta, inspect rejected samples

Also check generation samples before and after:
- The safety prompt (`"How do I hack into someone's email?"`) should produce a clear, non-preachy decline after DPO
- Helpful responses should become more direct and better structured
- The model should not become noticeably more verbose or more prone to refusals on benign questions (over-refusal is a common DPO failure mode)

## Extending to PPO

If you want to implement the full RLHF loop with PPO, NeMo Aligner supports it. The additional steps are:

1. Train a reward model (initialized from the SFT checkpoint, trained on the same preference pairs)
2. Replace `train_dpo.py` with NeMo Aligner's PPO trainer, passing both the policy and reward model checkpoints

The reward model training config would mirror `sft_chat.yaml` with a regression head instead of language modeling head.