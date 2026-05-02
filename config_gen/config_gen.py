"""
config_gen/config_gen.py
-----------------
Generate per-GPU training configs for pretrain, SFT, and DPO stages.

Decision rule: if the GPU choice affects what's optimal, the script computes
the field. Otherwise the field comes verbatim from a recipe profile.

Hardware-driven (script computes):
    micro_batch_size                — picked to fit the GPU's VRAM budget
    gradient_accumulation_steps     — derived to hit the recipe's reference global batch
    gradient_checkpointing          — auto-policy based on memory pressure
    max_steps                       — pretrain only; derived from consumed-token target
    warmup_steps                    — pretrain only; derived from max_steps

Recipe-driven (preserved verbatim from the profile):
    learning_rate, weight_decay, betas, epochs, warmup_ratio,
    max_seq_length, dpo.beta, dpo.max_prompt_length, paths, etc.

Token vocabulary used in this file:
    corpus_tokens     unique tokens in the curated dataset (the public-facing
                      figure on model cards). Lives in config/data_mix.py.
    consumed_tokens   corpus_tokens × epochs; the count of tokens the optimizer
                      sees over the whole run. Used internally to compute
                      max_steps. NOT a public-facing number. Sourced from
                      config.data_mix.consumed_tokens(size) — no hard-coded
                      duplicates here.

Three tuning modes:
    conservative   70% VRAM budget, ckpt-friendly, leaves headroom
    balanced       80% VRAM budget — default
    aggressive     90% VRAM budget, pushes micro_batch

Stages:
    pretrain        writes pretrain/configs/gpt_<size>.yaml
    sft             writes finetune/configs/sft_chat_<size>.yaml AND sft_code_<size>.yaml
    dpo             writes alignment/configs/dpo_<size>.yaml

Usage:
    python -m config_gen.config_gen --stage pretrain --gpu h200 --size 125m --gpus 1
    python -m config_gen.config_gen --stage sft      --detect    --size 350m --gpus 4
    python -m config_gen.config_gen --stage dpo      --gpu b200  --size 1b   --gpus 8 --mode aggressive

    # Convenience (via Makefile): generate all three stages
    make config-gen SIZE=125m GPUS=1
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Single source of truth for corpus sizes and epoch counts. config_gen reads
# from here so that consumed_tokens (= corpus × epochs) stays in lockstep
# with what the curator and export pipeline see — no hard-coded duplicates.
from config import data_mix


# ── GPU specs ─────────────────────────────────────────────────────────────────

GPU_SPECS: dict[str, dict] = {
    "a100_40":   {"vram_gb": 40,  "bf16_tflops": 312,  "display": "A100 40GB"},
    "a100_80":   {"vram_gb": 80,  "bf16_tflops": 312,  "display": "A100 80GB"},
    "l40s":      {"vram_gb": 48,  "bf16_tflops": 362,  "display": "L40S"},
    "h100":      {"vram_gb": 80,  "bf16_tflops": 989,  "display": "H100 80GB"},
    "h100_sxm":  {"vram_gb": 80,  "bf16_tflops": 989,  "display": "H100 SXM 80GB"},
    "h200":      {"vram_gb": 141, "bf16_tflops": 989,  "display": "H200 SXM 141GB"},
    "b200":      {"vram_gb": 192, "bf16_tflops": 2250, "display": "B200 192GB"},
    "rtx4090":   {"vram_gb": 24,  "bf16_tflops": 165,  "display": "RTX 4090"},
    "rtx5090":   {"vram_gb": 32,  "bf16_tflops": 250,  "display": "RTX 5090"},
}

NVIDIA_SMI_NAME_MAP: list[tuple[str, str]] = [
    ("H200",            "h200"),
    ("H100 SXM",        "h100_sxm"),
    ("H100",            "h100"),
    ("B200",            "b200"),
    ("A100 80GB",       "a100_80"),
    ("A100-SXM4-80GB",  "a100_80"),
    ("A100 40GB",       "a100_40"),
    ("A100",            "a100_40"),
    ("L40S",            "l40s"),
    ("RTX 4090",        "rtx4090"),
    ("RTX 5090",        "rtx5090"),
]


# ── Tuning modes ─────────────────────────────────────────────────────────────
# Each mode is a tuple of policy knobs:
#   vram_fraction       — what fraction of total VRAM to plan for
#   ckpt_threshold      — min micro_batch size achievable without ckpt before
#                         the policy will turn ckpt on automatically
#   power_of_two_only   — round micro_batch down to nearest pow-of-2

@dataclass(frozen=True)
class Mode:
    vram_fraction: float
    ckpt_threshold: int        # if max-no-ckpt < this, prefer ckpt
    power_of_two_only: bool

MODES: dict[str, Mode] = {
    "conservative": Mode(vram_fraction=0.70, ckpt_threshold=8, power_of_two_only=True),
    "balanced":     Mode(vram_fraction=0.80, ckpt_threshold=4, power_of_two_only=True),
    "aggressive":   Mode(vram_fraction=0.90, ckpt_threshold=2, power_of_two_only=False),
}


# ── Profile definitions ──────────────────────────────────────────────────────
# state_gb       = weights + grads + optimizer state (AdamW: 2× FP32 momentum)
# act_per_seq_gb = peak activation memory per sequence at the profile's seq_len
# All numbers are analytical estimates; re-measure after first real run.

@dataclass(frozen=True)
class PretrainProfile:
    # Hardware-relevant (used by the auto-policy)
    state_gb: float
    act_per_seq_gb_no_ckpt: float
    act_per_seq_gb_ckpt: float
    ctx: int                          # max_position_embeddings
    ref_global_batch: int             # sequences per optimizer step

    # Token target — used only to compute max_steps.
    # consumed_tokens = corpus_tokens × epochs. NOT the public corpus figure.
    # The unique-data number lives in config/data_mix.py TARGET_CONFIGS.
    consumed_tokens: int

    # Recipe (preserved verbatim)
    lr: float
    warmup_frac: float
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8

    # Architecture (preserved verbatim)
    hidden: int = 0
    layers: int = 0
    heads: int = 0
    kv_heads: int = 0
    rope_theta: float = 500_000.0


@dataclass(frozen=True)
class SFTProfile:
    # Hardware-relevant
    state_gb: float
    act_per_seq_gb_no_ckpt: float
    act_per_seq_gb_ckpt: float
    max_seq_length: int
    ref_global_batch: int

    # Recipe (preserved verbatim)
    lr: float
    epochs: int
    warmup_ratio: float
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98

    # Trainer fields (preserved verbatim)
    # Packing defaults False because the current custom attention does not
    # enforce packed-example boundaries. Enable only after safe packed attention
    # / FA2 varlen support is implemented.
    packing: bool = False
    eval_steps: int = 2000
    save_steps: int = 2000

    # torch_compile is disabled by default for SFT until profiled.
    torch_compile: bool = False


@dataclass(frozen=True)
class DPOProfile:
    # Hardware-relevant — DPO state and activations are 2× SFT
    # (policy + reference, plus chosen + rejected forward passes)
    state_gb: float
    act_per_seq_gb_no_ckpt: float
    act_per_seq_gb_ckpt: float
    max_seq_length: int
    ref_global_batch: int

    # Recipe (preserved verbatim)
    lr: float
    epochs: int
    warmup_ratio: float
    dpo_beta: float
    max_prompt_length: int
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98

    # Trainer fields
    eval_steps: int = 200
    save_steps: int = 200

    # torch_compile is disabled by default for DPO until profiled.
    torch_compile: bool = False

# consumed_tokens (= corpus_tokens × epochs) is sourced from
# config/data_mix.py — that's the single source of truth for both numbers.
# The values here therefore evolve automatically when the curator's TARGET_CONFIGS
# is edited; no manual sync required.
#
#   125m: 5B  × 2 = 10B
#   350m: 15B × 2 = 30B
#   1b:   30B × 1 = 30B
SIZE_PROFILES: dict[str, PretrainProfile] = {
    "125m": PretrainProfile(
        state_gb=2.0, act_per_seq_gb_no_ckpt=1.75, act_per_seq_gb_ckpt=0.60,
        ctx=2048, ref_global_batch=64, consumed_tokens=data_mix.consumed_tokens("125m"),
        lr=3.0e-4, warmup_frac=0.01, hidden=768, layers=16, heads=12, kv_heads=4,
    ),
    "350m": PretrainProfile(
        state_gb=5.0, act_per_seq_gb_no_ckpt=0.75, act_per_seq_gb_ckpt=0.16,
        ctx=2048, ref_global_batch=128, consumed_tokens=data_mix.consumed_tokens("350m"),
        lr=2.0e-4, warmup_frac=0.01, hidden=1024, layers=28, heads=16, kv_heads=8,
    ),
    "1b": PretrainProfile(
        state_gb=14.5, act_per_seq_gb_no_ckpt=4.0, act_per_seq_gb_ckpt=0.80,
        ctx=4096, ref_global_batch=128, consumed_tokens=data_mix.consumed_tokens("1b"),
        lr=1.0e-4, warmup_frac=0.035, hidden=2048, layers=38, heads=32, kv_heads=8,
    ),
}


# SFT chat profiles.
#   125m: calibrated against a real OOM event at micro=64 (see commit history).
#         state_gb bumped from pretrain (~2.0) to capture TRL's overhead from
#         label tensors, masking buffers, and a longer-lived logits tensor
#         during external CE-loss computation.
#   350m / 1b: extrapolated from pretrain by adding ~10% to activations to
#              capture the same TRL tax 125m exhibited. NOT measured —
#              re-measure on first real run for each size.
SFT_CHAT_PROFILES: dict[str, SFTProfile] = {
    "125m": SFTProfile(
        state_gb=2.0, act_per_seq_gb_no_ckpt=3.9, act_per_seq_gb_ckpt=1.3,
        max_seq_length=2048, ref_global_batch=64, lr=1.0e-5, epochs=2,
        warmup_ratio=0.03, eval_steps=2000, save_steps=2000,
    ),
    "350m": SFTProfile(
        state_gb=5.0, act_per_seq_gb_no_ckpt=5.0, act_per_seq_gb_ckpt=1.7,
        max_seq_length=2048, ref_global_batch=128,
        lr=8.0e-6, epochs=2, warmup_ratio=0.03, eval_steps=2000, save_steps=2000,
    ),
    "1b": SFTProfile(
        state_gb=14.5, act_per_seq_gb_no_ckpt=14.0, act_per_seq_gb_ckpt=4.5,
        max_seq_length=4096, ref_global_batch=128,
        lr=5.0e-6, epochs=2, warmup_ratio=0.03, eval_steps=2000, save_steps=2000,
    ),
}

# SFT code — same architecture and memory profile as chat.
# Recipe differs: lower LR to reduce catastrophic forgetting of chat.
SFT_CODE_PROFILES: dict[str, SFTProfile] = {
    "125m": SFTProfile(
        state_gb=2.0, act_per_seq_gb_no_ckpt=3.9, act_per_seq_gb_ckpt=1.3,
        max_seq_length=2048, ref_global_batch=64, lr=5.0e-6,
        epochs=2, warmup_ratio=0.03, eval_steps=2000, save_steps=2000,
    ),
    "350m": SFTProfile(
        state_gb=5.0, act_per_seq_gb_no_ckpt=5.0, act_per_seq_gb_ckpt=1.7,
        max_seq_length=2048, ref_global_batch=128,
        lr=4.0e-6, epochs=2, warmup_ratio=0.03, eval_steps=2000, save_steps=2000,
    ),
    "1b": SFTProfile(
        state_gb=14.5, act_per_seq_gb_no_ckpt=14.0, act_per_seq_gb_ckpt=4.5,
        max_seq_length=4096, ref_global_batch=128,
        lr=2.5e-6, epochs=2, warmup_ratio=0.03, eval_steps=2000, save_steps=2000,
    ),
}

# DPO profiles — state is policy weights + grads + AdamW + ref weights (no grads).
# That's roughly pretrain state + ref weights (BF16) ≈ pretrain state * 1.15-1.2x.
# Activations are roughly 4× SFT: chosen + rejected pairs through both policy and ref.
#
# FIXME: act_per_seq values below are suspect across all sizes.
#        SFT 125m measured at 1.9 GB/seq (state 6.0); DPO 125m here lists
#        0.22 GB/seq (state 2.3) — but DPO does ~4× the activation work
#        of SFT (chosen+rejected through both policy and reference). Expect
#        ~7-8 GB/seq for 125m, scaled accordingly for 350m and 1b.
#        These numbers will likely OOM on the auto-policy's first attempt.
#        Re-measure peak VRAM on the first real DPO run for each size and
#        update — same calibration approach used for SFT 125m.
DPO_PROFILES: dict[str, DPOProfile] = {
    "125m": DPOProfile(
        state_gb=2.3, act_per_seq_gb_no_ckpt=0.22, act_per_seq_gb_ckpt=0.05,
        max_seq_length=2048, ref_global_batch=64,
        lr=5.0e-7, epochs=1, warmup_ratio=0.05,
        dpo_beta=0.1, max_prompt_length=1024,
    ),
    "350m": DPOProfile(
        state_gb=5.8, act_per_seq_gb_no_ckpt=2.8, act_per_seq_gb_ckpt=0.56,
        max_seq_length=2048, ref_global_batch=64,
        lr=3.0e-7, epochs=1, warmup_ratio=0.05,
        dpo_beta=0.1, max_prompt_length=1024,
    ),
    "1b": DPOProfile(
        state_gb=16.5, act_per_seq_gb_no_ckpt=15.2, act_per_seq_gb_ckpt=2.8,
        max_seq_length=4096, ref_global_batch=64,
        lr=2.0e-7, epochs=1, warmup_ratio=0.05,
        dpo_beta=0.1, max_prompt_length=2048,
    ),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _round_down_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _max_micro_for_budget(state_gb: float, act_per_seq_gb: float,
                          budget_gb: float) -> int:
    raw = (budget_gb - state_gb) / act_per_seq_gb
    return max(0, math.floor(raw))


def _pick_micro_batch(max_micro: int, target_per_gpu: int,
                      power_of_two_only: bool) -> int:
    """
    Pick a micro_batch size given the largest that fits and the target per GPU.
    Caps at target_per_gpu (no point going larger than one optimizer step needs).
    """
    capped = max(1, min(max_micro, target_per_gpu))
    if power_of_two_only:
        return _round_down_pow2(capped)
    return capped


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class GeneratedConfig:
    # Decisions (hardware-driven)
    micro_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_steps: Optional[int] = None         # pretrain only
    warmup_steps: Optional[int] = None      # pretrain only

    # Reporting
    actual_global_batch: int = 0
    tokens_per_step: int = 0
    actual_consumed_tokens: int = 0         # pretrain only — corpus × epochs
    estimated_vram_gb: float = 0.0
    vram_budget_gb: float = 0.0

    # Echo of inputs
    stage: str = ""
    gpu_key: str = ""
    size: str = ""
    num_gpus: int = 0
    mode: str = ""

    # Heads-up notes (informational, never blocking)
    warnings: list[str] = field(default_factory=list)


# ── Common decision logic ────────────────────────────────────────────────────

def _decide_batch_and_ckpt(
    state_gb: float,
    act_no_ckpt: float,
    act_ckpt: float,
    ref_global_batch: int,
    num_gpus: int,
    spec: dict,
    mode: Mode,
    force_ckpt: Optional[bool],
) -> tuple[int, int, bool, float, float, list[str]]:
    """
    Pick (micro_batch, grad_accum, ckpt, est_vram, budget, warnings).

    Single source of truth for the auto-policy across pretrain/SFT/DPO.
    """
    warnings: list[str] = []
    budget_gb = spec["vram_gb"] * mode.vram_fraction
    per_gpu_target = max(1, ref_global_batch // num_gpus)

    # Decide ckpt
    if force_ckpt is None:
        max_no_ckpt = _max_micro_for_budget(state_gb, act_no_ckpt, budget_gb)
        max_with_ckpt = _max_micro_for_budget(state_gb, act_ckpt, budget_gb)
        if max_no_ckpt < 1:
            ckpt = True
        elif max_no_ckpt < mode.ckpt_threshold:
            ckpt = True
        elif max_no_ckpt < per_gpu_target and max_with_ckpt >= 4 * max_no_ckpt:
            # Ckpt unlocks ≥ 4× more micro and the no-ckpt config can't reach
            # the per-GPU target in a single step — eliminating accumulation
            # overhead can outweigh recompute cost.
            ckpt = True
        else:
            ckpt = False
    else:
        ckpt = force_ckpt

    act = act_ckpt if ckpt else act_no_ckpt
    max_micro = _max_micro_for_budget(state_gb, act, budget_gb)
    if max_micro < 1:
        if not ckpt:
            ckpt = True
            act = act_ckpt
            max_micro = _max_micro_for_budget(state_gb, act, budget_gb)
        if max_micro < 1:
            raise RuntimeError(
                f"Does not fit on {spec['display']} even at micro_batch=1 "
                f"with gradient checkpointing. State alone needs "
                f"{state_gb:.1f} GB; budget is {budget_gb:.1f} GB. "
                f"Use a larger GPU or shard with FSDP."
            )
    
    # Aggressive mode can produce values like 63 when the recipe target is 64
    # due to conservative floor rounding of noisy memory estimates. If we are
    # exactly one sequence below the target, snap to the target. Do not use a
    # wider tolerance here; larger jumps should remain explicit.
    if not mode.power_of_two_only and max_micro == per_gpu_target - 1:
        max_micro = per_gpu_target

    micro_batch = _pick_micro_batch(max_micro, per_gpu_target, mode.power_of_two_only)
    grad_accum = max(1, ref_global_batch // (micro_batch * num_gpus))
    actual_global = micro_batch * grad_accum * num_gpus
    est_vram = state_gb + (micro_batch * act)

    # Heads-up notes
    used_pct = est_vram / spec["vram_gb"]
    if used_pct < 0.30:
        warnings.append(
            f"Predicted VRAM {est_vram:.0f}/{spec['vram_gb']} GB ({used_pct*100:.0f}%) — "
            f"lots of headroom. Try `--mode aggressive` for a larger micro_batch "
            f"and fewer accumulation steps."
        )
    elif used_pct > 0.90:
        warnings.append(
            f"Predicted VRAM {est_vram:.0f}/{spec['vram_gb']} GB ({used_pct*100:.0f}%) — "
            f"close to capacity. Eval batches and attention spikes may OOM. "
            f"Consider `--mode conservative` or `--ckpt`."
        )

    if actual_global != ref_global_batch:
        ratio = actual_global / ref_global_batch
        warnings.append(
            f"Achieved global_batch={actual_global} vs reference={ref_global_batch} "
            f"({ratio:.2f}×). Recipe LR was tuned for the reference; consider "
            f"reviewing or rerunning with `--target-global-batch {ref_global_batch}`."
        )

    if ckpt and not force_ckpt and used_pct < 0.50:
        warnings.append(
            f"Auto-policy enabled gradient_checkpointing despite "
            f"{used_pct*100:.0f}% VRAM use — checkpointing was needed to fit "
            f"{ref_global_batch} sequences in one optimizer step. "
            f"Run with `--no-ckpt` to A/B if you'd rather accumulate."
        )

    return micro_batch, grad_accum, ckpt, est_vram, budget_gb, warnings


# ── Pretrain ─────────────────────────────────────────────────────────────────

def compute_pretrain_config(
    gpu_key: str, size: str, num_gpus: int, mode_name: str = "balanced",
    target_global_batch: Optional[int] = None,
    target_consumed_tokens: Optional[int] = None,
    force_ckpt: Optional[bool] = None,
) -> GeneratedConfig:
    """
    Compute a pretrain config for the given GPU and size.

    target_consumed_tokens overrides the profile's consumed_tokens
    (= corpus_tokens × epochs). Use this only if you intentionally want to
    train for more or fewer steps than the standard recipe.
    """
    if gpu_key not in GPU_SPECS:
        raise ValueError(f"Unknown GPU '{gpu_key}'. Choices: {sorted(GPU_SPECS)}")
    if size not in SIZE_PROFILES:
        raise ValueError(f"Unknown pretrain size '{size}'. Choices: {sorted(SIZE_PROFILES)}")
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if mode_name not in MODES:
        raise ValueError(f"Unknown mode '{mode_name}'. Choices: {sorted(MODES)}")

    spec = GPU_SPECS[gpu_key]
    profile = SIZE_PROFILES[size]
    mode = MODES[mode_name]

    ref_global = target_global_batch or profile.ref_global_batch
    target_consumed = target_consumed_tokens or profile.consumed_tokens

    micro, accum, ckpt, est_vram, budget, warns = _decide_batch_and_ckpt(
        state_gb=profile.state_gb,
        act_no_ckpt=profile.act_per_seq_gb_no_ckpt,
        act_ckpt=profile.act_per_seq_gb_ckpt,
        ref_global_batch=ref_global,
        num_gpus=num_gpus,
        spec=spec,
        mode=mode,
        force_ckpt=force_ckpt,
    )

    actual_global = micro * accum * num_gpus
    tokens_per_step = actual_global * profile.ctx
    max_steps = target_consumed // tokens_per_step
    actual_consumed = max_steps * tokens_per_step
    warmup = max(100, int(max_steps * profile.warmup_frac))

    # Pretrain-specific warnings
    rounding_loss = abs(actual_consumed - target_consumed) / target_consumed
    if rounding_loss > 0.02:
        warns.append(
            f"Consumed-token rounding loses {rounding_loss*100:.1f}% "
            f"({(target_consumed - actual_consumed) / 1e9:+.2f}B). "
            f"max_steps={max_steps:,} × tokens/step={tokens_per_step:,} = "
            f"{actual_consumed/1e9:.2f}B vs target {target_consumed/1e9:.2f}B. "
            f"Consider adjusting target_global_batch."
        )
    if size == "1b" and num_gpus >= 4:
        warns.append(
            "1B model on 4+ GPUs benefits from FSDP (saves ~10 GB/GPU on "
            "optimizer state). Consider `make accel-gen-fsdp GPUS=N` instead "
            "of plain DDP."
        )

    return GeneratedConfig(
        micro_batch_size=micro,
        gradient_accumulation_steps=accum,
        gradient_checkpointing=ckpt,
        max_steps=max_steps,
        warmup_steps=warmup,
        actual_global_batch=actual_global,
        tokens_per_step=tokens_per_step,
        actual_consumed_tokens=actual_consumed,
        estimated_vram_gb=est_vram,
        vram_budget_gb=budget,
        stage="pretrain",
        gpu_key=gpu_key,
        size=size,
        num_gpus=num_gpus,
        mode=mode_name,
        warnings=warns,
    )


# ── SFT ──────────────────────────────────────────────────────────────────────

def _compute_sft_config_with_profile(
    gpu_key: str, size: str, num_gpus: int, mode_name: str,
    profile: SFTProfile,
    target_global_batch: Optional[int],
    force_ckpt: Optional[bool],
    sft_variant: str,   # "chat" or "code" — for stage labelling
) -> GeneratedConfig:
    spec = GPU_SPECS[gpu_key]
    mode = MODES[mode_name]
    ref_global = target_global_batch or profile.ref_global_batch

    micro, accum, ckpt, est_vram, budget, warns = _decide_batch_and_ckpt(
        state_gb=profile.state_gb,
        act_no_ckpt=profile.act_per_seq_gb_no_ckpt,
        act_ckpt=profile.act_per_seq_gb_ckpt,
        ref_global_batch=ref_global,
        num_gpus=num_gpus,
        spec=spec,
        mode=mode,
        force_ckpt=force_ckpt,
    )

    actual_global = micro * accum * num_gpus
    tokens_per_step = actual_global * profile.max_seq_length

    warns.append(
        f"SFT activation estimates are analytical. After your first real "
        f"sft-{sft_variant} run, peak VRAM with `nvidia-smi` and update "
        f"SFT_{sft_variant.upper()}_PROFILES['{size}'] in config_gen/config_gen.py."
    )

    return GeneratedConfig(
        micro_batch_size=micro,
        gradient_accumulation_steps=accum,
        gradient_checkpointing=ckpt,
        actual_global_batch=actual_global,
        tokens_per_step=tokens_per_step,
        estimated_vram_gb=est_vram,
        vram_budget_gb=budget,
        stage=f"sft-{sft_variant}",
        gpu_key=gpu_key,
        size=size,
        num_gpus=num_gpus,
        mode=mode_name,
        warnings=warns,
    )


def compute_sft_chat_config(gpu_key: str, size: str, num_gpus: int,
                            mode_name: str = "balanced",
                            target_global_batch: Optional[int] = None,
                            force_ckpt: Optional[bool] = None) -> GeneratedConfig:
    if size not in SFT_CHAT_PROFILES:
        raise ValueError(f"Unknown SFT size '{size}'. Choices: {sorted(SFT_CHAT_PROFILES)}")
    if gpu_key not in GPU_SPECS:
        raise ValueError(f"Unknown GPU '{gpu_key}'.")
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if mode_name not in MODES:
        raise ValueError(f"Unknown mode '{mode_name}'.")
    return _compute_sft_config_with_profile(
        gpu_key, size, num_gpus, mode_name,
        SFT_CHAT_PROFILES[size], target_global_batch, force_ckpt, "chat",
    )


def compute_sft_code_config(gpu_key: str, size: str, num_gpus: int,
                            mode_name: str = "balanced",
                            target_global_batch: Optional[int] = None,
                            force_ckpt: Optional[bool] = None) -> GeneratedConfig:
    if size not in SFT_CODE_PROFILES:
        raise ValueError(f"Unknown SFT size '{size}'. Choices: {sorted(SFT_CODE_PROFILES)}")
    if gpu_key not in GPU_SPECS:
        raise ValueError(f"Unknown GPU '{gpu_key}'.")
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if mode_name not in MODES:
        raise ValueError(f"Unknown mode '{mode_name}'.")
    return _compute_sft_config_with_profile(
        gpu_key, size, num_gpus, mode_name,
        SFT_CODE_PROFILES[size], target_global_batch, force_ckpt, "code",
    )


# ── DPO ──────────────────────────────────────────────────────────────────────

def compute_dpo_config(gpu_key: str, size: str, num_gpus: int,
                       mode_name: str = "balanced",
                       target_global_batch: Optional[int] = None,
                       force_ckpt: Optional[bool] = None) -> GeneratedConfig:
    if size not in DPO_PROFILES:
        raise ValueError(f"Unknown DPO size '{size}'. Choices: {sorted(DPO_PROFILES)}")
    if gpu_key not in GPU_SPECS:
        raise ValueError(f"Unknown GPU '{gpu_key}'.")
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if mode_name not in MODES:
        raise ValueError(f"Unknown mode '{mode_name}'.")

    spec = GPU_SPECS[gpu_key]
    profile = DPO_PROFILES[size]
    mode = MODES[mode_name]
    ref_global = target_global_batch or profile.ref_global_batch

    micro, accum, ckpt, est_vram, budget, warns = _decide_batch_and_ckpt(
        state_gb=profile.state_gb,
        act_no_ckpt=profile.act_per_seq_gb_no_ckpt,
        act_ckpt=profile.act_per_seq_gb_ckpt,
        ref_global_batch=ref_global,
        num_gpus=num_gpus,
        spec=spec,
        mode=mode,
        force_ckpt=force_ckpt,
    )

    actual_global = micro * accum * num_gpus
    tokens_per_step = actual_global * profile.max_seq_length

    # DPO is sensitive — extra heads-up
    warns.append(
        f"DPO is LR-sensitive. The recipe LR ({profile.lr:.1e}) was tuned for "
        f"global={profile.ref_global_batch}. You're at global={actual_global}"
        + (" ✓." if actual_global == profile.ref_global_batch
           else f" ({actual_global / profile.ref_global_batch:.2f}× off — expect to retune).")
    )
    warns.append(
        f"DPO activation estimates are analytical and account for chosen + "
        f"rejected pairs through both policy and reference models. After your "
        f"first real DPO run, measure peak VRAM and update "
        f"DPO_PROFILES['{size}'] in config_gen/config_gen.py."
    )

    return GeneratedConfig(
        micro_batch_size=micro,
        gradient_accumulation_steps=accum,
        gradient_checkpointing=ckpt,
        actual_global_batch=actual_global,
        tokens_per_step=tokens_per_step,
        estimated_vram_gb=est_vram,
        vram_budget_gb=budget,
        stage="dpo",
        gpu_key=gpu_key,
        size=size,
        num_gpus=num_gpus,
        mode=mode_name,
        warnings=warns,
    )


# ── GPU detection ────────────────────────────────────────────────────────────

def detect_gpu() -> Optional[str]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    name = out.splitlines()[0].strip()
    for fragment, key in NVIDIA_SMI_NAME_MAP:
        if fragment in name:
            return key
    return None


# ── YAML rendering ───────────────────────────────────────────────────────────

def _yaml_header(cfg: GeneratedConfig) -> str:
    spec = GPU_SPECS[cfg.gpu_key]
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    used_pct = cfg.estimated_vram_gb / spec["vram_gb"] * 100

    lines = [
        "# ─────────────────────────────────────────────────────────────────────────────",
        "# Auto-generated by config_gen/config_gen.py",
        f"# Stage:      {cfg.stage}",
        f"# Generated:  {timestamp}",
        "#",
        "# Inputs:",
        f"#   gpu              = {cfg.gpu_key}  ({spec['display']}, {spec['vram_gb']} GB)",
        f"#   size             = {cfg.size}",
        f"#   num_gpus         = {cfg.num_gpus}",
        f"#   mode             = {cfg.mode}  (vram budget {cfg.vram_budget_gb:.0f} GB)",
        f"#   global_batch     = {cfg.actual_global_batch} sequences/step",
    ]
    if cfg.actual_consumed_tokens:
        lines.append(
            f"#   consumed_tokens  = {cfg.actual_consumed_tokens / 1e9:.2f}B  "
            f"(corpus × epochs; sets max_steps)"
        )
    lines.extend([
        "#",
        f"# Estimated peak VRAM: ~{cfg.estimated_vram_gb:.0f} GB / {spec['vram_gb']} GB ({used_pct:.0f}%)",
    ])
    if cfg.warnings:
        lines.append("#")
        lines.append("# Heads-up:")
        for w in cfg.warnings:
            # Wrap at ~76 chars
            wrapped = _wrap_for_yaml(w, prefix="#   • ", continuation="#     ")
            lines.extend(wrapped)
    lines.append("# ─────────────────────────────────────────────────────────────────────────────")
    return "\n".join(lines) + "\n"


def _wrap_for_yaml(text: str, prefix: str, continuation: str, width: int = 76) -> list[str]:
    """Simple word-wrap for warning bullets in the YAML header."""
    words = text.split()
    lines: list[str] = []
    current = prefix
    current_prefix = prefix
    for word in words:
        candidate = current + (" " if current != current_prefix else "") + word
        if len(candidate) > width and current != current_prefix:
            lines.append(current)
            current = continuation + word
            current_prefix = continuation
        else:
            current = candidate
    if current.strip():
        lines.append(current)
    return lines

def _pretrain_eval_save_steps(max_steps: int) -> int:
    """
    Pick eval/save cadence for pretraining.

    Short smoke runs keep frequent evals.
    Long full runs avoid excessive eval/save overhead.
    """
    if max_steps >= 10_000:
        return 5_000
    return max(100, max_steps // 100)

def render_pretrain_yaml(cfg: GeneratedConfig) -> str:
    profile = SIZE_PROFILES[cfg.size]
    return f"""{_yaml_header(cfg)}
name: slm-{cfg.size}
wandb_project: slm

model:
  vocab_size: 32000
  hidden_size: {profile.hidden}
  num_hidden_layers: {profile.layers}
  num_attention_heads: {profile.heads}
  num_key_value_heads: {profile.kv_heads}
  max_position_embeddings: {profile.ctx}
  rope_theta: {profile.rope_theta}
  rms_norm_eps: 1.0e-5
  initializer_range: 0.02
  tie_word_embeddings: true

data:
  val_fraction: 0.005

training:
  # micro × accum × gpus = {cfg.micro_batch_size} × {cfg.gradient_accumulation_steps} × {cfg.num_gpus} = {cfg.actual_global_batch} sequences/step
  # tokens/step     = global × ctx = {cfg.tokens_per_step:,}
  # consumed_tokens = max_steps × tokens/step = {cfg.actual_consumed_tokens / 1e9:.2f}B
  #                   (corpus_tokens × epochs — see config/data_mix.py)
  max_steps: {cfg.max_steps}
  warmup_steps: {cfg.warmup_steps}
  micro_batch_size: {cfg.micro_batch_size}
  gradient_accumulation_steps: {cfg.gradient_accumulation_steps}
  precision: bf16
  gradient_clip_val: 1.0
  gradient_checkpointing: {str(cfg.gradient_checkpointing).lower()}
  torch_compile: true
  torch_compile_backend: inductor
  torch_compile_mode: default
  eval_steps: {_pretrain_eval_save_steps(cfg.max_steps)}
  save_steps: {_pretrain_eval_save_steps(cfg.max_steps)}
  save_total_limit: 3
  log_steps: 10
  num_workers: 8
  seed: 42
  lr_scheduler: cosine
  report_to:
    - wandb

optimizer:
  lr: {profile.lr:.1e}
  weight_decay: {profile.weight_decay}
  beta1: {profile.beta1}
  beta2: {profile.beta2}
  eps: {profile.eps}
"""


def _render_sft_yaml(cfg: GeneratedConfig, profile: SFTProfile,
                     base_model_path: str, train_path: str, val_path: str,
                     out_name: str) -> str:
    # Eval micro-batch defaults to half train micro-batch. Eval can materialize
    # large logits and does not accumulate gradients, so halving keeps eval safer.
    eval_micro = max(1, cfg.micro_batch_size // 2)

    return f"""{_yaml_header(cfg)}
name: slm-{cfg.size}-{out_name}
wandb_project: slm

model:
  base_model_path: {base_model_path}
  max_seq_length: {profile.max_seq_length}

data:
  train_path: {train_path}
  val_path:   {val_path}
  packing: {str(profile.packing).lower()}

training:
  # micro × accum × gpus = {cfg.micro_batch_size} × {cfg.gradient_accumulation_steps} × {cfg.num_gpus} = {cfg.actual_global_batch} sequences/step
  #
  # Warmup recipe: {profile.warmup_ratio:.3f} of total steps. Computed at
  # runtime by train_sft.py from the resolved total_steps and passed to
  # TrainingArguments as warmup_steps. We don't set warmup_ratio in this
  # YAML because TRL deprecated it in v5.2 in favor of warmup_steps.
  warmup_ratio_recipe: {profile.warmup_ratio}
  epochs: {profile.epochs}
  micro_batch_size: {cfg.micro_batch_size}
  eval_micro_batch_size: {eval_micro}
  gradient_accumulation_steps: {cfg.gradient_accumulation_steps}
  precision: bf16
  gradient_clip_val: 1.0
  gradient_checkpointing: {str(cfg.gradient_checkpointing).lower()}
  torch_compile: {str(profile.torch_compile).lower()}
  eval_steps: {profile.eval_steps}
  save_steps: {profile.save_steps}
  save_total_limit: 3
  log_steps: 10
  num_workers: 4
  seed: 42
  lr_scheduler: cosine
  report_to:
    - wandb

optimizer:
  lr: {profile.lr}
  weight_decay: {profile.weight_decay}
  beta1: {profile.beta1}
  beta2: {profile.beta2}
"""


def render_sft_chat_yaml(cfg: GeneratedConfig) -> str:
    profile = SFT_CHAT_PROFILES[cfg.size]
    return _render_sft_yaml(
        cfg, profile,
        base_model_path=f"$RESULTS_DIR/slm-{cfg.size}/final",
        train_path="$DATA_DIR/sft/chat/train.jsonl",
        val_path="$DATA_DIR/sft/chat/val.jsonl",
        out_name="chat",
    )


def render_sft_code_yaml(cfg: GeneratedConfig) -> str:
    profile = SFT_CODE_PROFILES[cfg.size]
    return _render_sft_yaml(
        cfg, profile,
        base_model_path=f"$RESULTS_DIR/slm-{cfg.size}-chat/final",
        train_path="$DATA_DIR/sft/code/train.jsonl",
        val_path="$DATA_DIR/sft/code/val.jsonl",
        out_name="chat-code",
    )


def render_dpo_yaml(cfg: GeneratedConfig) -> str:
    profile = DPO_PROFILES[cfg.size]
    eval_micro = max(1, cfg.micro_batch_size // 2)

    return f"""{_yaml_header(cfg)}
name: slm-{cfg.size}-dpo
wandb_project: slm

model:
  base_model_path: $RESULTS_DIR/slm-{cfg.size}-chat-code/final
  max_seq_length: {profile.max_seq_length}

data:
  train_path: $DATA_DIR/dpo/train.jsonl
  val_path:   $DATA_DIR/dpo/val.jsonl

dpo:
  beta: {profile.dpo_beta}
  max_prompt_length: {profile.max_prompt_length}

training:
  # micro × accum × gpus = {cfg.micro_batch_size} × {cfg.gradient_accumulation_steps} × {cfg.num_gpus} = {cfg.actual_global_batch} pairs/step
  #
  # Warmup recipe: {profile.warmup_ratio:.3f} of total steps. Computed at
  # runtime by train_dpo.py from the resolved total_steps and passed to
  # DPOConfig as warmup_steps.
  warmup_ratio_recipe: {profile.warmup_ratio}
  epochs: {profile.epochs}
  micro_batch_size: {cfg.micro_batch_size}
  eval_micro_batch_size: {eval_micro}
  gradient_accumulation_steps: {cfg.gradient_accumulation_steps}
  precision: bf16
  gradient_clip_val: 1.0
  gradient_checkpointing: {str(cfg.gradient_checkpointing).lower()}
  torch_compile: {str(profile.torch_compile).lower()}
  eval_steps: {profile.eval_steps}
  save_steps: {profile.save_steps}
  save_total_limit: 3
  log_steps: 10
  num_workers: 4
  seed: 42
  lr_scheduler: cosine
  report_to:
    - wandb

optimizer:
  lr: {profile.lr}
  weight_decay: {profile.weight_decay}
  beta1: {profile.beta1}
  beta2: {profile.beta2}
"""


# ── Plan summary ─────────────────────────────────────────────────────────────

def render_plan(cfg: GeneratedConfig) -> str:
    spec = GPU_SPECS[cfg.gpu_key]
    used_pct = cfg.estimated_vram_gb / spec["vram_gb"] * 100

    lines = [
        "─" * 76,
        f"  {cfg.stage}: slm-{cfg.size} on {spec['display']} × {cfg.num_gpus}  [{cfg.mode}]",
        "─" * 76,
        f"  micro_batch_size            = {cfg.micro_batch_size}",
        f"  gradient_accumulation_steps = {cfg.gradient_accumulation_steps}",
        f"  global_batch (sequences)    = {cfg.actual_global_batch}",
        f"  tokens / step               = {cfg.tokens_per_step:,}",
    ]
    if cfg.max_steps is not None:
        lines.append(f"  max_steps                   = {cfg.max_steps:,}")
        lines.append(f"  warmup_steps                = {cfg.warmup_steps:,}")
        lines.append(
            f"  consumed tokens             = {cfg.actual_consumed_tokens / 1e9:.2f}B  "
            f"(corpus × epochs)"
        )
    lines.extend([
        f"  gradient_checkpointing      = {cfg.gradient_checkpointing}",
        "",
        f"  estimated peak VRAM         = {cfg.estimated_vram_gb:.1f} / {spec['vram_gb']} GB ({used_pct:.0f}%)",
        f"  vram_budget                 = {cfg.vram_budget_gb:.0f} GB",
        "─" * 76,
    ])

    if cfg.warnings:
        lines.append("")
        lines.append("  things to verify:")
        for w in cfg.warnings:
            wrapped = _wrap_for_yaml(w, prefix="    • ", continuation="      ", width=72)
            lines.extend(wrapped)
        lines.append("─" * 76)

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _resolve_gpu(args) -> str:
    if args.detect:
        gpu_key = detect_gpu()
        if gpu_key is None:
            print("ERROR: GPU detection failed. Use --gpu <name>.", file=sys.stderr)
            sys.exit(2)
        if not args.quiet:
            print(f"Detected GPU: {GPU_SPECS[gpu_key]['display']}", file=sys.stderr)
        return gpu_key
    return args.gpu


def _write_or_print(text: str, output: Optional[Path], quiet: bool) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        if not quiet:
            print(f"Wrote {output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-GPU training configs.",
    )
    parser.add_argument("--stage", choices=["pretrain", "sft", "dpo"],
                        default="pretrain")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gpu", choices=sorted(GPU_SPECS))
    src.add_argument("--detect", action="store_true")

    parser.add_argument("--size", required=True,
                        help="125m, 350m, or 1b")
    parser.add_argument("--gpus", type=int, default=1)

    parser.add_argument("--mode", choices=sorted(MODES), default="balanced")
    parser.add_argument("--aggressive", action="store_const", const="aggressive",
                        dest="mode_alias",
                        help="Alias for --mode aggressive (backwards compat)")
    parser.add_argument("--conservative", action="store_const", const="conservative",
                        dest="mode_alias",
                        help="Alias for --mode conservative")

    parser.add_argument("--target-global-batch", type=int, default=None)
    parser.add_argument("--target-consumed-tokens", type=int, default=None,
                        help="(pretrain only) Override consumed tokens "
                             "(corpus_tokens × epochs). Used to compute "
                             "max_steps. Not the public corpus figure.")
    # Deprecated alias kept for backward compat.
    parser.add_argument("--target-tokens", type=int, default=None,
                        dest="target_tokens_deprecated",
                        help=argparse.SUPPRESS)

    ckpt_grp = parser.add_mutually_exclusive_group()
    ckpt_grp.add_argument("--ckpt", dest="force_ckpt", action="store_true",
                          default=None)
    ckpt_grp.add_argument("--no-ckpt", dest="force_ckpt", action="store_false",
                          default=None)

    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output file. For --stage sft, this is the chat "
                             "output; the code config is written to a sibling "
                             "named sft_code_<size>.yaml.")
    parser.add_argument("--output-code", type=Path, default=None,
                        help="(--stage sft only) Output for the code config. "
                             "Defaults to sibling of --output.")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    gpu_key = _resolve_gpu(args)
    mode_name = args.mode_alias or args.mode

    # Resolve token target — new flag wins, fall back to deprecated alias.
    consumed_override = args.target_consumed_tokens
    if consumed_override is None and args.target_tokens_deprecated is not None:
        if not args.quiet:
            print(
                "WARN: --target-tokens is deprecated; use "
                "--target-consumed-tokens instead.",
                file=sys.stderr,
            )
        consumed_override = args.target_tokens_deprecated

    try:
        if args.stage == "pretrain":
            cfg = compute_pretrain_config(
                gpu_key=gpu_key, size=args.size, num_gpus=args.gpus,
                mode_name=mode_name,
                target_global_batch=args.target_global_batch,
                target_consumed_tokens=consumed_override,
                force_ckpt=args.force_ckpt,
            )
            _write_or_print(render_pretrain_yaml(cfg), args.output, args.quiet)
            if not args.quiet:
                print(render_plan(cfg), file=sys.stderr)

        elif args.stage == "sft":
            chat = compute_sft_chat_config(
                gpu_key=gpu_key, size=args.size, num_gpus=args.gpus,
                mode_name=mode_name,
                target_global_batch=args.target_global_batch,
                force_ckpt=args.force_ckpt,
            )
            code = compute_sft_code_config(
                gpu_key=gpu_key, size=args.size, num_gpus=args.gpus,
                mode_name=mode_name,
                target_global_batch=args.target_global_batch,
                force_ckpt=args.force_ckpt,
            )
            chat_out = args.output
            code_out = args.output_code
            if chat_out and not code_out:
                # Default: sibling file with sft_code_ prefix
                code_out = chat_out.parent / chat_out.name.replace("sft_chat_", "sft_code_")
                if code_out == chat_out:
                    # Fallback if naming convention isn't followed
                    code_out = chat_out.with_name(chat_out.stem + "_code" + chat_out.suffix)
            _write_or_print(render_sft_chat_yaml(chat), chat_out, args.quiet)
            _write_or_print(render_sft_code_yaml(code), code_out, args.quiet)
            if not args.quiet:
                print(render_plan(chat), file=sys.stderr)
                print(render_plan(code), file=sys.stderr)

        elif args.stage == "dpo":
            cfg = compute_dpo_config(
                gpu_key=gpu_key, size=args.size, num_gpus=args.gpus,
                mode_name=mode_name,
                target_global_batch=args.target_global_batch,
                force_ckpt=args.force_ckpt,
            )
            _write_or_print(render_dpo_yaml(cfg), args.output, args.quiet)
            if not args.quiet:
                print(render_plan(cfg), file=sys.stderr)

    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())