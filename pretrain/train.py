"""
pretrain/train.py
-----------------
NeMo GPT pre-training entry point.
Called by train.sh via torchrun or direct python.

NeMo uses Hydra for config management — all hyperparameters
come from the YAML config, with CLI overrides supported.
"""

import os
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer

import hydra

# NeMo imports
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="configs", config_name="gpt_125m")
def main(cfg: DictConfig) -> None:
    logging.info("\n\n============ SLM Pre-Training ============")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Precision plugin ───────────────────────────────────────────────────────
    # BF16 on A100/A6000 — no loss scaling needed (unlike FP16)
    plugins = []
    if cfg.trainer.precision in ["bf16", "bf16-mixed"]:
        plugins.append(
            MegatronHalfPrecisionPlugin(
                precision=cfg.trainer.precision,
                device="cuda",
            )
        )
    elif cfg.trainer.precision in [16, "fp16", "16-mixed"]:
        plugins.append(
            MegatronHalfPrecisionPlugin(
                precision="16-mixed",
                device="cuda",
                scaler=GradScaler(
                    init_scale=cfg.model.get("native_amp_init_scale", 2**32),
                    growth_interval=cfg.model.get("native_amp_growth_interval", 1000),
                ),
            )
        )

    # ── DDP Strategy ───────────────────────────────────────────────────────────
    # NLPDDPStrategy handles Megatron's model parallelism on top of PyTorch DDP
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,   # Megatron handles its own gradient sync
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        plugins=plugins,
        strategy=strategy,
        **cfg.trainer,
    )

    # ── Experiment manager ─────────────────────────────────────────────────────
    # Handles checkpointing, logging, W&B integration
    exp_manager(trainer, cfg.get("exp_manager", None))

    # ── Model ──────────────────────────────────────────────────────────────────
    model = MegatronGPTModel(cfg.model, trainer)

    logging.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )

    # ── Training ───────────────────────────────────────────────────────────────
    trainer.fit(model)

    logging.info("Pre-training complete.")


if __name__ == "__main__":
    main()
