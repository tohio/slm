"""
pretrain/train.py
-----------------
NeMo 2.x GPT pre-training entry point.
Called by train.sh via torchrun or direct python.

NeMo 2.x uses megatron-core directly for GPT training.
The pytorch_lightning Trainer is still used for orchestration
but NeMo 2.x wraps it with its own MegatronStrategy.

All hyperparameters come from the YAML config via Hydra,
with CLI overrides supported.
"""

import os
import torch
from omegaconf import OmegaConf, DictConfig

import hydra
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

# NeMo 2.x GPT model — uses megatron-core under the hood
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

# NeMo 2.x trainer — wraps pytorch_lightning with Megatron strategy
from nemo.lightning import MegatronStrategy, Trainer


@hydra_runner(config_path="configs", config_name="gpt_125m")
def main(cfg: DictConfig) -> None:
    logging.info("\n\n============ SLM Pre-Training ============")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Trainer ────────────────────────────────────────────────────────────
    # NeMo 2.x uses nemo.lightning.Trainer which handles:
    #   - Megatron tensor/pipeline parallelism
    #   - BF16/FP16 precision via MegatronStrategy
    #   - NCCL communication for multi-GPU
    trainer = Trainer(
        strategy=MegatronStrategy(
            tensor_model_parallel_size=cfg.model.get("tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=cfg.model.get("pipeline_model_parallel_size", 1),
            gradient_as_bucket_view=cfg.model.get("gradient_as_bucket_view", True),
        ),
        **cfg.trainer,
    )

    # ── Experiment manager ─────────────────────────────────────────────────
    # Handles checkpointing, W&B logging, resume logic
    exp_manager(trainer, cfg.get("exp_manager", None))

    # ── Model ──────────────────────────────────────────────────────────────
    model = MegatronGPTModel(cfg.model, trainer)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model parameters: {param_count:.1f}M")
    logging.info(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logging.info(f"  GPU {i}: {props.name} | {props.total_memory // 1024**3}GB")

    # ── Training ───────────────────────────────────────────────────────────
    trainer.fit(model)
    logging.info("Pre-training complete.")


if __name__ == "__main__":
    main()