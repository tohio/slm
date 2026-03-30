"""
alignment/train_dpo.py
-----------------------
NeMo-Aligner DPO entry point (nvcr.io/nvidia/nemo:25.02).

DPO reference: Rafailov et al., 2023 — https://arxiv.org/abs/2305.18290

How DPO works in NeMo-Aligner:
  1. Load SFT checkpoint as both trainable policy AND frozen reference
  2. For each (prompt, chosen, rejected) triplet:
     - Compute log-probs of chosen/rejected under policy
     - Compute log-probs of chosen/rejected under reference (frozen)
     - DPO loss = -log(sigmoid(beta * ((log_pi_chosen - log_ref_chosen)
                                     - (log_pi_rejected - log_ref_rejected))))
  3. Gradient update on policy only — reference stays frozen

Called by:
  bash alignment/scripts/train_dpo.sh

Base container: nvcr.io/nvidia/nemo:25.02
NeMo-Aligner:  0.7.0
"""

import torch
from omegaconf import OmegaConf, DictConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo_aligner.models.nlp.gpt.gpt_dpo_model import GPTDPOModel
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    resolve_and_create_trainer,
)


@hydra_runner(config_path="configs", config_name="dpo")
def main(cfg: DictConfig) -> None:
    logging.info("\n\n============ SLM DPO Alignment ============")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logging.info(f"DPO beta: {cfg.model.dpo.beta}")

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = resolve_and_create_trainer(cfg, "dpo")
    exp_manager(trainer, cfg.get("exp_manager", None))

    # ── Model ──────────────────────────────────────────────────────────────────
    # GPTDPOModel internally manages both the trainable policy
    # and the frozen reference model from the same SFT checkpoint.
    model = GPTDPOModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=cfg.model,
        strict=True,
    )

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logging.info(f"Total parameters:     {param_count:.1f}M")
    logging.info(f"Trainable parameters: {trainable_count:.1f}M")
    logging.info("Reference model is frozen (DPO)")

    # ── Training ───────────────────────────────────────────────────────────────
    trainer.fit(model)
    logging.info("DPO alignment complete.")

    best_ckpt = getattr(trainer.checkpoint_callback, "best_model_path", "see /results/slm_dpo/")
    logging.info(f"Final checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()