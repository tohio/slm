"""
alignment/train_dpo.py
-----------------------
NeMo Aligner DPO (Direct Preference Optimization) entry point.

DPO reference:
  Rafailov et al., 2023 — https://arxiv.org/abs/2305.18290

How DPO works in NeMo Aligner:
  1. Load SFT checkpoint as both the trainable policy AND frozen reference
  2. For each (prompt, chosen, rejected) triplet:
     - Compute log-probs of chosen and rejected under policy
     - Compute log-probs of chosen and rejected under reference (frozen)
     - DPO loss = -log(sigmoid(beta * ((log_pi_chosen - log_ref_chosen)
                                     - (log_pi_rejected - log_ref_rejected))))
  3. Gradient update on policy only — reference stays frozen

Key hyperparameter: beta
  Controls the KL penalty from reference policy.
  Low beta (0.05–0.1): more aggressive preference optimization
  High beta (0.5+):    stay closer to SFT behavior
  Start at 0.1, adjust based on reward metrics.
"""

import torch
from omegaconf import OmegaConf, DictConfig

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

# NeMo Aligner DPO
from nemo_aligner.models.nlp.gpt.gpt_dpo_model import GPTDPOModel
from nemo_aligner.utils.train_script_utils import resolve_and_create_trainer
from nemo_aligner.utils.distributed import Timer


@hydra_runner(config_path="configs", config_name="dpo")
def main(cfg: DictConfig) -> None:
    logging.info("\n\n============ SLM DPO Alignment ============")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logging.info(f"DPO beta: {cfg.model.dpo.beta}")

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = resolve_and_create_trainer(cfg, "dpo")
    exp_manager(trainer, cfg.get("exp_manager", None))

    # ── Model ──────────────────────────────────────────────────────────────────
    # GPTDPOModel internally manages both policy and reference model.
    # The reference model is loaded from the same checkpoint but frozen.
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
    logging.info(f"Reference model is frozen (DPO)")

    # ── Training ───────────────────────────────────────────────────────────────
    timer = Timer(cfg.get("max_time_per_run", "0:23:30:00"))  # safety timer
    trainer.fit(model)

    logging.info("DPO alignment complete.")
    logging.info(f"Final checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
