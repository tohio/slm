"""
finetune/train_sft.py
---------------------
NeMo Aligner SFT entry point.
Handles both chat and code fine-tuning stages — config controls behavior.

Key difference from pre-training:
  answer_only_loss=true → loss computed only on assistant turns,
  not the human prompt. This prevents the model from "learning"
  to generate user messages.
"""

import torch
from omegaconf import OmegaConf, DictConfig

import hydra
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

from pytorch_lightning import Trainer

# NeMo Aligner SFT
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    init_distributed,
    resolve_and_create_trainer,
)


@hydra_runner(config_path="configs", config_name="sft_chat")
def main(cfg: DictConfig) -> None:
    logging.info("\n\n============ SLM SFT Training ============")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.get("exp_manager", None))

    # ── Model ──────────────────────────────────────────────────────────────────
    # GPTSFTModel wraps MegatronGPTModel with SFT-specific loss
    model = GPTSFTModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=cfg.model,
        save_restore_connector=None,
        strict=True,
    )

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model parameters: {param_count:.1f}M")
    logging.info(f"answer_only_loss: {cfg.model.get('answer_only_loss', False)}")

    # ── Training ───────────────────────────────────────────────────────────────
    trainer.fit(model)

    logging.info("SFT training complete.")


if __name__ == "__main__":
    main()
