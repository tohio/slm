"""
finetune/train_sft.py
---------------------
NeMo Aligner SFT entry point.
Handles both chat and code fine-tuning stages — config controls behavior.

Key difference from pre-training:
  answer_only_loss=true → loss computed only on assistant turns,
  not the human prompt. This prevents the model from "learning"
  to generate user messages and diluting the gradient signal.

Used by:
  bash finetune/scripts/train_sft.sh --stage chat
  bash finetune/scripts/train_sft.sh --stage code
"""

import torch
from omegaconf import OmegaConf, DictConfig

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

# NeMo Aligner 0.7.0 imports
# GPTSFTModel handles answer_only_loss and chat template formatting
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    resolve_and_create_trainer,
)


@hydra_runner(config_path="configs", config_name="sft_chat")
def main(cfg: DictConfig) -> None:
    logging.info("\n\n============ SLM SFT Training ============")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Trainer ────────────────────────────────────────────────────────────
    # resolve_and_create_trainer handles MegatronStrategy setup for NeMo Aligner
    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.get("exp_manager", None))

    # ── Model ──────────────────────────────────────────────────────────────
    # GPTSFTModel wraps MegatronGPTModel with:
    #   - answer_only_loss: masks loss on prompt tokens
    #   - chat template formatting via data.chat=true
    model = GPTSFTModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=cfg.model,
        strict=True,
    )

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model parameters:  {param_count:.1f}M")
    logging.info(f"answer_only_loss:  {cfg.model.get('answer_only_loss', False)}")
    logging.info(f"chat mode:         {cfg.model.data.get('chat', False)}")

    # ── Training ───────────────────────────────────────────────────────────
    trainer.fit(model)
    logging.info("SFT training complete.")


if __name__ == "__main__":
    main()