"""
finetune/train_sft.py
---------------------
NeMo-Aligner SFT entry point (nvcr.io/nvidia/nemo:25.02).

Handles both chat and code fine-tuning stages.
Loads from a mcore_gpt.nemo checkpoint produced by make convert-pretrain.

Key NeMo-Aligner features used:
  - GPTSFTModel: answer_only_loss=True masks loss on prompt tokens
  - resolve_and_create_trainer: sets up MegatronStrategy for NeMo-Aligner
  - chat=True: applies <|user|>/<|assistant|> template formatting

Called by:
  bash finetune/scripts/train_sft.sh --stage chat
  bash finetune/scripts/train_sft.sh --stage code

Base container: nvcr.io/nvidia/nemo:25.02
NeMo-Aligner:  0.7.0
"""

import torch
from omegaconf import OmegaConf, DictConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    resolve_and_create_trainer,
)


@hydra_runner(config_path="configs", config_name="sft_chat")
def main(cfg: DictConfig) -> None:
    logging.info("\n\n============ SLM SFT Training ============")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Trainer ────────────────────────────────────────────────────────────────
    # resolve_and_create_trainer handles MegatronStrategy for NeMo-Aligner
    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.get("exp_manager", None))

    # ── Model ──────────────────────────────────────────────────────────────────
    # GPTSFTModel wraps MegatronGPTModel with answer_only_loss and chat support.
    # restore_from_path must point to a mcore_gpt.nemo file.
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

    # ── Training ───────────────────────────────────────────────────────────────
    trainer.fit(model)
    logging.info("SFT training complete.")


if __name__ == "__main__":
    main()