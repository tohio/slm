"""
inference/utils.py
-------------------
Shared helpers for inference scripts (chat.py, generate.py).

Centralises:
    - Model + tokenizer loading from local paths and the HuggingFace Hub
    - Special-token ID resolution from the actual tokenizer (no hardcoded IDs)
    - Registration of SLMConfig / SLMForCausalLM with AutoConfig / AutoModel

Hardcoding token IDs (PAD=0, EOS=3, ENDOFTURN=7) is fragile: a tokenizer
retrain that adds or reorders specials will silently move the IDs, and
generation will stop on the wrong token. Reading from the loaded tokenizer
eliminates the drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# Special-token string constants. Keep in sync with tokenizer/train_tokenizer.py;
# these are the NAMES (stable), not the IDs (which depend on tokenizer version).
PAD_TOKEN       = "<PAD>"
EOS_TOKEN       = "<EOS>"
BOS_TOKEN       = "<BOS>"
ENDOFTURN_TOKEN = "<|endofturn|>"


@dataclass(frozen=True)
class SpecialTokenIds:
    """
    Resolved special-token IDs for a loaded tokenizer.

    eos_list is the list of IDs treated as end-of-generation by model.generate —
    both <EOS> (sequence end) and <|endofturn|> (end of one assistant turn).
    """
    pad:       int
    eos:       int
    bos:       int
    endofturn: int

    @property
    def eos_list(self) -> list[int]:
        return [self.eos, self.endofturn]


def _require_token_id(tokenizer, token: str) -> int:
    """
    Resolve a token string to its ID, asserting it's not the unknown-token fallback.
    convert_tokens_to_ids returns unk_token_id for unknown strings, which would
    silently break generation — we fail loudly instead.
    """
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id   = tokenizer.unk_token_id
    if token_id is None or (unk_id is not None and token_id == unk_id):
        raise ValueError(
            f"Tokenizer is missing required special token {token!r}. "
            f"Retrain the tokenizer: python tokenizer/train_tokenizer.py"
        )
    return token_id


def resolve_special_token_ids(tokenizer) -> SpecialTokenIds:
    """Read all SLM special-token IDs from the loaded tokenizer."""
    return SpecialTokenIds(
        pad       = _require_token_id(tokenizer, PAD_TOKEN),
        eos       = _require_token_id(tokenizer, EOS_TOKEN),
        bos       = _require_token_id(tokenizer, BOS_TOKEN),
        endofturn = _require_token_id(tokenizer, ENDOFTURN_TOKEN),
    )


def resolve_tokenizer_path(model_path: str) -> str:
    """
    Resolve where to load the tokenizer from.

    For local checkpoints: prefer the tokenizer/ subdirectory written by
    train.py / train_sft.py / train_dpo.py, fall back to the model root.
    For Hub IDs (no local path exists): pass the ID through unchanged;
    from_pretrained handles Hub resolution.
    """
    local_path = Path(model_path)
    if local_path.exists():
        sub = local_path / "tokenizer"
        if (sub / "tokenizer_config.json").exists():
            return str(sub)
        return str(local_path)
    return model_path  # Hub ID


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    require_chat_template: bool = True,
):
    """
    Load an SLM model and its tokenizer, registering the custom architecture
    with AutoConfig / AutoModelForCausalLM first.

    Returns (model, tokenizer, special_ids).

    Raises:
        ValueError if require_chat_template is True and the tokenizer has
        no chat_template set — prevents inference/export from running with
        a tokenizer that'd produce garbage chat output.
    """
    import sys
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

    # Ensure repo root is importable so `model` package resolves.
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from model import SLMConfig, SLMForCausalLM

    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    log.info(f"Loading model from {model_path} (dtype={dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()

    tokenizer_path = resolve_tokenizer_path(model_path)
    log.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    if require_chat_template and not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer at {tokenizer_path} has no chat_template. "
            f"Retrain the tokenizer: python tokenizer/train_tokenizer.py"
        )

    special_ids = resolve_special_token_ids(tokenizer)
    log.info(
        f"Special token IDs: PAD={special_ids.pad}, BOS={special_ids.bos}, "
        f"EOS={special_ids.eos}, ENDOFTURN={special_ids.endofturn}"
    )

    return model, tokenizer, special_ids