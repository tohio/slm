"""
scripts/reinit_special_embeds.py
--------------------------------
Re-initialize chat-template special token embeddings before SFT.

This script intentionally avoids SLMForCausalLM.from_pretrained() and
SLMForCausalLM.save_pretrained().

Why:
    The current HF/custom-model integration has shown unsafe behavior when
    loading this model through from_pretrained(). This script uses direct
    safetensors I/O for both load and save.

Usage:
    Default behavior:
        python scripts/reinit_special_embeds.py --size 125m

    Safer explicit recovery behavior:
        python scripts/reinit_special_embeds.py \
          --src results/slm-125m/checkpoint-152000 \
          --dst results/slm-125m/final

    Via Makefile:
        make reinit-embeds SIZE=125m
"""

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safetensors.torch
import torch
from model import SLMForCausalLM
from transformers import AutoConfig, PreTrainedTokenizerFast


SPECIALS_TO_REINIT = [
    "<BOS>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endofturn|>",
]


def parse_args():
    p = argparse.ArgumentParser(description="Reinitialize chat special-token embeddings.")

    p.add_argument(
        "--size",
        default="125m",
        help="Model size, matching results/slm-<size>/final. Default: 125m",
    )

    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Source checkpoint directory. Default: results/slm-<size>/final",
    )

    p.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination checkpoint directory. Default: same as --src",
    )

    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup of dst before overwriting it.",
    )

    return p.parse_args()


def safe_load_model(ckpt: Path) -> SLMForCausalLM:
    """
    Safe SLM loader.

    Uses:
        AutoConfig -> SLMForCausalLM(config) -> safetensors -> load_state_dict

    Avoids:
        SLMForCausalLM.from_pretrained()
    """
    config = AutoConfig.from_pretrained(str(ckpt))
    model = SLMForCausalLM(config)

    weights_path = ckpt / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    result = model.load_state_dict(state_dict, strict=False)

    allowed_missing = set()
    if getattr(config, "tie_word_embeddings", False):
        allowed_missing.add("lm_head.weight")

    unexpected_missing = [k for k in result.missing_keys if k not in allowed_missing]

    if unexpected_missing:
        raise RuntimeError(f"Missing keys while loading {ckpt}: {unexpected_missing}")

    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading {ckpt}: {result.unexpected_keys}")

    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()

    model.eval()
    return model


def safe_save_model(model: SLMForCausalLM, dst: Path) -> None:
    """
    Safe SLM saver.

    Saves weights directly through safetensors.

    Avoids:
        SLMForCausalLM.save_pretrained()

    Note:
        This function intentionally writes only model.safetensors.
        The caller should copy src -> dst first to preserve config.json,
        tokenizer/, generation_config.json, auto_map, and any repo files.
    """
    dst.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    cleaned_state_dict = {}

    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)

    for k, v in state_dict.items():
        if tie_word_embeddings and k == "lm_head.weight":
            # lm_head.weight is tied to model.embed_tokens.weight.
            # Drop it to avoid safetensors shared-storage issues.
            # It is restored by model.tie_weights() on load.
            continue

        cleaned_state_dict[k] = v.detach().cpu().contiguous()

    if not tie_word_embeddings and "lm_head.weight" not in cleaned_state_dict:
        raise RuntimeError(
            "tie_word_embeddings=False but lm_head.weight is missing from save. "
            "This would produce a broken checkpoint."
        )

    safetensors.torch.save_file(
        cleaned_state_dict,
        str(dst / "model.safetensors"),
    )


def make_backup(dst: Path) -> None:
    if not dst.exists():
        return

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = dst.parent / f"{dst.name}.backup-before-reinit-{ts}"

    print(f"Creating backup: {backup_dir}")
    shutil.copytree(dst, backup_dir)


def validate_special_ids(tok: PreTrainedTokenizerFast, specials: list[str]) -> list[int]:
    ids = []

    for s in specials:
        token_id = tok.convert_tokens_to_ids(s)

        if token_id is None or token_id == tok.unk_token_id:
            raise ValueError(f"Special token {s!r} not found in tokenizer vocab")

        ids.append(token_id)

    return ids


def log_ignored_additional_specials(tok: PreTrainedTokenizerFast) -> None:
    configured = set(SPECIALS_TO_REINIT)
    additional = set(tok.special_tokens_map.get("additional_special_tokens", []))
    ignored = sorted(additional - configured)

    if ignored:
        print("Note: tokenizer has additional_special_tokens not being reinitialized:")
        for token in ignored:
            print(f"  - {token}")
        print("If these are new chat/control tokens, add them to SPECIALS_TO_REINIT.")


def main():
    args = parse_args()

    src = args.src or Path(f"results/slm-{args.size}/final")
    dst = args.dst or src

    if not src.exists():
        sys.exit(
            f"Source checkpoint not found: {src}\n"
            f"Run 'make pretrain SIZE={args.size}' first, or pass --src explicitly."
        )

    if not (src / "model.safetensors").exists():
        sys.exit(f"Source weights not found: {src / 'model.safetensors'}")

    if not (src / "tokenizer").exists():
        sys.exit(f"Tokenizer directory not found in source checkpoint: {src / 'tokenizer'}")

    if dst.exists() and not args.no_backup:
        make_backup(dst)

    print(f"Source checkpoint:      {src}")
    print(f"Destination checkpoint: {dst}")

    # Conservative behavior:
    #   1. Copy the entire source checkpoint/repo directory to dst.
    #   2. Patch only dst/model.safetensors.
    #
    # This preserves config.json, tokenizer/, generation_config.json,
    # auto_map, code files, model card files, and anything else in src.
    if src.resolve() != dst.resolve():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    model = safe_load_model(src)
    tok = PreTrainedTokenizerFast.from_pretrained(str(src / "tokenizer"))

    try:
        ids = validate_special_ids(tok, SPECIALS_TO_REINIT)
    except ValueError as e:
        sys.exit(str(e))

    log_ignored_additional_specials(tok)

    print("Special token IDs:", dict(zip(SPECIALS_TO_REINIT, ids)))

    emb = model.get_input_embeddings().weight.data
    before = emb.detach().clone()

    mask = torch.ones(emb.size(0), dtype=torch.bool)
    mask[ids] = False

    mean_emb = emb[mask].mean(dim=0)

    with torch.no_grad():
        for token_id in ids:
            emb[token_id] = mean_emb

    changed_rows = sorted(
        (before - emb)
        .abs()
        .sum(dim=1)
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
    )

    expected_rows = sorted(ids)

    if changed_rows != expected_rows:
        raise RuntimeError(
            "Unexpected embedding rows changed.\n"
            f"Expected changed rows: {expected_rows}\n"
            f"Actual changed rows:   {changed_rows}"
        )

    safe_save_model(model, dst)

    print(f"Re-initialized {len(ids)} special-token embeddings.")
    print(f"Saved patched checkpoint to: {dst}")
    print("Only these rows changed:", dict(zip(SPECIALS_TO_REINIT, ids)))


if __name__ == "__main__":
    main()