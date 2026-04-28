"""
scripts/reinit_special_embeds.py
--------------------------------
Re-initialize chat-template special token embeddings before SFT.

Why:
    The base model's tokenizer reserves IDs for chat tokens (<BOS>,
    <|system|>, <|user|>, <|assistant|>, <|endofturn|>) but the
    pretraining corpus (FineWeb, Wikipedia, pg19, peS2o, etc.) does not
    contain these markers. So those embedding rows received essentially
    no gradient signal during pretraining and remain at their random
    init values.

    When SFT starts, every training example is wrapped in these tokens
    via the chat template. The model then has to predict assistant
    content conditioned on garbage embeddings, which sends initial loss
    far above the pretraining baseline and wastes early SFT steps just
    learning sensible representations for the structure tokens.

    Replacing each unseen special token's embedding with the mean of
    all *seen* token embeddings gives them a sensible prior — close to
    the centroid of the learned embedding manifold, so the model treats
    them as "average tokens" rather than outliers, and SFT loss starts
    near the pretraining loss instead of ~11.

    EOS is *not* reinitialized: it was used as a document separator in
    pretraining (verified ~1400 occurrences per 1M tokens in train.bin),
    so its embedding is already trained. Touching it would be harmful.

Usage:
    Run once, after pretraining and before SFT:
        make reinit-embeds SIZE=125m
        # or directly:
        python scripts/reinit_special_embeds.py --size 125m

    Overwrites results/slm-<size>/final/ in place. Idempotent in effect
    (running twice just sets them to the same mean again), but avoid
    running it after SFT has started or you'll erase learned weights.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model import SLMForCausalLM
from transformers import PreTrainedTokenizerFast


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument(
        "--size",
        default="125m",
        help="Model size, matching results/slm-<size>/final (default: 125m)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = Path(f"results/slm-{args.size}/final")

    if not ckpt.exists():
        sys.exit(
            f"Checkpoint not found: {ckpt}\n"
            f"Run 'make pretrain SIZE={args.size}' first."
        )

    # Load the pretrained checkpoint and its tokenizer. We don't need
    # optimizer state — this script only touches the embedding matrix.
    model = SLMForCausalLM.from_pretrained(str(ckpt))
    tok = PreTrainedTokenizerFast.from_pretrained(str(ckpt / "tokenizer"))

    # Tokens that exist in the vocab but were absent from the pretraining
    # corpus, so their embedding rows are effectively random. EOS is omitted
    # because it *was* seen (used as document separator).
    specials = ["<BOS>", "<|system|>", "<|user|>", "<|assistant|>", "<|endofturn|>"]
    ids = [tok.convert_tokens_to_ids(s) for s in specials]
    print("IDs:", dict(zip(specials, ids)))

    # Sanity check: convert_tokens_to_ids returns None or unk_token_id for
    # unknown tokens. Either case means the tokenizer doesn't actually have
    # this special — fail loudly rather than silently overwrite a normal
    # vocab row with the mean.
    assert all(i is not None and i != tok.unk_token_id for i in ids), \
        "Special token lookup failed — tokenizer doesn't contain one of these strings"

    # Compute the mean of all *other* embeddings, i.e. tokens that did get
    # trained during pretraining. This is the "average token" in embedding
    # space and serves as a neutral prior for the unseen specials.
    emb = model.get_input_embeddings().weight.data
    mask = torch.ones(emb.size(0), dtype=torch.bool)
    mask[ids] = False  # exclude the specials themselves from the mean
    mean_emb = emb[mask].mean(dim=0)

    # Overwrite each unseen special with the mean. lm_head.weight is tied
    # to embed_tokens.weight (see SLMForCausalLM.tie_weights), so this also
    # updates the output projection automatically — no separate edit needed.
    with torch.no_grad():
        for i in ids:
            emb[i] = mean_emb

    # Persist back to the same checkpoint dir. SLMForCausalLM.save_pretrained
    # handles the tied-weight serialization correctly (it temporarily breaks
    # the tie, saves, then restores).
    model.save_pretrained(str(ckpt))
    print(f"Re-initialized {len(ids)} special token embeddings in {ckpt}")


if __name__ == "__main__":
    main()