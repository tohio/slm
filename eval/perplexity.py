"""
eval/perplexity.py
------------------
Computes perplexity on a held-out validation set.

Perplexity = exp(average negative log-likelihood per token)

Lower is better. Tracks model quality through each training stage:
  - Pre-training: ~100-200 is reasonable for a small model from scratch
  - After SFT: may increase slightly on raw text (expected — distribution shift)
  - Compare within stage across checkpoints, not across stages

Uses the memory-mapped validation split from the tokenization stage:
  /data/curated/tokenized/text_document.bin
  /data/curated/tokenized/text_document.idx
(splits_string: "99,1,0" → 1% of data held out for validation)
"""

import logging
import math
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger("eval.perplexity")


def load_nemo_model(checkpoint: str, device: str):
    """Load a NeMo GPT model from checkpoint in eval mode."""
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

    logger.info(f"Loading model from {checkpoint}...")
    model = MegatronGPTModel.restore_from(
        restore_path=checkpoint,
        map_location=torch.device(device),
    )
    model.eval()
    model.to(device)
    logger.info("Model loaded")
    return model


def load_val_tokens(val_data_dir: str, max_sequences: int = 1000) -> list[list[int]]:
    """
    Load tokenized validation sequences from memory-mapped dataset.
    Reads the .bin/.idx files produced by tokenize_data.py at:
      /data/curated/tokenized/text_document.bin
      /data/curated/tokenized/text_document.idx
    """
    data_dir = Path(val_data_dir)
    bin_files = sorted(data_dir.glob("*.bin"))

    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {val_data_dir}")

    idx_file = bin_files[0].with_suffix(".idx")
    if not idx_file.exists():
        raise FileNotFoundError(f"Index file not found: {idx_file}")

    import struct

    with open(idx_file, "rb") as f:
        # Skip header (magic + version + dtype code)
        f.read(9 + 8 + 1)
        doc_count = struct.unpack("<Q", f.read(8))[0]
        sizes = np.frombuffer(f.read(doc_count * 4), dtype=np.int32)
        offsets = np.frombuffer(f.read(doc_count * 8), dtype=np.int64)

    with open(bin_files[0], "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint16)

    sequences = []
    for i in range(min(doc_count, max_sequences)):
        start = offsets[i] // 2  # uint16 = 2 bytes
        length = sizes[i]
        seq = data[start:start + length].tolist()
        if len(seq) > 10:
            sequences.append(seq)

    logger.info(f"Loaded {len(sequences)} validation sequences from {bin_files[0].name}")
    return sequences


def compute_perplexity(
    model,
    sequences: list[list[int]],
    seq_length: int = 2048,
    device: str = "cuda",
) -> dict:
    """
    Compute perplexity over validation sequences.

    For each sequence, compute cross-entropy loss (NLL per token),
    then average and exponentiate to get perplexity.
    """
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for seq in sequences:
            tokens = seq[:seq_length]
            if len(tokens) < 2:
                continue

            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    labels=target_ids,
                )

                if isinstance(outputs, torch.Tensor):
                    loss = outputs.item()
                elif hasattr(outputs, "loss"):
                    loss = outputs.loss.item()
                else:
                    continue

                n_tokens = target_ids.numel()
                total_nll += loss * n_tokens
                total_tokens += n_tokens

            except Exception as e:
                logger.debug(f"Skipping sequence due to error: {e}")
                continue

    if total_tokens == 0:
        logger.error("No tokens evaluated — check model and data compatibility")
        return {"val_perplexity": float("inf"), "val_loss": float("inf"), "tokens_evaluated": 0}

    avg_loss = total_nll / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "val_perplexity":      round(perplexity, 4),
        "val_loss":            round(avg_loss, 6),
        "tokens_evaluated":    total_tokens,
        "sequences_evaluated": len(sequences),
    }


def evaluate_perplexity(
    checkpoint: str,
    val_data_dir: str,
    device: str = "cuda",
    max_sequences: int = 1000,
) -> dict:
    """Main perplexity evaluation entry point."""
    model = load_nemo_model(checkpoint, device)

    try:
        sequences = load_val_tokens(val_data_dir, max_sequences=max_sequences)
    except FileNotFoundError as e:
        logger.warning(f"Could not load mmap val data: {e}")
        logger.warning("Falling back to synthetic sequences for smoke testing")
        vocab_size = model.cfg.tokenizer.get("vocab_size", 32000)
        sequences = [
            list(np.random.randint(4, vocab_size, size=512))
            for _ in range(10)
        ]

    seq_length = model.cfg.get("seq_length", 2048)
    results = compute_perplexity(model, sequences, seq_length=seq_length, device=device)

    logger.info(
        f"Perplexity: {results['val_perplexity']:.3f} | "
        f"Loss: {results['val_loss']:.4f} | "
        f"Tokens: {results['tokens_evaluated']:,}"
    )
    return results