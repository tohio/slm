"""
pretrain/data/dataset.py
-------------------------
Memory-mapped dataset for pretraining.

Wraps the tokenized .bin file with a PyTorch Dataset interface.
Uses np.memmap for zero-copy, constant-memory access to the token array
regardless of dataset size.

Train / val split:
    The train / val split is produced by the curator (blend stage), not at
    runtime. The blend stage writes:
        data/curated/train.jsonl   — ~99.5% of shuffled documents
        data/curated/val.jsonl     — ~0.5% of shuffled documents (same distribution)

    Tokenization (pretrain/data/tokenize_data.py) then produces:
        data/tokenized/train.bin   + train.json metadata
        data/tokenized/val.bin     + val.json metadata

    This module just wraps each .bin independently. No runtime splitting,
    no risk of stale split files drifting out of sync with the underlying
    tokenization, and val is a clean uniform sample rather than the tail
    of the tokenized stream.

Sequence packing:
    Each training example is a fixed-length window of tokens sliced
    from the flat token array. No padding — sequences are packed
    end-to-end with documents separated by EOS tokens.

    Example with seq_len=4 and tokens [1,2,3,4,5,6,7,8,9]:
        Example 0: input=[1,2,3,4], labels=[1,2,3,4]
        Example 1: input=[5,6,7,8], labels=[5,6,7,8]   (with stride=seq_len)

    The stride controls overlap between examples:
        stride = seq_len     → no overlap (standard)
        stride = seq_len // 2 → 50% overlap (more examples, correlated)

    Default: stride = seq_len (no overlap, maximum efficiency).

Labels:
    Labels equal input_ids. The model's forward() performs the next-token
    shift internally — logits[..., :-1, :] predicts labels[..., 1:] — so
    the dataset MUST NOT pre-shift labels. Pre-shifting here causes the
    model to be trained on a 2-token-ahead objective (silent disaster:
    training loss looks fine, generation produces gibberish).
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# uint16 supports token IDs up to 65535. The metadata check catches drift
# if someone regenerates the binary with a different dtype.
_EXPECTED_DTYPE = "uint16"


class PretrainingDataset(Dataset):
    """
    Memory-mapped pretraining dataset.

    Reads token IDs from a flat binary file using np.memmap.
    Returns (input_ids, labels) pairs of fixed length seq_len, where
    labels equal input_ids (the model handles the next-token shift).

    Args:
        bin_path: Path to tokenized .bin file (uint16 token IDs).
        seq_len: Sequence length for training. Must be > 0. Default: 2048.
        stride: Stride between consecutive examples. Must be > 0. Default: seq_len.
        split: Split name for logging. Default: "train".

    Raises:
        FileNotFoundError: if bin_path doesn't exist.
        ValueError: if seq_len <= 0, stride <= 0, or metadata mismatches.

    Example::

        dataset = PretrainingDataset(
            bin_path=Path("data/tokenized/train.bin"),
            seq_len=2048,
        )
        print(f"Examples: {len(dataset):,}")

        item = dataset[0]
        print(item["input_ids"].shape)   # (2048,)
        print(item["labels"].shape)      # (2048,)
    """

    def __init__(
        self,
        bin_path: Path,
        seq_len: int = 2048,
        stride: int | None = None,
        split: str = "train",
    ):
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.split = split

        # ── Argument validation ───────────────────────────────────────────────
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found: {self.bin_path}\n"
                f"Run: python pretrain/data/tokenize_data.py"
            )

        # ── Metadata ──────────────────────────────────────────────────────────
        # .json sidecar is expected but not strictly required — if missing we
        # log a warning and skip the consistency checks. If present we verify
        # dtype and token count match what's actually on disk.
        meta_path = self.bin_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            log.warning(
                f"No metadata file at {meta_path} — skipping dtype / count "
                f"consistency checks. Regenerate with tokenize_data.py to "
                f"get these safety checks."
            )
            self.meta = {}

        # Memory-map the token array — zero copy, no RAM usage
        self.data = np.memmap(str(self.bin_path), dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

        # ── Metadata consistency ──────────────────────────────────────────────
        # Catches the case where someone regenerated the binary with a
        # different dtype (e.g. accidentally int32) — memmap would read half
        # the bytes as uint16 and return garbage token IDs. The error is
        # silent without this check.
        if self.meta:
            meta_dtype = self.meta.get("dtype")
            if meta_dtype and meta_dtype != _EXPECTED_DTYPE:
                raise ValueError(
                    f"{self.bin_path}: metadata says dtype={meta_dtype!r}, "
                    f"but PretrainingDataset reads as {_EXPECTED_DTYPE}. "
                    f"Regenerate the binary or update the dataset reader."
                )
            meta_n_tokens = self.meta.get("n_tokens")
            if meta_n_tokens is not None and meta_n_tokens != self.n_tokens:
                raise ValueError(
                    f"{self.bin_path}: metadata says n_tokens={meta_n_tokens:,} "
                    f"but memmap sees {self.n_tokens:,}. Binary was truncated "
                    f"or replaced without updating metadata."
                )

        # ── Example count ─────────────────────────────────────────────────────
        # Need seq_len tokens per example. Labels equal input_ids; the model
        # performs the next-token shift internally.
        if self.n_tokens < self.seq_len:
            raise ValueError(
                f"{self.bin_path}: only {self.n_tokens:,} tokens but "
                f"seq_len={self.seq_len} needs at least {self.seq_len}. "
                f"Dataset is too small for this seq_len."
            )

        self.n_examples = ((self.n_tokens - self.seq_len) // self.stride) + 1

        log.info(
            f"PretrainingDataset ({split}): "
            f"{self.n_tokens:,} tokens, "
            f"{self.n_examples:,} examples, "
            f"seq_len={seq_len}, stride={self.stride}"
        )

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return a training example.

        Standard Hugging Face causal-LM contract:

            input_ids = [t0, t1, t2, ...]
            labels    = [t0, t1, t2, ...]

        The model shifts internally:

            logits[..., :-1, :] predicts labels[..., 1:]

        Do NOT pre-shift labels here, or the model learns to predict
        token t+2 instead of token t+1.
        """
        start = idx * self.stride
        chunk = self.data[start : start + self.seq_len].astype(np.int64)

        input_ids = torch.from_numpy(chunk)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def estimate_epochs(self, batch_size: int, steps: int) -> float:
        """Estimate how many epochs a given number of steps covers."""
        total_examples = steps * batch_size
        return total_examples / max(self.n_examples, 1)

    def token_budget(self) -> dict:
        """Return token budget statistics."""
        return {
            "n_tokens": self.n_tokens,
            "n_examples": self.n_examples,
            "seq_len": self.seq_len,
            "stride": self.stride,
            "tokens_per_example": self.seq_len,
            "total_training_tokens": self.n_examples * self.seq_len,
            "utilization": self.n_examples * self.seq_len / max(self.n_tokens, 1),
        }


def load_train_val(
    tokenized_dir: Path,
    seq_len: int = 2048,
    stride: int | None = None,
) -> tuple[PretrainingDataset, PretrainingDataset]:
    """
    Load the train + val datasets from the tokenized directory.

    Expects the curator to have produced train.jsonl and val.jsonl, and
    tokenize_data.py to have turned them into train.bin + val.bin with
    matching .json metadata sidecars.

    Args:
        tokenized_dir: directory containing train.bin, val.bin, and their
            .json metadata sidecars.
        seq_len: sequence length for training.
        stride: stride between examples. Default: seq_len.

    Returns:
        (train_dataset, val_dataset)

    Raises:
        FileNotFoundError: if either train.bin or val.bin is missing.
    """
    tokenized_dir = Path(tokenized_dir)
    train_bin = tokenized_dir / "train.bin"
    val_bin = tokenized_dir / "val.bin"

    if not train_bin.exists():
        raise FileNotFoundError(
            f"train.bin not found at {train_bin}\n"
            f"Run: python pretrain/data/tokenize_data.py"
        )
    if not val_bin.exists():
        raise FileNotFoundError(
            f"val.bin not found at {val_bin}\n"
            f"The curator produces both train.jsonl and val.jsonl; "
            f"tokenize_data.py must tokenize both. If you're resuming an "
            f"older run with only train.bin, re-run: "
            f"make curate && make tokenize"
        )

    train = PretrainingDataset(train_bin, seq_len=seq_len, stride=stride, split="train")
    val = PretrainingDataset(val_bin, seq_len=seq_len, stride=stride, split="val")
    return train, val