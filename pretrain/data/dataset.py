"""
pretrain/data/dataset.py
-------------------------
Memory-mapped dataset for pretraining.

Wraps the tokenized .bin file with a PyTorch Dataset interface.
Uses np.memmap for zero-copy, constant-memory access to the token array
regardless of dataset size.

Sequence packing:
    Each training example is a fixed-length window of tokens sliced
    from the flat token array. No padding — sequences are packed
    end-to-end with documents separated by EOS tokens.

    Example with seq_len=4 and tokens [1,2,3,4,5,6,7,8,9]:
        Example 0: input=[1,2,3,4], label=[2,3,4,5]
        Example 1: input=[2,3,4,5], label=[3,4,5,6]
        ...

    The stride controls overlap between examples:
        stride = seq_len     → no overlap (standard)
        stride = seq_len // 2 → 50% overlap (more examples, correlated)

    Default: stride = seq_len (no overlap, maximum efficiency).

Labels:
    Labels are input_ids shifted left by one position — each token
    predicts the next. The loss is computed over all tokens equally
    (standard language modelling objective).
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class PretrainingDataset(Dataset):
    """
    Memory-mapped pretraining dataset.

    Reads token IDs from a flat binary file using np.memmap.
    Returns (input_ids, labels) pairs of fixed length seq_len.

    Args:
        bin_path: Path to tokenized .bin file (uint16 token IDs).
        seq_len: Sequence length for training. Default: 2048.
        stride: Stride between consecutive examples. Default: seq_len.
        split: Split name for logging. Default: "train".

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

        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found: {self.bin_path}\n"
                f"Run: python pretrain/data/tokenize_data.py"
            )

        # Load metadata
        meta_path = self.bin_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        # Memory-map the token array — zero copy, no RAM usage
        self.data = np.memmap(str(self.bin_path), dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

        # Number of examples — need seq_len + 1 tokens per example (for labels shift)
        self.n_examples = (self.n_tokens - self.seq_len) // self.stride

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

        Args:
            idx: Example index.

        Returns:
            Dict with:
                input_ids: (seq_len,) long tensor
                labels:    (seq_len,) long tensor — input_ids shifted left by 1
        """
        start = idx * self.stride
        # +1 to get labels (next token prediction)
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)

        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])

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


class PretrainingDatasetWithValidation:
    """
    Convenience wrapper that creates train and validation splits
    from a single tokenized dataset by reserving the last N tokens.

    Args:
        bin_path: Path to tokenized .bin file.
        seq_len: Sequence length. Default: 2048.
        val_fraction: Fraction of tokens to reserve for validation. Default: 0.005 (0.5%).
        stride: Stride between examples. Default: seq_len.
    """

    def __init__(
        self,
        bin_path: Path,
        seq_len: int = 2048,
        val_fraction: float = 0.005,
        stride: int | None = None,
    ):
        bin_path = Path(bin_path)
        data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        n_tokens = len(data)
        n_val_tokens = max(seq_len * 100, int(n_tokens * val_fraction))

        # Split point
        train_tokens = n_tokens - n_val_tokens
        log.info(
            f"Dataset split: "
            f"{train_tokens:,} train tokens, "
            f"{n_val_tokens:,} val tokens "
            f"({100 * n_val_tokens / n_tokens:.2f}%)"
        )

        # Write split files if they don't exist
        train_path = bin_path.parent / "train_split.bin"
        val_path = bin_path.parent / "val_split.bin"

        if not train_path.exists():
            data[:train_tokens].tofile(str(train_path))
            log.info(f"Wrote train split: {train_path}")

        if not val_path.exists():
            data[train_tokens:].tofile(str(val_path))
            log.info(f"Wrote val split: {val_path}")

        self.train = PretrainingDataset(train_path, seq_len=seq_len, stride=stride, split="train")
        self.val = PretrainingDataset(val_path, seq_len=seq_len, stride=stride, split="val")