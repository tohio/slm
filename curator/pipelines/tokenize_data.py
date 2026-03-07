"""
Stage 8: Tokenization
----------------------
Converts curated JSONL documents to NeMo's memory-mapped binary format
(.bin + .idx file pair) for efficient streaming during pre-training.

Memory-mapped datasets avoid loading the entire corpus into RAM —
critical when training on billions of tokens.

Uses the custom SentencePiece BPE tokenizer trained on our data.

Output structure:
  /data/curated/tokenized/
    general_text.bin      ← raw token IDs (uint16 or uint32)
    general_text.idx      ← index of document offsets
    code_text.bin
    code_text.idx
"""

import json
import logging
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
import sentencepiece as spm

logger = logging.getLogger("curator.tokenize")

# NeMo uses uint16 for vocab sizes <= 65535, uint32 otherwise
# With vocab_size=32000, uint16 is sufficient
DTYPE = np.uint16
HDR_MAGIC = b"MMIDIDX\x00\x00"   # NeMo mmap index magic bytes
VERSION = 1


class MMapIndexedDatasetBuilder:
    """
    Builds NeMo-compatible memory-mapped binary datasets.
    Writes two files:
      .bin  — concatenated token ID arrays
      .idx  — document lengths and byte offsets for random access
    """

    def __init__(self, output_prefix: str, dtype=DTYPE):
        self.output_prefix = output_prefix
        self.dtype = dtype
        self._bin_file = open(f"{output_prefix}.bin", "wb")
        self._sizes: list[int] = []           # token count per document
        self._byte_offsets: list[int] = [0]   # byte offset of each document
        self._doc_count = 0

    def add_document(self, token_ids: list[int], eos_id: int = 3):
        """Add a single document's tokens to the dataset."""
        # Append EOS token
        tokens = token_ids + [eos_id]
        arr = np.array(tokens, dtype=self.dtype)
        self._bin_file.write(arr.tobytes())
        self._sizes.append(len(tokens))
        self._byte_offsets.append(
            self._byte_offsets[-1] + arr.nbytes
        )
        self._doc_count += 1

    def finalize(self):
        """Write the .idx file and close both files."""
        self._bin_file.close()

        with open(f"{self.output_prefix}.idx", "wb") as idx_file:
            # Header
            idx_file.write(HDR_MAGIC)
            idx_file.write(struct.pack("<Q", VERSION))
            # Dtype code (1=uint8, 2=int8, 3=int16, 4=int32, 5=int64, 6=float32, 7=float64, 8=uint16)
            dtype_code = 8  # uint16
            idx_file.write(struct.pack("<B", dtype_code))
            # Document count
            idx_file.write(struct.pack("<Q", self._doc_count))
            # Sizes array
            sizes_arr = np.array(self._sizes, dtype=np.int32)
            idx_file.write(sizes_arr.tobytes())
            # Pointer (byte offset) array
            # Use int64 for offsets to handle large datasets
            ptrs_arr = np.array(self._byte_offsets[:-1], dtype=np.int64)
            idx_file.write(ptrs_arr.tobytes())

        logger.info(
            f"Written {self._doc_count:,} documents to {self.output_prefix}.{{bin,idx}}"
        )

        total_tokens = sum(self._sizes)
        logger.info(f"  Total tokens: {total_tokens:,} ({total_tokens/1e9:.3f}B)")


def iter_documents(input_path: Path) -> Iterator[dict]:
    """Iterate over all documents in a directory of JSONL files."""
    for jsonl_file in sorted(input_path.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def run_tokenization(input_path: Path, output_path: Path, cfg: dict):
    """
    Main tokenization entry point.
    Reads all curated JSONL, tokenizes, writes mmap dataset.
    """
    if not cfg.get("enabled", True):
        logger.info("Tokenization disabled")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer_model = cfg["tokenizer_model"]
    output_format = cfg.get("output_format", "mmap")

    if output_format != "mmap":
        raise ValueError(f"Unsupported output format: {output_format}. Only 'mmap' is supported.")

    # Load tokenizer
    logger.info(f"Loading SentencePiece model from {tokenizer_model}")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model)

    vocab_size = sp.GetPieceSize()
    logger.info(f"Tokenizer loaded. Vocab size: {vocab_size:,}")

    # Use uint32 if vocab > 65535
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    # Output prefix — NeMo expects the prefix, not the full filename
    output_prefix = str(output_path / "text_document")
    builder = MMapIndexedDatasetBuilder(output_prefix, dtype=dtype)

    total_docs = 0
    total_tokens = 0
    skipped = 0

    for doc in iter_documents(input_path):
        text = doc.get("text", "").strip()
        if not text:
            skipped += 1
            continue

        token_ids = sp.EncodeAsIds(text)

        # Skip very short tokenized documents
        if len(token_ids) < 10:
            skipped += 1
            continue

        builder.add_document(token_ids, eos_id=sp.eos_id())
        total_docs += 1
        total_tokens += len(token_ids)

        if total_docs % 100000 == 0:
            logger.info(
                f"  Tokenized {total_docs:,} documents, "
                f"{total_tokens:,} tokens ({total_tokens/1e9:.3f}B)"
            )

    builder.finalize()

    logger.info(f"Tokenization complete:")
    logger.info(f"  Documents:    {total_docs:,}")
    logger.info(f"  Tokens:       {total_tokens:,} ({total_tokens/1e9:.3f}B)")
    logger.info(f"  Skipped:      {skipped:,}")
    logger.info(f"  Output:       {output_prefix}.{{bin,idx}}")
