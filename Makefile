"""
pretrain/data/tokenize_data.py
--------------------------
Tokenize the validated JSONL dataset into a memory-mapped binary file
for efficient pretraining.

Tokenizes once, saves to disk as a flat array of uint16 token IDs.
During training, the dataset is loaded with np.memmap — zero-copy,
constant memory regardless of dataset size, and much faster than
tokenizing on the fly.

Format:
    Single flat binary file of uint16 token IDs.
    Documents are concatenated with EOS token as separator.
    No padding — sequences are packed end-to-end.

    [doc1_tok1, doc1_tok2, ..., doc1_tokN, EOS,
     doc2_tok1, doc2_tok2, ..., doc2_tokM, EOS, ...]

    uint16 supports vocab sizes up to 65,535 — sufficient for 32k vocab.

Output:
    data/tokenized/train.bin    — token IDs as uint16
    data/tokenized/train.json   — metadata (n_tokens, n_docs, vocab_size)

Usage:
    python pretrain/data/tokenize_data.py
    python pretrain/data/tokenize_data.py --input data/validated/train.jsonl
    python pretrain/data/tokenize_data.py --workers 8
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TOKENIZED_DIR = DATA_DIR / "tokenized"


def tokenize_document(args: tuple) -> list[int]:
    """
    Tokenize a single document. Designed to run in a worker process.

    Args:
        args: (text, tokenizer_path, eos_id)

    Returns:
        List of token IDs with EOS appended.
    """
    text, tokenizer_path, eos_id = args
    # Import inside worker to avoid serialization issues
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encoded = tokenizer.encode(text)
    return encoded.ids + [eos_id]


def tokenize_dataset(
    input_path: Path,
    output_dir: Path,
    tokenizer_path: Path,
    eos_id: int = 3,
    workers: int = 4,
    split: str = "train",
    shard_size: int = 100_000,
) -> dict:
    """
    Tokenize a JSONL dataset to a memory-mapped binary file.

    Uses multiprocessing to parallelize tokenization across documents.
    Writes in shards to avoid holding all tokens in memory.

    Args:
        input_path: Path to validated JSONL file.
        output_dir: Directory to write output files.
        tokenizer_path: Path to slm_tokenizer.json.
        eos_id: EOS token ID used as document separator. Default: 3.
        workers: Number of parallel tokenization workers.
        split: Dataset split name (used in output filenames).
        shard_size: Number of documents to tokenize per batch.

    Returns:
        Metadata dict with n_tokens, n_docs, vocab_size.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{split}.bin"
    meta_path = output_dir / f"{split}.json"

    if bin_path.exists() and meta_path.exists():
        log.info(f"Already tokenized: {bin_path}")
        with open(meta_path) as f:
            return json.load(f)

    log.info(f"Tokenizing {input_path} → {bin_path}")
    log.info(f"Workers: {workers}, Shard size: {shard_size:,}")

    # Count documents for progress bar
    log.info("Counting documents...")
    n_docs = sum(1 for _ in open(input_path))
    log.info(f"Total documents: {n_docs:,}")

    # Write tokens in shards using a memory-mapped array
    # We don't know total tokens upfront — write to a temp list then save
    all_tokens = []
    n_processed = 0

    tokenizer_path_str = str(tokenizer_path)

    with open(input_path) as f:
        shard = []
        for line in tqdm(f, total=n_docs, desc="Reading", unit="doc"):
            record = json.loads(line)
            text = record.get("text", "").strip()
            if text:
                shard.append((text, tokenizer_path_str, eos_id))

            if len(shard) >= shard_size:
                tokens = _process_shard(shard, workers)
                all_tokens.extend(tokens)
                n_processed += len(shard)
                shard = []

        # Process remaining
        if shard:
            tokens = _process_shard(shard, workers)
            all_tokens.extend(tokens)
            n_processed += len(shard)

    n_tokens = len(all_tokens)
    log.info(f"Total tokens: {n_tokens:,} ({n_tokens / 1e9:.2f}B)")
    log.info(f"Total documents: {n_processed:,}")
    log.info(f"Avg tokens per document: {n_tokens // max(n_processed, 1):,}")

    # Save as uint16 memory-mapped array
    log.info(f"Writing binary file: {bin_path}")
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(str(bin_path))

    # Save metadata
    meta = {
        "n_tokens": n_tokens,
        "n_docs": n_processed,
        "eos_id": eos_id,
        "dtype": "uint16",
        "split": split,
        "input": str(input_path),
        "tokenizer": str(tokenizer_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Metadata saved: {meta_path}")
    log.info(f"Binary size: {bin_path.stat().st_size / 1e9:.2f} GB")

    return meta


def _process_shard(shard: list[tuple], workers: int) -> list[int]:
    """Tokenize a shard of documents using multiprocessing."""
    if workers <= 1:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(shard[0][1])
        tokens = []
        for text, _, eos_id in shard:
            encoded = tokenizer.encode(text)
            tokens.extend(encoded.ids + [eos_id])
        return tokens

    with mp.Pool(workers) as pool:
        results = pool.map(tokenize_document, shard)

    tokens = []
    for result in results:
        tokens.extend(result)
    return tokens


def verify_dataset(bin_path: Path, meta_path: Path) -> None:
    """Quick sanity check on the tokenized dataset."""
    with open(meta_path) as f:
        meta = json.load(f)

    arr = np.memmap(str(bin_path), dtype=np.uint16, mode="r")

    log.info("=== Dataset Verification ===")
    log.info(f"  File:         {bin_path}")
    log.info(f"  Shape:        {arr.shape}")
    log.info(f"  N tokens:     {len(arr):,} (expected {meta['n_tokens']:,})")
    log.info(f"  Min token ID: {arr.min()}")
    log.info(f"  Max token ID: {arr.max()}")
    log.info(f"  EOS count:    {(arr == meta['eos_id']).sum():,} (≈ n_docs)")
    log.info(f"  First 20 IDs: {arr[:20].tolist()}")

    assert len(arr) == meta["n_tokens"], "Token count mismatch"
    log.info("  ✓ Verification passed")


def main():
    parser = argparse.ArgumentParser(description="Tokenize dataset for pretraining")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "validated" / "train.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TOKENIZED_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DATA_DIR / "tokenizer" / "slm_tokenizer.json",
        help="Tokenizer JSON file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 2),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name (train/val)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output after tokenization",
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error(f"Input not found: {args.input}")
        log.error("Run: python validation/scripts/validate.py")
        sys.exit(1)

    if not args.tokenizer.exists():
        log.error(f"Tokenizer not found: {args.tokenizer}")
        log.error("Run: python tokenizer/train_tokenizer.py")
        sys.exit(1)

    log.info(f"Input:     {args.input}")
    log.info(f"Output:    {args.output}")
    log.info(f"Tokenizer: {args.tokenizer}")
    log.info(f"Workers:   {args.workers}")

    meta = tokenize_dataset(
        input_path=args.input,
        output_dir=args.output,
        tokenizer_path=args.tokenizer,
        workers=args.workers,
        split=args.split,
    )

    if args.verify:
        verify_dataset(
            bin_path=args.output / f"{args.split}.bin",
            meta_path=args.output / f"{args.split}.json",
        )

    log.info("Tokenization complete.")
    log.info(f"Next step: python pretrain/train.py")


if __name__ == "__main__":
    main()