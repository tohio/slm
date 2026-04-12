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

Performance notes:
    - Tokenizer is loaded once per worker process (not per document)
    - Documents are batched into chunks before dispatch to amortise IPC overhead
    - Tokens are streamed directly to disk in shard-sized chunks — peak RAM
      is O(shard_size) not O(corpus_size)
    - One persistent mp.Pool for the full run — no per-shard pool startup cost

Usage:
    python pretrain/data/tokenize_data.py
    python pretrain/data/tokenize_data.py --input data/validated/train.jsonl
    python pretrain/data/tokenize_data.py --workers 24
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from itertools import chain
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

# Global tokenizer instance — loaded once per worker process via initializer,
# not once per document. Avoids 3.4M tokenizer loads in the original code.
_worker_tokenizer = None
_worker_eos_id = None


def _worker_init(tokenizer_path: str, eos_id: int) -> None:
    """
    Initialize the tokenizer once per worker process.

    Called by mp.Pool as the initializer — runs once when each worker
    process starts, not once per task. The tokenizer is stored as a
    module-level global so all tasks in that worker reuse it.
    """
    global _worker_tokenizer, _worker_eos_id
    from tokenizers import Tokenizer
    _worker_tokenizer = Tokenizer.from_file(tokenizer_path)
    _worker_eos_id = eos_id


def _tokenize_chunk(texts: list[str]) -> list[int]:
    """
    Tokenize a chunk of documents in a single worker task.

    Receives a list of texts, encodes them all using the process-local
    tokenizer (loaded once via _worker_init), and returns a flat list
    of token IDs with EOS appended after each document.

    Batching documents into chunks (rather than one doc per task) amortises
    the multiprocessing IPC overhead across many documents per round-trip.
    """
    global _worker_tokenizer, _worker_eos_id
    tokens = []
    # encode_batch encodes all texts in one call — faster than a loop
    encodings = _worker_tokenizer.encode_batch(texts)
    for enc in encodings:
        tokens.extend(enc.ids)
        tokens.append(_worker_eos_id)
    return tokens


def tokenize_dataset(
    input_path: Path,
    output_dir: Path,
    tokenizer_path: Path,
    eos_id: int = 3,
    workers: int = 4,
    split: str = "train",
    shard_size: int = 100_000,
    chunk_size: int = 256,
) -> dict:
    """
    Tokenize a JSONL dataset to a memory-mapped binary file.

    Uses a persistent multiprocessing pool with one tokenizer per worker
    process. Documents are batched into chunks before dispatch to amortise
    IPC overhead. Tokens are streamed to disk in shard-sized batches so
    peak RAM is O(shard_size), not O(corpus_size).

    Args:
        input_path:    Path to validated JSONL file.
        output_dir:    Directory to write output files.
        tokenizer_path: Path to slm_tokenizer.json.
        eos_id:        EOS token ID used as document separator. Default: 3.
        workers:       Number of parallel tokenization worker processes.
        split:         Dataset split name (used in output filenames).
        shard_size:    Documents per shard — controls how often tokens are
                       flushed to disk. Lower = less RAM, more I/O.
        chunk_size:    Documents per worker task. Higher = less IPC overhead,
                       more latency before first result.

    Returns:
        Metadata dict with n_tokens, n_docs, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{split}.bin"
    meta_path = output_dir / f"{split}.json"

    if bin_path.exists() and meta_path.exists():
        log.info(f"Already tokenized: {bin_path}")
        with open(meta_path) as f:
            return json.load(f)

    log.info(f"Tokenizing {input_path} → {bin_path}")
    log.info(f"Workers: {workers}, Shard size: {shard_size:,}, Chunk size: {chunk_size}")

    # Count documents for progress bar
    log.info("Counting documents...")
    n_docs = sum(1 for _ in open(input_path))
    log.info(f"Total documents: {n_docs:,}")

    tokenizer_path_str = str(tokenizer_path)
    n_tokens = 0
    n_processed = 0

    # Open output file for streaming writes — no large in-memory accumulation
    with open(bin_path, "wb") as bin_file, \
         open(input_path) as f, \
         mp.Pool(
             processes=workers,
             initializer=_worker_init,
             initargs=(tokenizer_path_str, eos_id),
         ) as pool:

        shard_texts: list[str] = []
        pbar = tqdm(total=n_docs, desc="Tokenizing", unit="doc")

        for line in f:
            record = json.loads(line)
            text = record.get("text", "").strip()
            if text:
                shard_texts.append(text)

            if len(shard_texts) >= shard_size:
                tokens = _flush_shard(shard_texts, pool, chunk_size)
                _write_tokens(tokens, bin_file)
                n_tokens += len(tokens)
                n_processed += len(shard_texts)
                pbar.update(len(shard_texts))
                shard_texts = []

        # Flush remainder
        if shard_texts:
            tokens = _flush_shard(shard_texts, pool, chunk_size)
            _write_tokens(tokens, bin_file)
            n_tokens += len(tokens)
            n_processed += len(shard_texts)
            pbar.update(len(shard_texts))

        pbar.close()

    log.info(f"Total tokens:     {n_tokens:,} ({n_tokens / 1e9:.2f}B)")
    log.info(f"Total documents:  {n_processed:,}")
    log.info(f"Avg tokens/doc:   {n_tokens // max(n_processed, 1):,}")
    log.info(f"Binary size:      {bin_path.stat().st_size / 1e9:.2f} GB")

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
    log.info(f"Metadata saved:   {meta_path}")

    return meta


def _flush_shard(texts: list[str], pool: mp.Pool, chunk_size: int) -> list[int]:
    """
    Tokenize a shard of documents using the worker pool.

    Splits texts into chunks of chunk_size and dispatches to workers
    via pool.map. Returns a flat list of all token IDs.
    """
    chunks = [
        texts[i : i + chunk_size]
        for i in range(0, len(texts), chunk_size)
    ]
    results = pool.map(_tokenize_chunk, chunks)
    # chain.from_iterable avoids repeated list concatenation
    return list(chain.from_iterable(results))


def _write_tokens(tokens: list[int], bin_file) -> None:
    """Write a flat list of token IDs to the binary file as uint16."""
    arr = np.array(tokens, dtype=np.uint16)
    bin_file.write(arr.tobytes())


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
        help="Number of parallel workers. Default: cpu_count - 2",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Documents per worker task. Higher = less IPC overhead. Default: 256",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Documents per disk flush. Lower = less RAM. Default: 100000",
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

    log.info(f"Input:      {args.input}")
    log.info(f"Output:     {args.output}")
    log.info(f"Tokenizer:  {args.tokenizer}")
    log.info(f"Workers:    {args.workers}")
    log.info(f"Chunk size: {args.chunk_size}")
    log.info(f"Shard size: {args.shard_size:,}")

    meta = tokenize_dataset(
        input_path=args.input,
        output_dir=args.output,
        tokenizer_path=args.tokenizer,
        workers=args.workers,
        split=args.split,
        chunk_size=args.chunk_size,
        shard_size=args.shard_size,
    )

    if args.verify:
        verify_dataset(
            bin_path=args.output / f"{args.split}.bin",
            meta_path=args.output / f"{args.split}.json",
        )

    log.info("Tokenization complete.")
    log.info("Next step: python pretrain/train.py")


if __name__ == "__main__":
    main()