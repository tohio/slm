"""
pretrain/data/tokenize_data.py
--------------------------
Tokenize the validated JSONL datasets into memory-mapped binary files
for efficient pretraining.

Tokenizes once, saves to disk as a flat array of uint16 token IDs.
During training, each dataset is loaded with np.memmap — zero-copy,
constant memory regardless of dataset size, and much faster than
tokenizing on the fly.

Format:
    Single flat binary file of uint16 token IDs per split.
    Documents are concatenated with EOS token as separator.
    No padding — sequences are packed end-to-end.

    [doc1_tok1, doc1_tok2, ..., doc1_tokN, EOS,
     doc2_tok1, doc2_tok2, ..., doc2_tokM, EOS, ...]

    uint16 supports vocab sizes up to 65,535 — sufficient for 32k vocab.

Output:
    data/tokenized/train.bin    — token IDs as uint16
    data/tokenized/train.json   — metadata (n_tokens, n_docs, dtype, vocab_size)
    data/tokenized/val.bin      — same, for validation split
    data/tokenized/val.json

Inputs:
    By default both validated/train.jsonl and validated/val.jsonl are tokenized.
    The val split was produced upstream by the curator's blend stage as a
    uniform random sample of the shuffled documents, so val and train come
    from the same distribution.

Tokenizer:
    Uses the raw tokenizers.Tokenizer from slm_tokenizer.json directly —
    not PreTrainedTokenizerFast. This is intentional: bulk tokenization
    only needs text → token IDs conversion. The raw tokenizer is faster
    and has no dependency on tokenizer_config.json or the chat_template,
    which are only needed at training and inference time.

Performance notes:
    - Tokenizer is loaded once per worker process (not per document)
    - Documents are batched into chunks before dispatch to amortise IPC overhead
    - Tokens are streamed directly to disk as chunks complete via
      pool.imap_unordered — peak RAM per split is O(chunk_size × avg_tokens),
      not O(shard_size) or O(corpus_size).
    - One persistent mp.Pool for both splits — no pool startup cost per split

Usage:
    python pretrain/data/tokenize_data.py
    python pretrain/data/tokenize_data.py --train data/validated/train.jsonl \\
                                          --val   data/validated/val.jsonl
    python pretrain/data/tokenize_data.py --workers 24
"""

import argparse
import hashlib
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

DATA_DIR      = Path(os.environ.get("DATA_DIR", "data"))
TOKENIZED_DIR = DATA_DIR / "tokenized"

# uint16 supports up to this vocab size (exclusive). If our tokenizer ever
# exceeds this we must switch the binary format to uint32 — silently
# overflowing uint16 would corrupt every token ID above 65535.
UINT16_MAX_VOCAB = 65_536

# Global tokenizer instance — loaded once per worker process via initializer,
# not once per document. Avoids one tokenizer load per document in the
# original code.
_worker_tokenizer = None
_worker_eos_id    = None


def tokenizer_fingerprint(tokenizer_path: Path) -> str:
    """
    Return SHA256 of the tokenizer's canonical serialized form.

    Hashes the loaded tokenizer's behavior, not the on-disk bytes.
    Two files that load to the same tokenizer (e.g., re-saved with
    different whitespace) produce the same fingerprint; two files
    that produce different IDs for the same input cannot collide.
    """
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    return hashlib.sha256(tok.to_str().encode("utf-8")).hexdigest()


def _worker_init(tokenizer_path: str, eos_id: int) -> None:
    """
    Initialize the tokenizer once per worker process.

    Called by mp.Pool as the initializer — runs once when each worker
    process starts, not once per task. The tokenizer is stored as a
    module-level global so all tasks in that worker reuse it.

    Uses the raw tokenizers.Tokenizer (not PreTrainedTokenizerFast) —
    the chat_template and tokenizer_config.json are not needed here.
    """
    global _worker_tokenizer, _worker_eos_id
    from tokenizers import Tokenizer
    _worker_tokenizer = Tokenizer.from_file(tokenizer_path)
    _worker_eos_id    = eos_id


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
    tokens: list[int] = []
    # encode_batch encodes all texts in one call — faster than a loop
    encodings = _worker_tokenizer.encode_batch(texts)
    for enc in encodings:
        tokens.extend(enc.ids)
        tokens.append(_worker_eos_id)
    return tokens


def _count_docs(path: Path) -> int:
    """
    Count documents (non-empty lines) in a JSONL file.

    This is a separate pass over the input so tqdm can show an ETA during
    the main tokenization loop. For a multi-hour run at 1b scale the extra
    ~30 seconds of counting is a good trade for actionable progress
    reporting. For small runs the overhead is negligible.
    """
    return sum(1 for line in open(path) if line.strip())


def _chunked(iterable, size: int):
    """Yield successive size-sized chunks from iterable."""
    buf: list = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _tokenize_split(
    input_path: Path,
    output_dir: Path,
    split: str,
    pool: mp.Pool,
    eos_id: int,
    tokenizer_path: Path,
    chunk_size: int,
) -> dict:
    """
    Tokenize one JSONL file to {output_dir}/{split}.bin + .json.

    Streams through the input file, batching documents into chunks and
    dispatching via pool.imap_unordered. Results arrive in whatever order
    workers finish — the order of documents in the output binary differs
    from the input, but since documents are separated by EOS and position
    within the binary carries no meaning, this has no effect on training.

    Returns the metadata dict written to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path  = output_dir / f"{split}.bin"
    meta_path = output_dir / f"{split}.json"

    current_tokenizer_sha256 = tokenizer_fingerprint(tokenizer_path)

    if bin_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        saved_tokenizer_sha256 = meta.get("tokenizer_sha256")

        if saved_tokenizer_sha256 != current_tokenizer_sha256:
            raise RuntimeError(
                f"[{split}] Existing tokenized data was created with a different "
                f"or unknown tokenizer.\n"
                f"Existing tokenizer_sha256: {saved_tokenizer_sha256}\n"
                f"Current tokenizer_sha256:  {current_tokenizer_sha256}\n"
                f"Delete {bin_path} and {meta_path}, or run a clean tokenize target."
            )

        log.info(f"[{split}] Already tokenized and tokenizer hash matches: {bin_path}")
        return meta

    log.info(f"[{split}] Counting documents in {input_path}...")
    n_docs_total = _count_docs(input_path)
    log.info(f"[{split}] Total documents: {n_docs_total:,}")

    n_tokens    = 0
    n_processed = 0

    with open(bin_path, "wb") as bin_file, open(input_path) as f:
        # Read and chunk documents lazily — no full-corpus list in RAM.
        def _doc_iter():
            for line in f:
                record = json.loads(line)
                text = record.get("text", "").strip()
                if text:
                    yield text

        chunks = _chunked(_doc_iter(), chunk_size)

        # imap_unordered streams results back as each worker finishes a
        # chunk. The token list for each chunk is written immediately and
        # discarded, so peak RAM is bounded by (pool size × chunk size).
        pbar = tqdm(total=n_docs_total, desc=f"Tokenizing {split}", unit="doc")
        for tokens in pool.imap_unordered(_tokenize_chunk, chunks):
            _write_tokens(tokens, bin_file)
            n_tokens    += len(tokens)
            # Can't know exactly how many docs produced this chunk (workers
            # drop empty texts upstream), but we know each chunk holds at
            # most chunk_size. EOS tokens produced = docs produced.
            docs_in_chunk = sum(1 for t in tokens if t == eos_id)
            n_processed  += docs_in_chunk
            pbar.update(docs_in_chunk)
        pbar.close()

    log.info(f"[{split}] Total tokens:     {n_tokens:,} ({n_tokens / 1e9:.2f}B)")
    log.info(f"[{split}] Total documents:  {n_processed:,}")
    log.info(f"[{split}] Avg tokens/doc:   {n_tokens // max(n_processed, 1):,}")
    log.info(f"[{split}] Binary size:      {bin_path.stat().st_size / 1e9:.2f} GB")

    meta = {
        "n_tokens":  n_tokens,
        "n_docs":    n_processed,
        "eos_id":    eos_id,
        "dtype":     "uint16",
        "split":     split,
        "input":     str(input_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": current_tokenizer_sha256,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"[{split}] Metadata saved:   {meta_path}")

    return meta


def _write_tokens(tokens: list[int], bin_file) -> None:
    """Write a flat list of token IDs to the binary file as uint16."""
    arr = np.array(tokens, dtype=np.uint16)
    bin_file.write(arr.tobytes())


def _assert_vocab_fits_uint16(tokenizer_path: Path) -> int:
    """
    Verify the tokenizer's vocab size fits in uint16. Returns vocab size.

    If vocab_size > 65535, uint16 silently overflows and every token ID
    above 65535 gets written as (id mod 65536). Training on the resulting
    garbage would not fail — the model would just see a corrupted vocab
    distribution. This check makes the failure mode loud.
    """
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tok.get_vocab_size()
    if vocab_size >= UINT16_MAX_VOCAB:
        raise RuntimeError(
            f"Tokenizer vocab_size={vocab_size:,} does not fit in uint16 "
            f"(max {UINT16_MAX_VOCAB - 1:,}). Either reduce the vocab size "
            f"or switch the binary format in tokenize_data.py and "
            f"dataset.py to uint32."
        )
    log.info(f"Tokenizer vocab size: {vocab_size:,} (fits in uint16)")
    return vocab_size


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
        "--train",
        type=Path,
        default=DATA_DIR / "validated" / "train.jsonl",
        help="Input train JSONL file",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DATA_DIR / "validated" / "val.jsonl",
        help="Input val JSONL file (skipped if missing)",
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
        help="Path to slm_tokenizer.json (raw BPE tokenizer)",
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
        "--verify",
        action="store_true",
        help="Verify output after tokenization",
    )
    args = parser.parse_args()

    # Pre-flight checks — fail with clear messages before spawning the pool
    if not args.train.exists():
        log.error(f"Train input not found: {args.train}")
        log.error("Run: make validate")
        sys.exit(1)

    if not args.tokenizer.exists():
        log.error(f"Tokenizer not found: {args.tokenizer}")
        log.error("Run: make tokenizer && make tokenizer-upload")
        log.error("Or:  make tokenizer-download")
        sys.exit(1)

    # Verify vocab fits in uint16 BEFORE spawning workers — fail fast
    eos_id = 3
    _assert_vocab_fits_uint16(args.tokenizer)

    val_available = args.val.exists()
    if not val_available:
        log.warning(
            f"Val input not found: {args.val}\n"
            f"Only train will be tokenized. The curator's blend stage produces "
            f"both train.jsonl and val.jsonl — re-run 'make curate' to get val."
        )

    log.info(f"Train:      {args.train}")
    log.info(f"Val:        {args.val if val_available else '(not found, skipping)'}")
    log.info(f"Output:     {args.output}")
    log.info(f"Tokenizer:  {args.tokenizer}")
    log.info(f"Workers:    {args.workers}")
    log.info(f"Chunk size: {args.chunk_size}")

    tokenizer_path_str = str(args.tokenizer)

    # One persistent pool for both splits — avoids pool startup cost twice.
    with mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(tokenizer_path_str, eos_id),
    ) as pool:
        _tokenize_split(
            input_path=args.train,
            output_dir=args.output,
            split="train",
            pool=pool,
            eos_id=eos_id,
            tokenizer_path=args.tokenizer,
            chunk_size=args.chunk_size,
        )

        if val_available:
            _tokenize_split(
                input_path=args.val,
                output_dir=args.output,
                split="val",
                pool=pool,
                eos_id=eos_id,
                tokenizer_path=args.tokenizer,
                chunk_size=args.chunk_size,
            )

    if args.verify:
        verify_dataset(
            bin_path=args.output / "train.bin",
            meta_path=args.output / "train.json",
        )
        if val_available:
            verify_dataset(
                bin_path=args.output / "val.bin",
                meta_path=args.output / "val.json",
            )

    log.info("Tokenization complete.")
    log.info("Next step: make tokenize-upload")


if __name__ == "__main__":
    main()