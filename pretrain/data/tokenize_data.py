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
    Documents are bracketed by BOS at start and EOS at end.
    No padding — sequences are packed end-to-end.

    [BOS, doc1_tok1, doc1_tok2, ..., doc1_tokN, EOS,
     BOS, doc2_tok1, doc2_tok2, ..., doc2_tokM, EOS, ...]

    Training slices fixed windows from this continuous stream. It does not
    reset position IDs or construct a block-diagonal mask at EOS, so a later
    document may attend to earlier documents that share its training window.

    uint16 supports vocab sizes up to 65,535 — sufficient for 32k vocab.

Output:
    data/runs/<size>/tokenized/train.bin    — token IDs as uint16
    data/runs/<size>/tokenized/train.json   — metadata (n_tokens, n_docs, dtype, vocab_size)
    data/runs/<size>/tokenized/val.bin      — same, for validation split
    data/runs/<size>/tokenized/val.json
    data/runs/<size>/tokenized/token_mixture.json
                                              — configured vs realized token shares

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
    - Tokens are streamed directly to disk in deterministic input order via
      pool.imap — peak RAM per split is O(chunk_size × avg_tokens),
      not O(shard_size) or O(corpus_size).
    - One persistent mp.Pool for both splits — no pool startup cost per split

Usage:
    python pretrain/data/tokenize_data.py
    python pretrain/data/tokenize_data.py --size 125m
python pretrain/data/tokenize_data.py --train data/runs/125m/validated/train.jsonl \\
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

from config.paths import validated_dir, tokenizer_dir, tokenized_dir, BASE_DATA_DIR
from curator.state import (
    atomic_write_json,
    code_fingerprint,
    manifest_outputs_match,
    stable_digest,
    write_manifest,
)
from pretrain.data.mixture import build_realized_mixture_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = BASE_DATA_DIR
TOKENIZED_DIR = tokenized_dir("125m")

# uint16 supports up to this vocab size (exclusive). If our tokenizer ever
# exceeds this we must switch the binary format to uint32 — silently
# overflowing uint16 would corrupt every token ID above 65535.
UINT16_MAX_VOCAB = 65_536

# Bump this whenever the binary token stream format changes.
# Used to prevent silently reusing stale tokenized files.
TOKENIZED_FORMAT_VERSION = "bos_doc_eos_literal_safe_v3"

# Global tokenizer instance — loaded once per worker process via initializer,
# not once per document. Avoids one tokenizer load per document in the
# original code.
_worker_tokenizer = None
_worker_bos_id    = None
_worker_eos_id    = None


def _configure_pretraining_tokenizer(tokenizer) -> None:
    """Encode reserved-token strings as text, not as structural token IDs."""
    # In Hugging Face tokenizers, this counterintuitive flag means special
    # token strings found in input are passed through the normal model instead
    # of being recognized as AddedToken IDs. The pipeline still inserts the
    # real BOS/EOS IDs explicitly around each encoded document below.
    tokenizer.encode_special_tokens = True

    # Fail at worker startup if a future tokenizers release changes this
    # behavior. Every registered special token must be representable as
    # ordinary text without emitting its reserved ID.
    for token_id, added_token in tokenizer.get_added_tokens_decoder().items():
        if not added_token.special:
            continue
        probe = tokenizer.encode(
            added_token.content,
            add_special_tokens=False,
        )
        if token_id in probe.ids:
            raise RuntimeError(
                f"Tokenizer encoded literal reserved token "
                f"{added_token.content!r} as structural ID {token_id}"
            )


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


def _validate_tokenizer(tokenizer_path: Path) -> tuple[int, int, int]:
    """
    Load and validate the tokenizer in one pass.

    Returns (vocab_size, bos_id, eos_id).

    Raises:
        RuntimeError: if vocab is too large for uint16, or if BOS/EOS
            special tokens are missing from the tokenizer.

    Combines what was previously two separate tokenizer loads
    (vocab check + special-token lookup) into one. Negligible
    perf benefit, but cleaner — both reads happen against the
    same in-memory tokenizer instance.
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

    bos_id = tok.token_to_id("<BOS>")
    eos_id = tok.token_to_id("<EOS>")
    if bos_id is None:
        raise RuntimeError("Tokenizer does not contain required <BOS> token")
    if eos_id is None:
        raise RuntimeError("Tokenizer does not contain required <EOS> token")

    log.info(f"Tokenizer vocab size: {vocab_size:,} (fits in uint16)")
    return vocab_size, bos_id, eos_id


def _worker_init(tokenizer_path: str, bos_id: int, eos_id: int) -> None:
    """
    Initialize the tokenizer once per worker process.

    Called by mp.Pool as the initializer — runs once when each worker
    process starts, not once per task. The tokenizer is stored as a
    module-level global so all tasks in that worker reuse it.

    Uses the raw tokenizers.Tokenizer (not PreTrainedTokenizerFast) —
    the chat_template and tokenizer_config.json are not needed here.
    """
    global _worker_tokenizer, _worker_bos_id, _worker_eos_id
    from tokenizers import Tokenizer
    _worker_tokenizer = Tokenizer.from_file(tokenizer_path)
    _configure_pretraining_tokenizer(_worker_tokenizer)
    _worker_bos_id    = bos_id
    _worker_eos_id    = eos_id


def _tokenize_chunk(
    documents: list[tuple[str, str]],
) -> tuple[list[int], int, dict[str, dict[str, int]]]:
    """
    Tokenize a chunk of documents.

    Returns:
        (tokens, n_docs, per-source document/token counts)

    n_docs is counted from the input chunk length, not by scanning for EOS,
    because document text may itself contain strings that tokenize to EOS.
    """
    global _worker_tokenizer, _worker_bos_id, _worker_eos_id

    tokens: list[int] = []
    texts = [text for text, _ in documents]
    encodings = _worker_tokenizer.encode_batch(texts, add_special_tokens=False)
    source_counts: dict[str, dict[str, int]] = {}

    for enc, (_, source) in zip(encodings, documents):
        tokens.append(_worker_bos_id)
        tokens.extend(enc.ids)
        tokens.append(_worker_eos_id)
        counts = source_counts.setdefault(
            source or "unknown",
            {"documents": 0, "tokens": 0},
        )
        counts["documents"] += 1
        counts["tokens"] += len(enc.ids) + 2

    return tokens, len(texts), source_counts


def _input_fingerprint(path: Path) -> tuple[int, str]:
    """
    Count documents and SHA-256 the exact validated input bytes in one pass.

    This is a separate pass over the input so tqdm can show an ETA during
    the main tokenization loop. For a multi-hour run at 1b scale the extra
    ~30 seconds of counting is a good trade for actionable progress
    reporting. For small runs the overhead is negligible.
    """
    digest = hashlib.sha256()
    docs = 0
    with open(path, "rb", buffering=8 * 1024 * 1024) as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL input: {path}") from exc
            if str(record.get("text", "")).strip():
                docs += 1
    return docs, digest.hexdigest()


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
    bos_id: int,
    eos_id: int,
    tokenizer_path: Path,
    chunk_size: int,
) -> dict:
    """
    Tokenize one JSONL file to {output_dir}/{split}.bin + .json.

    Streams through the input file, batching documents into chunks and
    dispatching via ordered pool.imap. This makes identical inputs,
    configuration, and tokenizer produce byte-identical binaries.

    Returns the metadata dict written to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path  = output_dir / f"{split}.bin"
    meta_path = output_dir / f"{split}.json"

    current_tokenizer_sha256 = tokenizer_fingerprint(tokenizer_path)
    current_implementation_sha256 = code_fingerprint(_tokenize_split)
    log.info(f"[{split}] Fingerprinting/counting {input_path}...")
    n_docs_total, input_sha256 = _input_fingerprint(input_path)
    log.info(f"[{split}] Total documents: {n_docs_total:,}")

    if bin_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        saved_n_tokens = meta.get("n_tokens")
        saved_n_docs = meta.get("n_docs")
        expected_bytes = (
            saved_n_tokens * np.dtype(np.uint16).itemsize
            if isinstance(saved_n_tokens, int)
            else None
        )
        if (
            meta.get("dtype") != "uint16"
            or saved_n_docs != n_docs_total
            or expected_bytes is None
            or bin_path.stat().st_size != expected_bytes
        ):
            log.warning(
                f"[{split}] Existing binary/metadata are incomplete or "
                "inconsistent; rebuilding this derived split"
            )
            bin_path.unlink()
            meta_path.unlink()
            return _tokenize_split(
                input_path, output_dir, split, pool, bos_id, eos_id,
                tokenizer_path, chunk_size,
            )

        saved_tokenizer_sha256 = meta.get("tokenizer_sha256")
        saved_format_version = meta.get("format_version")
        saved_input_sha256 = meta.get("input_sha256")
        saved_implementation_sha256 = meta.get("implementation_sha256")

        if saved_tokenizer_sha256 != current_tokenizer_sha256:
            log.warning(
                f"[{split}] Existing tokenized data was created with a different "
                f"or unknown tokenizer; rebuilding this derived split"
            )
            bin_path.unlink()
            meta_path.unlink()
            return _tokenize_split(
                input_path, output_dir, split, pool, bos_id, eos_id,
                tokenizer_path, chunk_size,
            )

        if saved_format_version != TOKENIZED_FORMAT_VERSION:
            log.warning(
                f"[{split}] Existing tokenized data uses an old or unknown format.\n"
                f"Existing format_version: {saved_format_version}\n"
                f"Current format_version:  {TOKENIZED_FORMAT_VERSION}; rebuilding"
            )
            bin_path.unlink()
            meta_path.unlink()
            return _tokenize_split(
                input_path, output_dir, split, pool, bos_id, eos_id,
                tokenizer_path, chunk_size,
            )

        if saved_input_sha256 != input_sha256:
            log.warning(
                f"[{split}] Existing tokenized data was created from a "
                f"different or unknown validated input; rebuilding"
            )
            bin_path.unlink()
            meta_path.unlink()
            return _tokenize_split(
                input_path, output_dir, split, pool, bos_id, eos_id,
                tokenizer_path, chunk_size,
            )

        if saved_implementation_sha256 != current_implementation_sha256:
            log.warning(
                f"[{split}] Tokenization implementation changed; rebuilding"
            )
            bin_path.unlink()
            meta_path.unlink()
            return _tokenize_split(
                input_path, output_dir, split, pool, bos_id, eos_id,
                tokenizer_path, chunk_size,
            )

        log.info(
            f"[{split}] Already tokenized and input/tokenizer/format match: {bin_path}"
        )
        return meta

    n_tokens    = 0
    n_processed = 0
    source_counts: dict[str, dict[str, int]] = {}
    tmp_bin_path = bin_path.with_name(f".{bin_path.name}.{os.getpid()}.tmp")

    with open(tmp_bin_path, "wb") as bin_file, open(input_path) as f:
        # Read and chunk documents lazily — no full-corpus list in RAM.
        def _doc_iter():
            for line in f:
                record = json.loads(line)
                text = record.get("text", "").strip()
                if text:
                    yield text, str(record.get("source", "unknown"))

        chunks = _chunked(_doc_iter(), chunk_size)

        # Ordered imap preserves input order while still tokenizing chunks in
        # parallel. The bounded prefetch queue keeps memory bounded.
        pbar = tqdm(total=n_docs_total, desc=f"Tokenizing {split}", unit="doc")
        for tokens, docs_in_chunk, chunk_counts in pool.imap(
            _tokenize_chunk,
            chunks,
        ):
            _write_tokens(tokens, bin_file)
            n_tokens += len(tokens)
            n_processed += docs_in_chunk
            for source, counts in chunk_counts.items():
                aggregate = source_counts.setdefault(
                    source,
                    {"documents": 0, "tokens": 0},
                )
                aggregate["documents"] += counts["documents"]
                aggregate["tokens"] += counts["tokens"]
            pbar.update(docs_in_chunk)
        pbar.close()
        bin_file.flush()
        os.fsync(bin_file.fileno())
    tmp_bin_path.replace(bin_path)

    log.info(f"[{split}] Total tokens:     {n_tokens:,} ({n_tokens / 1e9:.2f}B)")
    log.info(f"[{split}] Total documents:  {n_processed:,}")
    log.info(f"[{split}] Avg tokens/doc:   {n_tokens // max(n_processed, 1):,}")
    log.info(f"[{split}] Binary size:      {bin_path.stat().st_size / 1e9:.2f} GB")

    meta = {
        "n_tokens":  n_tokens,
        "n_docs":    n_processed,
        "bos_id":    bos_id,
        "eos_id":    eos_id,
        "dtype":     "uint16",
        "split":     split,
        "input":     str(input_path),
        "input_sha256": input_sha256,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": current_tokenizer_sha256,
        "format_version": TOKENIZED_FORMAT_VERSION,
        "implementation_sha256": current_implementation_sha256,
        "source_counts": source_counts,
    }

    atomic_write_json(meta_path, meta)
    log.info(f"[{split}] Metadata saved:   {meta_path}")

    return meta


def _write_tokens(tokens: list[int], bin_file) -> None:
    """Write a flat list of token IDs to the binary file as uint16."""
    arr = np.array(tokens, dtype=np.uint16)
    bin_file.write(arr.tobytes())


def verify_dataset(bin_path: Path, meta_path: Path) -> None:
    """Quick sanity check on the tokenized dataset."""
    with open(meta_path) as f:
        meta = json.load(f)

    if "bos_id" not in meta:
        raise RuntimeError(f"{meta_path} missing bos_id metadata")

    arr = np.memmap(str(bin_path), dtype=np.uint16, mode="r")

    log.info("=== Dataset Verification ===")
    log.info(f"  File:         {bin_path}")
    log.info(f"  Format ver:   {meta.get('format_version', '<unknown>')}")
    log.info(f"  Shape:        {arr.shape}")
    log.info(f"  N tokens:     {len(arr):,} (expected {meta['n_tokens']:,})")
    log.info(f"  Min token ID: {arr.min()}")
    log.info(f"  Max token ID: {arr.max()}")
    
    bos_count = int((arr == meta["bos_id"]).sum())
    eos_count = int((arr == meta["eos_id"]).sum())
    
    log.info(f"  BOS count:    {bos_count:,} (expected n_docs={meta['n_docs']:,})")
    log.info(f"  EOS count:    {eos_count:,} (expected n_docs={meta['n_docs']:,})")
    log.info(f"  First 20 IDs: {arr[:20].tolist()}")

    assert len(arr) == meta["n_tokens"], "Token count mismatch"

    if bos_count != meta["n_docs"]:
        raise RuntimeError(
            f"BOS count mismatch: BOS={bos_count:,}, n_docs={meta['n_docs']:,}"
        )

    if eos_count != meta["n_docs"]:
        raise RuntimeError(
            f"EOS count mismatch: EOS={eos_count:,}, n_docs={meta['n_docs']:,}"
        )

    # Every supported BOS/document/EOS binary layout must start with BOS.
    # Catches the case where BOS prepending is broken in a way that
    # still produces the right BOS count but wrong layout.
    if int(arr[0]) != meta["bos_id"]:
        raise RuntimeError(
            f"First token is {int(arr[0])}, expected BOS={meta['bos_id']}. "
            f"Binary layout is wrong — format_version={meta.get('format_version')!r} "
            f"requires BOS at position 0."
        )

    log.info("  ✓ Verification passed")


def main():
    parser = argparse.ArgumentParser(description="Tokenize dataset for pretraining")
    parser.add_argument("--size", default=os.environ.get("SIZE", "125m"), help="Run size")
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Input train JSONL file",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=None,
        help="Input val JSONL file (skipped if missing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
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

    run_validated_dir = validated_dir(args.size)
    args.train = args.train or (run_validated_dir / "train.jsonl")
    args.val = args.val or (run_validated_dir / "val.jsonl")
    args.output = args.output or tokenized_dir(args.size)
    args.tokenizer = args.tokenizer or (tokenizer_dir(args.size) / "slm_tokenizer.json")

    # Pre-flight checks — fail with clear messages before spawning the pool
    if not args.train.exists():
        log.error(f"Train input not found: {args.train}")
        log.error("Run: make validate SIZE=<size>")
        sys.exit(1)
    if (
        args.train.parent == run_validated_dir
        and args.val.parent == run_validated_dir
        and not manifest_outputs_match(
            run_validated_dir,
            output_pattern="*.json*",
        )
    ):
        raise RuntimeError(
            f"Validated tokenization inputs are not manifest-complete: "
            f"{run_validated_dir}"
        )

    if not args.tokenizer.exists():
        log.error(f"Tokenizer not found: {args.tokenizer}")
        log.error("Run: make tokenizer SIZE=<size>")
        log.error(
            "Or restore it with: make artifacts-download SIZE=<size> "
            "RUN_ID=<run-id> ARTIFACT_STAGES=tokenizer"
        )
        sys.exit(1)

    # Single-pass tokenizer validation: vocab size + special token IDs.
    # Fails fast if the tokenizer is unusable, before spawning workers.
    _, bos_id, eos_id = _validate_tokenizer(args.tokenizer)
    log.info(f"Special token IDs: BOS={bos_id}, EOS={eos_id}")

    if not args.val.exists():
        raise FileNotFoundError(
            f"Val input not found: {args.val}. Tokenization requires both "
            f"validated splits."
        )
    val_available = True

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
        initargs=(tokenizer_path_str, bos_id, eos_id),
    ) as pool:
        train_meta = _tokenize_split(
            input_path=args.train,
            output_dir=args.output,
            split="train",
            pool=pool,
            bos_id=bos_id,
            eos_id=eos_id,
            tokenizer_path=args.tokenizer,
            chunk_size=args.chunk_size,
        )

        if val_available:
            val_meta = _tokenize_split(
                input_path=args.val,
                output_dir=args.output,
                split="val",
                pool=pool,
                bos_id=bos_id,
                eos_id=eos_id,
                tokenizer_path=args.tokenizer,
                chunk_size=args.chunk_size,
            )

    mixture_report = build_realized_mixture_report(train_meta, val_meta)
    mixture_report_path = args.output / "token_mixture.json"
    atomic_write_json(mixture_report_path, mixture_report)
    log.info("Realized token mixture report saved: %s", mixture_report_path)

    tokenization_contract = {
        "implementation_sha256": code_fingerprint(_tokenize_split),
        "format_version": TOKENIZED_FORMAT_VERSION,
        "dtype": "uint16",
        "tokenizer_sha256": train_meta["tokenizer_sha256"],
        "inputs": {
            "train": train_meta["input_sha256"],
            "val": val_meta["input_sha256"],
        },
        "mixture_contract_sha256": mixture_report["contract_sha256"],
    }
    write_manifest(
        args.output,
        stage="tokenize",
        contract=tokenization_contract,
        input_signature=stable_digest(tokenization_contract["inputs"]),
        output_pattern="[tv]*",
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
    log.info('Next step: make artifacts-upload SIZE=<size> ARTIFACT_STAGES="raw,curated,validated,tokenized,tokenizer,metadata"')


if __name__ == "__main__":
    main()
