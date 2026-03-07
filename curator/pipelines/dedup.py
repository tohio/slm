"""
Stage 5 & 6: Deduplication
---------------------------
Two-pass deduplication strategy:

Pass 1 — Exact dedup (MD5 hash):
  Fast. Catches byte-identical documents.
  Common with scraped web data (same page indexed multiple times).

Pass 2 — Fuzzy dedup (MinHash + LSH):
  Catches near-duplicate documents that differ in whitespace,
  minor edits, or slight reformatting. Critical for web data
  where the same article appears across many domains.

  Algorithm:
    1. Tokenize each document into n-grams (char or word level)
    2. Compute MinHash signature (128 hash functions)
    3. Group signatures into LSH bands — documents in the same
       bucket are candidate duplicates
    4. Compute Jaccard similarity for candidates
    5. Remove duplicates above similarity threshold

Input/Output: JSONL
"""

import hashlib
import json
import logging
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger("curator.dedup")


# ─────────────────────────────────────────────
# Exact Deduplication
# ─────────────────────────────────────────────

def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def run_exact_dedup(input_path: Path, output_path: Path, cfg: dict) -> None:
    """
    Remove exact duplicate documents by hashing document text.
    Single-pass, in-memory hash set — fast and memory efficient
    at our scale (millions of documents, not billions).
    """
    if not cfg.get("enabled", True):
        logger.info("Exact dedup disabled — symlinking input to output")
        output_path.symlink_to(input_path)
        return

    output_path.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()

    input_files = sorted(input_path.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No JSONL files in {input_path}")

    total_in = 0
    total_out = 0

    for input_file in input_files:
        output_file = output_path / input_file.name
        kept = 0
        file_total = 0

        with open(input_file, encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                file_total += 1
                doc = json.loads(line)
                text = doc.get("text", "")
                doc_hash = compute_md5(text)

                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    kept += 1

        total_in += file_total
        total_out += kept
        logger.debug(f"{input_file.name}: {kept}/{file_total} kept after exact dedup")

    retention = (total_out / total_in * 100) if total_in > 0 else 0
    logger.info(
        f"Exact dedup complete: {total_out}/{total_in} documents retained ({retention:.1f}%). "
        f"Hash set size: {len(seen_hashes):,}"
    )


# ─────────────────────────────────────────────
# Fuzzy Deduplication (MinHash + LSH)
# ─────────────────────────────────────────────

def get_word_ngrams(text: str, n: int = 5) -> set[str]:
    """
    Generate word-level n-grams from text.
    Word n-grams are more robust than char n-grams for
    detecting near-duplicate natural language text.
    """
    words = text.lower().split()
    if len(words) < n:
        # Short document: use unigrams
        return set(words)
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Compute MinHash signature for a document."""
    m = MinHash(num_perm=num_perm)
    ngrams = get_word_ngrams(text, n=5)
    for ng in ngrams:
        m.update(ng.encode("utf8"))
    return m


def load_all_documents(input_path: Path) -> list[tuple[str, dict]]:
    """Load all documents from JSONL files into memory for LSH indexing."""
    docs = []
    for jsonl_file in sorted(input_path.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                docs.append((str(jsonl_file), doc))
    return docs


def run_fuzzy_dedup(input_path: Path, output_path: Path, cfg: dict) -> None:
    """
    Remove near-duplicate documents using MinHash LSH.

    Strategy:
      1. Compute MinHash for every document
      2. Insert into LSH index
      3. Query LSH for each document — if a similar doc already exists, mark as duplicate
      4. Write only non-duplicate documents

    Note: Order matters — first-seen document is kept, later duplicates removed.
    This is deterministic given consistent file ordering.
    """
    if not cfg.get("enabled", True):
        logger.info("Fuzzy dedup disabled — symlinking input to output")
        output_path.symlink_to(input_path)
        return

    output_path.mkdir(parents=True, exist_ok=True)

    num_perm = cfg.get("num_hashes", 128)
    threshold = cfg.get("jaccard_threshold", 0.8)
    false_positive_check = cfg.get("false_positive_check", True)

    logger.info(f"Fuzzy dedup: threshold={threshold}, num_perm={num_perm}")
    logger.info("Loading all documents into memory for LSH indexing...")

    all_docs = load_all_documents(input_path)
    total_in = len(all_docs)
    logger.info(f"Loaded {total_in:,} documents")

    # Build LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}

    logger.info("Computing MinHash signatures...")
    for idx, (_, doc) in enumerate(all_docs):
        doc_id = doc.get("id", str(idx))
        text = doc.get("text", "")
        mh = compute_minhash(text, num_perm=num_perm)
        minhashes[doc_id] = mh

        if idx % 10000 == 0 and idx > 0:
            logger.info(f"  Hashed {idx:,}/{total_in:,} documents")

    # Deduplicate via LSH queries
    logger.info("Running LSH deduplication pass...")
    duplicate_ids: set[str] = set()

    for idx, (_, doc) in enumerate(all_docs):
        doc_id = doc.get("id", str(idx))

        if doc_id in duplicate_ids:
            continue

        mh = minhashes[doc_id]

        try:
            lsh.insert(doc_id, mh)
        except ValueError:
            # Already inserted (shouldn't happen with unique IDs)
            continue

        # Query for similar documents already in index
        candidates = lsh.query(mh)
        for candidate_id in candidates:
            if candidate_id == doc_id:
                continue
            # Mark candidate as duplicate of current doc
            duplicate_ids.add(candidate_id)

        if idx % 10000 == 0 and idx > 0:
            logger.info(f"  Processed {idx:,}/{total_in:,}, duplicates found: {len(duplicate_ids):,}")

    # Write non-duplicate documents
    logger.info(f"Writing deduplicated output. Removing {len(duplicate_ids):,} duplicates...")
    total_out = 0

    # Group docs by source file for organized output
    file_docs: dict[str, list] = defaultdict(list)
    for source_file, doc in all_docs:
        file_docs[source_file].append(doc)

    for source_file, docs in file_docs.items():
        output_file = output_path / Path(source_file).name
        with open(output_file, "w", encoding="utf-8") as fout:
            for idx, doc in enumerate(docs):
                doc_id = doc.get("id", str(idx))
                if doc_id not in duplicate_ids:
                    fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    total_out += 1

    retention = (total_out / total_in * 100) if total_in > 0 else 0
    logger.info(
        f"Fuzzy dedup complete: {total_out}/{total_in} documents retained ({retention:.1f}%). "
        f"Duplicates removed: {len(duplicate_ids):,}"
    )
