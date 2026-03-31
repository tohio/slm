"""
curator/filters/dedup.py
-------------------------
MinHash-based fuzzy deduplication for pretraining data.

Exact deduplication (hash of full text) catches verbatim duplicates.
Fuzzy deduplication (MinHash LSH) catches near-duplicates — documents
that are slightly different but substantially the same content. Both
are needed for web crawl data which contains many scraped copies of
the same article with minor variations.

MinHash works by:
    1. Shingling — split text into overlapping n-grams (shingles)
    2. MinHashing — compute a compact signature (128 values) that
       approximates Jaccard similarity between documents
    3. LSH (Locality Sensitive Hashing) — bucket signatures so that
       similar documents end up in the same bucket
    4. Candidate pairs — documents in the same bucket are compared
       and near-duplicates are removed

Threshold: Jaccard similarity > 0.8 → considered duplicate.
This is the threshold used by FineWeb and most production pipelines.

References:
    datasketch: https://github.com/ekzhu/datasketch
    FineWeb dedup: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
"""

import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Iterator

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

log = logging.getLogger(__name__)

# Default MinHash parameters
NUM_PERM = 128          # number of hash permutations — higher = more accurate
SHINGLE_SIZE = 5        # n-gram size for shingling
JACCARD_THRESHOLD = 0.8 # similarity threshold for near-duplicate detection


def normalize(text: str) -> str:
    """
    Normalize text before shingling.

    Lowercases, removes punctuation and extra whitespace.
    Normalization ensures minor formatting differences don't
    prevent detection of semantically identical documents.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_shingles(text: str, n: int = SHINGLE_SIZE) -> set[bytes]:
    """
    Extract character n-gram shingles from text.

    Args:
        text: Normalized input text.
        n: Shingle size (character n-grams). Default: 5.

    Returns:
        Set of n-gram byte strings.
    """
    text = normalize(text)
    if len(text) < n:
        return {text.encode("utf-8")}
    return {text[i : i + n].encode("utf-8") for i in range(len(text) - n + 1)}


def compute_minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    """
    Compute a MinHash signature for a document.

    Args:
        text: Document text.
        num_perm: Number of hash permutations.

    Returns:
        MinHash object representing the document's signature.
    """
    m = MinHash(num_perm=num_perm)
    for shingle in get_shingles(text):
        m.update(shingle)
    return m


def exact_hash(text: str) -> str:
    """Compute SHA-256 hash of text for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Deduplicator:
    """
    Two-stage deduplicator: exact hash + MinHash LSH fuzzy dedup.

    Stage 1 — Exact dedup: SHA-256 hash of normalized text.
              Catches verbatim duplicates with zero false positives.

    Stage 2 — Fuzzy dedup: MinHash LSH with Jaccard threshold.
              Catches near-duplicates (same article, minor edits).

    The LSH index is built incrementally — documents are added one
    by one and checked for duplicates on insertion.

    Args:
        threshold: Jaccard similarity threshold for fuzzy dedup. Default: 0.8.
        num_perm: Number of MinHash permutations. Default: 128.
        index_path: Optional path to save/load the LSH index for resumability.

    Example::

        dedup = Deduplicator(threshold=0.8)
        records = [{"text": "..."}, ...]

        kept = []
        for record in records:
            if dedup.is_duplicate(record["text"]):
                continue
            dedup.add(record["text"])
            kept.append(record)

        print(dedup.report())
    """

    def __init__(
        self,
        threshold: float = JACCARD_THRESHOLD,
        num_perm: int = NUM_PERM,
        index_path: Path | None = None,
    ):
        self.threshold = threshold
        self.num_perm = num_perm
        self.index_path = index_path

        self.exact_hashes: set[str] = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._doc_count = 0

        self.stats = {
            "total": 0,
            "kept": 0,
            "exact_duplicates": 0,
            "fuzzy_duplicates": 0,
        }

        if index_path and Path(index_path).exists():
            self.load(index_path)
            log.info(f"Loaded dedup index from {index_path} ({len(self.exact_hashes):,} documents)")

    def is_duplicate(self, text: str) -> bool:
        """
        Check if a document is a duplicate without adding it to the index.

        Args:
            text: Document text.

        Returns:
            True if the document is a duplicate.
        """
        # Stage 1: exact hash
        h = exact_hash(normalize(text))
        if h in self.exact_hashes:
            return True

        # Stage 2: fuzzy MinHash
        m = compute_minhash(text, self.num_perm)
        result = self.lsh.query(m)
        return len(result) > 0

    def add(self, text: str) -> str:
        """
        Add a document to the deduplication index.

        Args:
            text: Document text.

        Returns:
            Document key (used as LSH index key).
        """
        key = f"doc_{self._doc_count}"
        self._doc_count += 1

        h = exact_hash(normalize(text))
        self.exact_hashes.add(h)

        m = compute_minhash(text, self.num_perm)
        try:
            self.lsh.insert(key, m)
        except ValueError:
            pass  # already inserted

        return key

    def check_and_add(self, text: str) -> bool:
        """
        Check if duplicate and add if not. Returns True if kept.

        Convenience method that combines is_duplicate() and add().
        """
        self.stats["total"] += 1

        # Stage 1: exact
        h = exact_hash(normalize(text))
        if h in self.exact_hashes:
            self.stats["exact_duplicates"] += 1
            return False

        # Stage 2: fuzzy
        m = compute_minhash(text, self.num_perm)
        if self.lsh.query(m):
            self.stats["fuzzy_duplicates"] += 1
            return False

        # Not a duplicate — add to index
        key = f"doc_{self._doc_count}"
        self._doc_count += 1
        self.exact_hashes.add(h)
        try:
            self.lsh.insert(key, m)
        except ValueError:
            pass

        self.stats["kept"] += 1
        return True

    def deduplicate_records(
        self,
        records: list[dict],
        text_key: str = "text",
    ) -> list[dict]:
        """
        Deduplicate a list of records in memory.

        Args:
            records: List of dicts with a text field.
            text_key: Key for the text field. Default: "text".

        Returns:
            Deduplicated list of records.
        """
        kept = []
        for record in tqdm(records, desc="Deduplicating", unit="doc"):
            text = record.get(text_key, "")
            if self.check_and_add(text):
                kept.append(record)
        return kept

    def deduplicate_jsonl(
        self,
        input_path: Path,
        output_path: Path,
        text_key: str = "text",
    ) -> dict:
        """
        Deduplicate a JSONL file, writing results to output_path.

        Streams the input — memory usage is bounded by the LSH index,
        not the size of the file.

        Args:
            input_path: Input JSONL file.
            output_path: Output JSONL file (deduplicated).
            text_key: Key for the text field.

        Returns:
            Stats dict.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0

        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in tqdm(fin, desc=f"Dedup {input_path.name}", unit="doc"):
                record = json.loads(line)
                text = record.get(text_key, "")
                if self.check_and_add(text):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

        return {
            "input": str(input_path),
            "output": str(output_path),
            "written": written,
            **self.stats,
        }

    def save(self, path: Path) -> None:
        """Save the dedup index to disk for resumability."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "exact_hashes": self.exact_hashes,
                "lsh": self.lsh,
                "doc_count": self._doc_count,
                "stats": self.stats,
            }, f)
        log.info(f"Saved dedup index to {path} ({len(self.exact_hashes):,} docs)")

    def load(self, path: Path) -> None:
        """Load a previously saved dedup index."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.exact_hashes = data["exact_hashes"]
        self.lsh = data["lsh"]
        self._doc_count = data["doc_count"]
        self.stats = data.get("stats", self.stats)

    def report(self) -> str:
        """Return a human-readable deduplication report."""
        total = self.stats["total"]
        kept = self.stats["kept"]
        exact = self.stats["exact_duplicates"]
        fuzzy = self.stats["fuzzy_duplicates"]
        return (
            f"Deduplication report:\n"
            f"  Total processed:    {total:>10,}\n"
            f"  Kept:               {kept:>10,}  ({100 * kept / max(total, 1):.1f}%)\n"
            f"  Exact duplicates:   {exact:>10,}  ({100 * exact / max(total, 1):.1f}%)\n"
            f"  Fuzzy duplicates:   {fuzzy:>10,}  ({100 * fuzzy / max(total, 1):.1f}%)\n"
            f"  Index size:         {len(self.exact_hashes):>10,} documents"
        )