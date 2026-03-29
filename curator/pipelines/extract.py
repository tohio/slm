"""
Stage 1: Extraction
-------------------
Reads raw WARC files from Common Crawl and extracts clean text.

Handles:
  - WARC parsing
  - HTML boilerplate removal (trafilatura)
  - Encoding normalization
  - Basic structural cleaning

Output: JSONL files with fields:
  {
    "id": "<warc_record_id>",
    "text": "<cleaned text>",
    "source": "common_crawl",
    "url": "<source url>",
    "warc_file": "<filename>"
  }

Parallelism:
  WARC records are batched and distributed across all Dask workers,
  not whole files. This ensures all n_workers are utilized regardless
  of how many WARC files are present.
  With 2 WARCs and 32 workers: ~32 batches, all workers active.
  With 20 WARCs and 32 workers: ~320 batches, all workers active.
"""

import gzip
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

import trafilatura
from trafilatura.settings import use_config
from warcio.archiveiterator import ArchiveIterator
import dask.bag as db
from dask.distributed import Client, LocalCluster

logger = logging.getLogger("curator.extract")

# Trafilatura config
TRAFILATURA_CONFIG = use_config()
TRAFILATURA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")

# Target partitions per worker — controls granularity of parallelism.
# Higher = more even load balancing but more scheduler overhead.
# 4 is a good default: each worker gets ~4 partitions in flight,
# so fast workers pick up new work while slow ones finish.
PARTITIONS_PER_WORKER = 4


def get_warc_files(input_dir: str) -> list[Path]:
    """Find all WARC and WARC.gz files in the input directory."""
    input_path = Path(input_dir)
    warc_files = (
        list(input_path.glob("**/*.warc.gz"))
        + list(input_path.glob("**/*.warc"))
    )
    logger.info(f"Found {len(warc_files)} WARC files in {input_dir}")
    return sorted(warc_files)


def extract_text_from_html(html_content: bytes, url: str = "") -> Optional[str]:
    """
    Extract main content from HTML using trafilatura.
    Removes boilerplate: navbars, footers, ads, sidebars.
    Returns None if extraction fails or content is too short.
    """
    try:
        text = trafilatura.extract(
            html_content,
            config=TRAFILATURA_CONFIG,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
            url=url,
        )
        return text
    except Exception as e:
        logger.debug(f"Trafilatura extraction failed for {url}: {e}")
        return None


def normalize_encoding(text: str) -> str:
    """Fix common encoding artifacts from web text."""
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def generate_doc_id(url: str, warc_file: str) -> str:
    """Generate a stable document ID from URL + warc file."""
    return hashlib.md5(f"{url}:{warc_file}".encode()).hexdigest()


def iter_warc_records(warc_path: Path) -> Iterator[dict]:
    """
    Iterate over a WARC file yielding raw HTML records as dicts.
    Keeps only response records with HTML content.
    This runs on the main process — lightweight, no extraction yet.
    """
    open_fn = gzip.open if str(warc_path).endswith(".gz") else open
    try:
        with open_fn(warc_path, "rb") as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type != "response":
                    continue
                content_type = record.http_headers.get_header("Content-Type", "")
                if "text/html" not in content_type:
                    continue
                url = record.rec_headers.get_header("WARC-Target-URI", "")
                try:
                    html_content = record.content_stream().read()
                except Exception:
                    continue
                if html_content:
                    yield {
                        "html": html_content,
                        "url": url,
                        "warc_file": warc_path.name,
                    }
    except Exception as e:
        logger.warning(f"Failed to iterate WARC file {warc_path}: {e}")


def process_record(record: dict) -> Optional[dict]:
    """
    Process a single WARC record: extract + normalize text.
    This is the unit of work distributed across Dask workers.
    Returns None if extraction fails or text is too short.
    """
    url = record["url"]
    warc_file = record["warc_file"]
    html_content = record["html"]

    text = extract_text_from_html(html_content, url=url)
    if text is None:
        return None

    text = normalize_encoding(text)
    if len(text) < 100:
        return None

    return {
        "id": generate_doc_id(url, warc_file),
        "text": text,
        "source": "common_crawl",
        "url": url,
        "warc_file": warc_file,
        "char_count": len(text),
        "word_count": len(text.split()),
    }


def write_jsonl(documents: list[dict], output_file: Path):
    """Write documents to a JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def run_extraction(input_dir: str, output_path: Path, cfg: dict):
    """
    Main extraction entry point.

    Processes WARCs one at a time to avoid loading all raw HTML into memory
    at once. Each WARC's records are distributed across all Dask workers,
    results written to disk, then memory freed before the next WARC.

    With 20 WARCs (~693k records, ~11GB raw HTML), loading everything at
    once causes the Dask scheduler to OOM during task graph submission.
    Processing one WARC at a time keeps peak memory under 1GB per batch.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    warc_files = get_warc_files(input_dir)

    if not warc_files:
        raise FileNotFoundError(f"No WARC files found in {input_dir}")

    dask_cfg = cfg.get("dask", {})
    n_workers = dask_cfg.get("n_workers", 4)
    memory_limit = dask_cfg.get("memory_limit", "2GB")

    logger.info(f"Starting extraction: {len(warc_files)} WARC files, {n_workers} workers")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=memory_limit,
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard: {client.dashboard_link}")

    total_docs = 0
    total_records = 0

    try:
        # Process one WARC at a time — keeps peak memory bounded
        for warc_path in warc_files:
            warc_records = list(iter_warc_records(warc_path))
            n_records = len(warc_records)
            total_records += n_records
            logger.info(f"  {warc_path.name}: {n_records} HTML records")

            if not warc_records:
                continue

            # Partitions: each worker gets PARTITIONS_PER_WORKER tasks
            n_partitions = min(n_workers * PARTITIONS_PER_WORKER, n_records)

            bag = db.from_sequence(warc_records, npartitions=n_partitions)
            results = (
                bag
                .map(process_record)
                .filter(lambda x: x is not None)
                .compute()
            )

            # Write output for this WARC
            if results:
                stem = warc_path.name.replace(".warc.gz", "").replace(".warc", "")
                output_file = output_path / f"{stem}.jsonl"
                write_jsonl(results, output_file)
                total_docs += len(results)
                retention = len(results) / n_records * 100
                logger.info(
                    f"    → {len(results)} docs written "
                    f"({retention:.1f}% retained) → {output_file.name}"
                )

            # Explicitly free memory before next WARC
            del warc_records, results

        logger.info(
            f"Extraction complete: {total_docs} documents from "
            f"{len(warc_files)} WARC files "
            f"({total_docs / total_records * 100:.1f}% of records kept)"
        )

    finally:
        client.close()
        cluster.close()