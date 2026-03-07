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
import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster

logger = logging.getLogger("curator.extract")


# --- Trafilatura config ---
TRAFILATURA_CONFIG = use_config()
TRAFILATURA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")


def get_warc_files(input_dir: str) -> list[Path]:
    """Find all WARC and WARC.gz files in the input directory."""
    input_path = Path(input_dir)
    warc_files = list(input_path.glob("**/*.warc.gz")) + list(input_path.glob("**/*.warc"))
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
            favor_precision=True,   # prefer quality over recall
            url=url,
        )
        return text
    except Exception as e:
        logger.debug(f"Trafilatura extraction failed for {url}: {e}")
        return None


def normalize_encoding(text: str) -> str:
    """
    Fix common encoding artifacts from web text.
    Handles mojibake, unusual unicode, excess whitespace.
    """
    # Normalize unicode to NFC form
    import unicodedata
    text = unicodedata.normalize("NFC", text)

    # Replace null bytes and other control chars (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize multiple newlines → max 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def generate_doc_id(url: str, warc_file: str) -> str:
    """Generate a stable document ID from URL + warc file."""
    content = f"{url}:{warc_file}"
    return hashlib.md5(content.encode()).hexdigest()


def process_warc_file(warc_path: Path) -> list[dict]:
    """
    Process a single WARC file, extracting text from all HTML records.
    Returns a list of document dicts.
    """
    documents = []
    open_fn = gzip.open if str(warc_path).endswith(".gz") else open

    try:
        with open_fn(warc_path, "rb") as stream:
            for record in ArchiveIterator(stream):
                # Only process response records with HTML content
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

                text = extract_text_from_html(html_content, url=url)
                if text is None:
                    continue

                text = normalize_encoding(text)

                # Skip very short extractions early
                if len(text) < 100:
                    continue

                doc = {
                    "id": generate_doc_id(url, str(warc_path.name)),
                    "text": text,
                    "source": "common_crawl",
                    "url": url,
                    "warc_file": warc_path.name,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }
                documents.append(doc)

    except Exception as e:
        logger.warning(f"Failed to process WARC file {warc_path}: {e}")

    logger.debug(f"Extracted {len(documents)} documents from {warc_path.name}")
    return documents


def write_jsonl(documents: list[dict], output_file: Path):
    """Write documents to a JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def run_extraction(input_dir: str, output_path: Path, cfg: dict):
    """
    Main extraction entry point.
    Processes all WARC files in parallel using Dask.
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

    try:
        # Process WARC files in parallel
        bag = db.from_sequence(warc_files, npartitions=len(warc_files))
        results = bag.map(process_warc_file).compute()

        # Write output — one JSONL per WARC file
        total_docs = 0
        for warc_path, documents in zip(warc_files, results):
            if not documents:
                continue
            output_file = output_path / f"{warc_path.stem}.jsonl"
            write_jsonl(documents, output_file)
            total_docs += len(documents)
            logger.debug(f"Written {len(documents)} docs → {output_file.name}")

        logger.info(f"Extraction complete: {total_docs} documents extracted from {len(warc_files)} WARC files")

    finally:
        client.close()
        cluster.close()
