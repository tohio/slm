"""
Stage 2: Language Filter
------------------------
Identifies and retains only English documents using fastText's
language identification model (lid.176.bin).

Removes:
  - Non-English documents
  - Documents where the language model is uncertain (low confidence)

Parallelism:
  Documents are distributed across all Dask workers at the record level,
  not the file level. This ensures full worker utilization regardless of
  how many JSONL files are present.

Input/Output: JSONL
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import fasttext
import dask.bag as db
from dask.distributed import Client, LocalCluster

logger = logging.getLogger("curator.language_filter")


def detect_language(record: dict, model_path: str, target_language: str, min_score: float) -> dict | None:
    """
    Detect language of a single document using fastText.
    Loads the model once per worker (cached in worker memory).
    Returns None if the document does not pass the language filter.
    """
    # Worker-local model cache — loaded once, reused for all records on this worker
    import fasttext as ft
    if not hasattr(detect_language, "_model"):
        detect_language._model = ft.load_model(model_path)

    text = record.get("text", "")
    if not text:
        return None

    sample = text[:1000].replace("\n", " ")
    predictions = detect_language._model.predict(sample, k=1)
    lang = predictions[0][0].replace("__label__", "")
    confidence = float(predictions[1][0])

    if lang != target_language or confidence < min_score:
        return None

    record["language"] = lang
    record["language_score"] = round(confidence, 4)
    return record


def run_language_filter(input_path: Path, output_path: Path, cfg: dict):
    """
    Main language filter entry point.
    Distributes all documents across Dask workers at record level.
    """
    if not cfg.get("enabled", True):
        logger.info("Language filter disabled — symlinking input to output")
        output_path.symlink_to(input_path)
        return

    output_path.mkdir(parents=True, exist_ok=True)

    model_path = cfg["model_path"]
    target_language = cfg.get("target_language", "en")
    min_score = cfg.get("min_language_score", 0.65)

    input_files = sorted(input_path.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {input_path}")

    logger.info(
        f"Language filtering {len(input_files)} files "
        f"(target={target_language}, min_score={min_score})"
    )

    # Load all records into memory — they are already filtered/small after extract
    all_records = []
    source_map = {}  # doc id → source filename for output grouping
    for input_file in input_files:
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                all_records.append(doc)
                source_map[doc["id"]] = input_file.name

    total_in = len(all_records)
    logger.info(f"Loaded {total_in:,} documents for language filtering")

    # Dask config from parent pipeline config
    dask_cfg = cfg.get("dask", {})
    n_workers = dask_cfg.get("n_workers", 4)
    memory_limit = dask_cfg.get("memory_limit", "2GB")

    # Dynamic partitioning — same strategy as extract.py
    partitions_per_worker = 4
    n_partitions = min(n_workers * partitions_per_worker, total_in)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=memory_limit,
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard: {client.dashboard_link}")

    try:
        bag = db.from_sequence(all_records, npartitions=n_partitions)
        results = (
            bag
            .map(detect_language, model_path=model_path,
                 target_language=target_language, min_score=min_score)
            .filter(lambda x: x is not None)
            .compute()
        )

        # Group results by source file and write output
        docs_by_file = defaultdict(list)
        for doc in results:
            source_file = source_map.get(doc["id"], "unknown.jsonl")
            docs_by_file[source_file].append(doc)

        for source_file, docs in docs_by_file.items():
            output_file = output_path / source_file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as fout:
                for doc in docs:
                    fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

        total_out = len(results)
        retention = (total_out / total_in * 100) if total_in > 0 else 0
        logger.info(
            f"Language filter complete: {total_out}/{total_in} "
            f"documents retained ({retention:.1f}%)"
        )

    finally:
        client.close()
        cluster.close()