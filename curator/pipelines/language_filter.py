"""
Stage 2: Language Filter
------------------------
Identifies and retains only English documents using fastText's
language identification model (lid.176.bin).

Removes:
  - Non-English documents
  - Documents where the language model is uncertain (low confidence)

Parallelism:
  Documents are distributed across all Dask workers at the record level.
  Files are processed one at a time to keep peak memory bounded —
  loading all 671k records at once causes the Dask scheduler to OOM
  when serializing the 2GB+ task graph. Processing one file at a time
  (~33k records, ~100MB) keeps peak memory well within budget.

Input/Output: JSONL
"""

import json
import logging
from pathlib import Path

import dask.bag as db
from dask.distributed import Client, LocalCluster

logger = logging.getLogger("curator.language_filter")

PARTITIONS_PER_WORKER = 4


def detect_language(record: dict, model_path: str, target_language: str, min_score: float) -> dict | None:
    """
    Detect language of a single document using fastText.
    Loads the model once per worker (cached in worker memory).
    Returns None if the document does not pass the language filter.
    """
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

    Processes one JSONL file at a time to keep peak memory bounded.
    Each file's records are distributed across all Dask workers,
    results written to disk, then memory freed before the next file.
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

    dask_cfg = cfg.get("dask", {})
    n_workers = dask_cfg.get("n_workers", 4)
    memory_limit = dask_cfg.get("memory_limit", "2GB")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=memory_limit,
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard: {client.dashboard_link}")

    total_in = 0
    total_out = 0

    try:
        for input_file in input_files:
            # Load one file at a time — keeps peak memory bounded
            records = []
            with open(input_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            n_records = len(records)
            total_in += n_records

            if not records:
                continue

            n_partitions = min(n_workers * PARTITIONS_PER_WORKER, n_records)

            bag = db.from_sequence(records, npartitions=n_partitions)
            results = (
                bag
                .map(detect_language, model_path=model_path,
                     target_language=target_language, min_score=min_score)
                .filter(lambda x: x is not None)
                .compute()
            )

            if results:
                output_file = output_path / input_file.name
                with open(output_file, "w", encoding="utf-8") as fout:
                    for doc in results:
                        fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

            n_out = len(results)
            total_out += n_out
            retention = n_out / n_records * 100 if n_records > 0 else 0
            logger.info(
                f"  {input_file.name}: {n_out}/{n_records} retained "
                f"({retention:.1f}%)"
            )

            # Free memory before next file
            del records, results

    finally:
        client.close()
        cluster.close()

    overall_retention = (total_out / total_in * 100) if total_in > 0 else 0
    logger.info(
        f"Language filter complete: {total_out:,}/{total_in:,} "
        f"documents retained ({overall_retention:.1f}%)"
    )