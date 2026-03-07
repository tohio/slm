"""
Stage 2: Language Filter
------------------------
Identifies and retains only English documents using fastText's
language identification model (lid.176.bin).

Removes:
  - Non-English documents
  - Documents where the language model is uncertain (low confidence)

Input/Output: JSONL
"""

import json
import logging
from pathlib import Path

import fasttext
import dask.bag as db
from dask.distributed import Client, LocalCluster

logger = logging.getLogger("curator.language_filter")

# Module-level model holder — loaded once per worker
_model = None


def load_model(model_path: str) -> fasttext.FastText._FastText:
    global _model
    if _model is None:
        _model = fasttext.load_model(model_path)
        logger.info(f"fastText language model loaded from {model_path}")
    return _model


def detect_language(text: str, model) -> tuple[str, float]:
    """
    Detect language of text using fastText.
    Returns (language_code, confidence_score).
    fastText returns labels like '__label__en', '__label__fr' etc.
    """
    # Use first 1000 chars for speed — sufficient for language detection
    sample = text[:1000].replace("\n", " ")
    predictions = model.predict(sample, k=1)
    label = predictions[0][0].replace("__label__", "")
    confidence = float(predictions[1][0])
    return label, confidence


def filter_document(doc: dict, model_path: str, target_language: str, min_score: float) -> dict | None:
    """Filter a single document by language. Returns None if filtered out."""
    model = load_model(model_path)
    text = doc.get("text", "")

    if not text:
        return None

    lang, confidence = detect_language(text, model)

    if lang != target_language or confidence < min_score:
        return None

    # Enrich doc with language metadata
    doc["language"] = lang
    doc["language_score"] = round(confidence, 4)
    return doc


def process_jsonl_file(
    input_file: Path,
    output_file: Path,
    model_path: str,
    target_language: str,
    min_score: float,
) -> dict:
    """Process a single JSONL file, writing filtered results."""
    kept = 0
    total = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            doc = json.loads(line)
            result = filter_document(doc, model_path, target_language, min_score)
            if result is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                kept += 1

    return {"file": input_file.name, "total": total, "kept": kept}


def run_language_filter(input_path: Path, output_path: Path, cfg: dict):
    """
    Main language filter entry point.
    Processes all JSONL files from the extraction stage.
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

    logger.info(f"Language filtering {len(input_files)} files (target={target_language}, min_score={min_score})")

    stats_list = []
    for input_file in input_files:
        output_file = output_path / input_file.name
        stats = process_jsonl_file(input_file, output_file, model_path, target_language, min_score)
        stats_list.append(stats)
        logger.debug(f"{stats['file']}: {stats['kept']}/{stats['total']} kept")

    total_in = sum(s["total"] for s in stats_list)
    total_out = sum(s["kept"] for s in stats_list)
    retention = (total_out / total_in * 100) if total_in > 0 else 0
    logger.info(f"Language filter complete: {total_out}/{total_in} documents retained ({retention:.1f}%)")
