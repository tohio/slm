"""
Stage 4: Quality Classifier Filter
------------------------------------
Uses a fastText binary classifier to distinguish high-quality text
(Wikipedia-like) from low-quality web text.

The classifier is trained separately on:
  - Positive examples: Wikipedia, books, curated sources
  - Negative examples: random CommonCrawl pages

This catches quality issues that heuristic rules miss — things like
grammatically correct but semantically empty SEO content.

Input/Output: JSONL
"""

import json
import logging
from pathlib import Path
from collections import Counter

import fasttext

logger = logging.getLogger("curator.quality_filter")

_model = None


def load_model(model_path: str):
    global _model
    if _model is None:
        _model = fasttext.load_model(model_path)
        logger.info(f"Quality classifier loaded from {model_path}")
    return _model


def score_document(text: str, model, label: str) -> float:
    """
    Score a document using the fastText classifier.
    Returns the probability for the high-quality label.
    Uses first 512 words — enough signal, avoids very long inference.
    """
    sample = " ".join(text.split()[:512]).replace("\n", " ")
    predictions = model.predict(sample, k=2)

    labels = predictions[0]
    probs = predictions[1]

    label_prob_map = {l: p for l, p in zip(labels, probs)}
    return float(label_prob_map.get(label, 0.0))


def process_jsonl_file(
    input_file: Path,
    output_file: Path,
    model_path: str,
    label: str,
    min_score: float,
) -> dict:
    """Process a single JSONL file through quality classifier."""
    kept = 0
    total = 0
    score_buckets = Counter()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path)

    with open(input_file, encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            doc = json.loads(line)
            text = doc.get("text", "")

            score = score_document(text, model, label)

            # Track score distribution for analysis
            bucket = round(score, 1)
            score_buckets[bucket] += 1

            if score >= min_score:
                doc["quality_score"] = round(score, 4)
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1

    return {
        "file": input_file.name,
        "total": total,
        "kept": kept,
        "score_distribution": dict(sorted(score_buckets.items())),
    }


def run_quality_filter(input_path: Path, output_path: Path, cfg: dict):
    """Main quality filter entry point."""
    if not cfg.get("enabled", True):
        logger.info("Quality filter disabled — symlinking input to output")
        output_path.symlink_to(input_path)
        return

    output_path.mkdir(parents=True, exist_ok=True)

    model_path = cfg["model_path"]
    label = cfg.get("label", "__label__hq")
    min_score = cfg.get("min_quality_score", 0.3)

    input_files = sorted(input_path.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {input_path}")

    logger.info(f"Quality filtering {len(input_files)} files (min_score={min_score}, label={label})")

    stats_list = []
    for input_file in input_files:
        output_file = output_path / input_file.name
        stats = process_jsonl_file(input_file, output_file, model_path, label, min_score)
        stats_list.append(stats)
        logger.debug(f"{stats['file']}: {stats['kept']}/{stats['total']} kept")

    total_in = sum(s["total"] for s in stats_list)
    total_out = sum(s["kept"] for s in stats_list)
    retention = (total_out / total_in * 100) if total_in > 0 else 0
    logger.info(f"Quality filter complete: {total_out}/{total_in} documents retained ({retention:.1f}%)")
