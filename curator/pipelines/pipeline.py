"""
SLM Data Curation Pipeline
--------------------------
Orchestrates the full curation sequence:
  1. Extract
  2. Language filter
  3. Heuristic filter
  4. Quality classifier filter
  5. Exact deduplication
  6. Fuzzy deduplication (MinHash)
  7. PII redaction
  8. Tokenization → memory-mapped

Each stage checkpoints to disk so the pipeline is resumable
after spot instance preemption.

Usage:
    python pipeline.py --config ../configs/curator.yaml [--start-stage extract]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from extract import run_extraction
from language_filter import run_language_filter
from heuristic_filter import run_heuristic_filter
from quality_filter import run_quality_filter
from dedup import run_exact_dedup, run_fuzzy_dedup
from pii import run_pii_redaction
from tokenize_data import run_tokenization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("curator.pipeline")

# Ordered list of pipeline stages
STAGES = [
    "extract",
    "language_filter",
    "heuristic_filter",
    "quality_filter",
    "exact_dedup",
    "fuzzy_dedup",
    "pii",
    "tokenize",
]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def stage_output_path(base_dir: str, stage: str) -> Path:
    """Each stage writes to its own subdirectory for clean checkpointing."""
    return Path(base_dir) / "stages" / stage


def stage_is_complete(output_path: Path) -> bool:
    """Check if a stage has already been completed (resume support)."""
    marker = output_path / ".complete"
    return marker.exists()


def mark_stage_complete(output_path: Path):
    """Write a completion marker after a stage finishes successfully."""
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / ".complete").touch()
    logger.info(f"Stage complete marker written: {output_path / '.complete'}")


def run_pipeline(config: dict, start_stage: str = "extract"):
    cfg = config["curation"]
    output_base = cfg["output_data_dir"]

    # Set up logging to file as well
    log_dir = Path(cfg.get("log_dir", "/logs/curator"))
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # Determine which stages to run
    if start_stage not in STAGES:
        raise ValueError(f"Unknown start stage: {start_stage}. Valid: {STAGES}")
    stages_to_run = STAGES[STAGES.index(start_stage):]

    logger.info(f"Starting pipeline from stage: {start_stage}")
    logger.info(f"Stages to run: {stages_to_run}")

    pipeline_start = time.time()

    for stage in stages_to_run:
        output_path = stage_output_path(output_base, stage)

        if stage_is_complete(output_path):
            logger.info(f"[SKIP] Stage '{stage}' already complete. Delete {output_path}/.complete to re-run.")
            continue

        logger.info(f"[START] Stage: {stage}")
        stage_start = time.time()

        try:
            if stage == "extract":
                input_path = cfg["input_data_dir"]
                run_extraction(input_path, output_path, cfg)

            elif stage == "language_filter":
                input_path = stage_output_path(output_base, "extract")
                run_language_filter(input_path, output_path, cfg["language_filter"])

            elif stage == "heuristic_filter":
                input_path = stage_output_path(output_base, "language_filter")
                run_heuristic_filter(input_path, output_path, cfg["heuristic_filter"])

            elif stage == "quality_filter":
                input_path = stage_output_path(output_base, "heuristic_filter")
                run_quality_filter(input_path, output_path, cfg["quality_filter"])

            elif stage == "exact_dedup":
                input_path = stage_output_path(output_base, "quality_filter")
                run_exact_dedup(input_path, output_path, cfg["deduplication"]["exact_dedup"])

            elif stage == "fuzzy_dedup":
                input_path = stage_output_path(output_base, "exact_dedup")
                run_fuzzy_dedup(input_path, output_path, cfg["deduplication"]["fuzzy_dedup"])

            elif stage == "pii":
                input_path = stage_output_path(output_base, "fuzzy_dedup")
                run_pii_redaction(input_path, output_path, cfg["pii"])

            elif stage == "tokenize":
                input_path = stage_output_path(output_base, "pii")
                run_tokenization(input_path, output_path, cfg["tokenization"])

        except Exception as e:
            logger.error(f"[FAILED] Stage '{stage}' failed: {e}", exc_info=True)
            logger.error("Pipeline halted. Fix the error and re-run with --start-stage to resume.")
            sys.exit(1)

        elapsed = time.time() - stage_start
        logger.info(f"[DONE] Stage '{stage}' completed in {elapsed:.1f}s")
        mark_stage_complete(output_path)

    total_elapsed = time.time() - pipeline_start
    logger.info(f"Pipeline complete. Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")


def main():
    parser = argparse.ArgumentParser(description="SLM Curator Pipeline")
    parser.add_argument("--config", required=True, help="Path to curator.yaml")
    parser.add_argument(
        "--start-stage",
        default="extract",
        choices=STAGES,
        help="Stage to start from (for resuming after preemption)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, start_stage=args.start_stage)


if __name__ == "__main__":
    main()
