"""
SLM Data Curation Pipeline
--------------------------
Orchestrates the full curation sequence:
  1. Extract
  2. Language filter
  3. Heuristic filter
  4. Quality classifier filter  (auto-skipped if model not found)
  5. Exact deduplication
  6. Fuzzy deduplication (MinHash)
  7. PII redaction
  8. Tokenization → memory-mapped  (auto-skipped if tokenizer not found)

Each stage checkpoints to disk so the pipeline is resumable
after instance preemption.

Auto-detection behaviour:
  - quality_filter: runs if enabled AND model file exists, skipped otherwise
  - tokenize:       runs if enabled AND tokenizer model exists, skipped otherwise
  No manual edits to curator.yaml required between curation passes.

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


def should_run_quality_filter(cfg: dict) -> bool:
    """
    Auto-detect whether to run quality filter.
    Runs only if both:
      - quality_filter.enabled is true in config
      - the model file actually exists on disk
    This avoids requiring manual curator.yaml edits between curation passes.
    """
    qf_cfg = cfg.get("quality_filter", {})
    if not qf_cfg.get("enabled", False):
        return False
    model_path = qf_cfg.get("model_path", "")
    if not Path(model_path).exists():
        logger.warning(
            f"[quality_filter] enabled=true but model not found at '{model_path}'. "
            f"Skipping. Run 'make train-quality-classifier' to generate it."
        )
        return False
    return True


def should_run_tokenization(cfg: dict) -> bool:
    """
    Auto-detect whether to run tokenization.
    Runs only if both:
      - tokenization.enabled is true in config
      - the tokenizer model file actually exists on disk
    This avoids requiring manual curator.yaml edits between curation passes.
    """
    tok_cfg = cfg.get("tokenization", {})
    if not tok_cfg.get("enabled", False):
        return False
    model_path = tok_cfg.get("tokenizer_model", "")
    if not Path(model_path).exists():
        logger.warning(
            f"[tokenize] enabled=true but tokenizer not found at '{model_path}'. "
            f"Skipping. Run 'make tokenizer' to generate it."
        )
        return False
    return True


def run_pipeline(config: dict, start_stage: str = "extract"):
    cfg = config["curation"]
    output_base = cfg["output_data_dir"]

    # Set up logging to file as well
    log_dir = Path(cfg.get("log_dir", "/data/logs/curator"))
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

    # Log auto-detection status upfront so it's clear before pipeline starts
    logger.info(
        f"[quality_filter] auto-detect: "
        f"{'WILL RUN' if should_run_quality_filter(cfg) else 'WILL SKIP'}"
    )
    logger.info(
        f"[tokenize] auto-detect: "
        f"{'WILL RUN' if should_run_tokenization(cfg) else 'WILL SKIP'}"
    )

    pipeline_start = time.time()

    for stage in stages_to_run:
        output_path = stage_output_path(output_base, stage)

        if stage_is_complete(output_path):
            logger.info(
                f"[SKIP] Stage '{stage}' already complete. "
                f"Delete {output_path}/.complete to re-run."
            )
            continue

        # Auto-detection for optional stages
        if stage == "quality_filter" and not should_run_quality_filter(cfg):
            logger.info("[SKIP] quality_filter — model not available, continuing to next stage")
            # Mark complete so exact_dedup reads from heuristic_filter output
            _skip_stage_redirect(output_base, "quality_filter", "heuristic_filter")
            continue

        if stage == "tokenize" and not should_run_tokenization(cfg):
            logger.info("[SKIP] tokenize — tokenizer model not available, pipeline complete")
            continue

        logger.info(f"[START] Stage: {stage}")
        stage_start = time.time()

        try:
            if stage == "extract":
                input_path = cfg["input_data_dir"]
                run_extraction(input_path, output_path, cfg)

            elif stage == "language_filter":
                input_path = stage_output_path(output_base, "extract")
                # Pass full cfg so language_filter can read dask config for n_workers
                lang_cfg = cfg["language_filter"]
                lang_cfg["dask"] = cfg.get("dask", {})
                run_language_filter(input_path, output_path, lang_cfg)

            elif stage == "heuristic_filter":
                input_path = stage_output_path(output_base, "language_filter")
                run_heuristic_filter(input_path, output_path, cfg["heuristic_filter"])

            elif stage == "quality_filter":
                input_path = stage_output_path(output_base, "heuristic_filter")
                run_quality_filter(input_path, output_path, cfg["quality_filter"])

            elif stage == "exact_dedup":
                # If quality_filter was skipped, read from heuristic_filter
                input_path = _resolve_input(output_base, "quality_filter", "heuristic_filter")
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
            logger.error(
                "Pipeline halted. Fix the error and re-run with "
                f"--start-stage {stage} to resume."
            )
            sys.exit(1)

        elapsed = time.time() - stage_start
        logger.info(f"[DONE] Stage '{stage}' completed in {elapsed:.1f}s")
        mark_stage_complete(output_path)

    total_elapsed = time.time() - pipeline_start
    logger.info(
        f"Pipeline complete. Total time: {total_elapsed:.1f}s "
        f"({total_elapsed/3600:.2f}h)"
    )


def _skip_stage_redirect(output_base: str, skipped_stage: str, fallback_stage: str):
    """
    When a stage is skipped, create a symlink so downstream stages can
    find their input without needing to know which stage was skipped.
    """
    skipped_path = stage_output_path(output_base, skipped_stage)
    fallback_path = stage_output_path(output_base, fallback_stage)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    if not skipped_path.exists():
        skipped_path.symlink_to(fallback_path.resolve())
        logger.info(
            f"[REDIRECT] {skipped_stage} → {fallback_stage} "
            f"(symlink: {skipped_path} → {fallback_path})"
        )


def _resolve_input(output_base: str, preferred_stage: str, fallback_stage: str) -> Path:
    """
    Return the output path of preferred_stage if it completed,
    otherwise fall back to fallback_stage output.
    """
    preferred = stage_output_path(output_base, preferred_stage)
    if stage_is_complete(preferred) or preferred.exists():
        return preferred
    fallback = stage_output_path(output_base, fallback_stage)
    logger.info(
        f"[RESOLVE] {preferred_stage} not found, using {fallback_stage} as input"
    )
    return fallback


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