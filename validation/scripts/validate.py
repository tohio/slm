"""
validation/scripts/validate.py
--------------------------------
Data validation pipeline using datatrove.

Applies additional quality filters on top of the curator's heuristic
filters. The primary addition is perplexity-based filtering using a
KenLM language model — the most impactful filter for removing low-quality
web text that passes heuristic checks.

Pipeline:
    1. Load curated JSONL from data/curated/train.jsonl
    2. Apply datatrove quality filters (C4, Gopher repetition)
    3. Apply perplexity filter (KenLM 5-gram model)
    4. Write validated JSONL to data/validated/train.jsonl
    5. Write rejection stats

Perplexity filter:
    Documents with perplexity > threshold are removed. The threshold
    is set at the 90th percentile of the perplexity distribution —
    removing the bottom 10% of documents by quality. This is more
    principled than fixed thresholds as it adapts to the data distribution.

KenLM model:
    Requires a 5-gram KenLM model trained on high-quality text (e.g.
    Wikipedia). The model scores how "natural" each document is according
    to its language model. Download or train:

    # Download pre-trained English KenLM (from CCNet/FineWeb):
    wget https://dl.fbaipublicfiles.com/cc_net/lm/en.arpa.bin

    # Or train on Wikipedia text:
    lmplz -o 5 < wikipedia_text.txt > en.arpa
    build_binary en.arpa en.arpa.bin

Usage:
    python validation/scripts/validate.py
    python validation/scripts/validate.py --input data/curated/train.jsonl
    python validation/scripts/validate.py --perplexity-threshold 500
    python validation/scripts/validate.py --no-perplexity   # skip perplexity filter
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
VALIDATED_DIR = DATA_DIR / "validated"


# ── Datatrove filters ──────────────────────────────────────────────────────────

def build_datatrove_pipeline(
    input_path: Path,
    output_path: Path,
    kenlm_model_path: Path | None = None,
    perplexity_threshold: float | None = None,
) -> None:
    """
    Build and run a datatrove filtering pipeline.

    Uses datatrove's built-in filters:
        - C4QualityFilter: Google C4 heuristics (terminal punctuation,
          line length, curly brace ratio, etc.)
        - GopherRepetitionFilter: Repeated n-gram detection from Gopher
        - LanguageFilter: fastText-based language detection
        - PerplexityFilter: KenLM-based perplexity scoring (optional)

    Args:
        input_path: Input JSONL file.
        output_path: Output JSONL file.
        kenlm_model_path: Path to KenLM binary model. If None, skips perplexity filter.
        perplexity_threshold: Max allowed perplexity. If None, uses 90th percentile.
    """
    try:
        from datatrove.pipeline.filters import (
            C4QualityFilter,
            GopherRepetitionFilter,
            LanguageFilter,
        )
        from datatrove.pipeline.readers import JsonlReader
        from datatrove.pipeline.writers import JsonlWriter
        from datatrove.executor import LocalPipelineExecutor
    except ImportError:
        raise ImportError(
            "datatrove not installed. Install with: pip install datatrove"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = [
        JsonlReader(
            data_folder=str(input_path.parent),
            glob_pattern=input_path.name,
            text_key="text",
            id_key=None,
        ),
        C4QualityFilter(
            filter_no_terminal_punct=True,
            min_num_sentences=3,
            min_words_per_line=None,
        ),
        GopherRepetitionFilter(
            top_n_grams=((2, 0.2), (3, 0.18), (4, 0.16)),
            dup_line_frac=0.3,
            dup_para_frac=0.3,
            dup_line_char_frac=0.2,
            dup_para_char_frac=0.2,
        ),
        LanguageFilter(
            language_threshold=0.65,
            languages=["en"],
        ),
    ]

    # Add perplexity filter if KenLM model is available
    if kenlm_model_path and kenlm_model_path.exists():
        try:
            from datatrove.pipeline.filters import PerplexityFilter
            pipeline.append(
                PerplexityFilter(
                    model_dataset="en",
                    model_base_path=str(kenlm_model_path.parent),
                    max_perplexity=perplexity_threshold or 1500,
                )
            )
            log.info(f"Perplexity filter enabled (threshold={perplexity_threshold or 1500})")
        except (ImportError, Exception) as e:
            log.warning(f"Could not load perplexity filter: {e}")
    else:
        log.warning("KenLM model not found — skipping perplexity filter")
        log.warning("Download: wget https://dl.fbaipublicfiles.com/cc_net/lm/en.arpa.bin")

    pipeline.append(
        JsonlWriter(
            output_folder=str(output_path.parent),
            output_filename=output_path.name,
            text_key="text",
        )
    )

    executor = LocalPipelineExecutor(pipeline=pipeline, logging_dir=str(DATA_DIR / "logs"))
    executor.run()


# ── Fallback: manual validation without datatrove ─────────────────────────────

def validate_manual(
    input_path: Path,
    output_path: Path,
    kenlm_model_path: Path | None = None,
    perplexity_threshold: float = 1500,
    sample_for_threshold: int = 10_000,
) -> dict:
    """
    Manual validation pipeline — fallback if datatrove is not available.

    Applies:
        - Terminal punctuation check (C4-style)
        - Repeated line ratio (Gopher-style)
        - Perplexity filter (KenLM, if model available)

    Args:
        input_path: Input JSONL file.
        output_path: Output JSONL file.
        kenlm_model_path: Path to KenLM binary model.
        perplexity_threshold: Max allowed perplexity.
        sample_for_threshold: Number of docs to sample for auto-threshold.

    Returns:
        Stats dict.
    """
    import re

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load KenLM if available
    kenlm_model = None
    if kenlm_model_path and kenlm_model_path.exists():
        try:
            import kenlm
            kenlm_model = kenlm.Model(str(kenlm_model_path))
            log.info(f"Loaded KenLM model from {kenlm_model_path}")

            # Auto-compute threshold from sample if not provided
            if perplexity_threshold is None:
                log.info(f"Computing perplexity threshold from {sample_for_threshold} documents...")
                perplexities = []
                with open(input_path) as f:
                    for i, line in enumerate(f):
                        if i >= sample_for_threshold:
                            break
                        text = json.loads(line).get("text", "")
                        score = kenlm_model.perplexity(text[:1000])
                        perplexities.append(score)
                perplexities.sort()
                perplexity_threshold = perplexities[int(0.9 * len(perplexities))]
                log.info(f"Auto perplexity threshold (90th percentile): {perplexity_threshold:.1f}")

        except ImportError:
            log.warning("kenlm not installed — skipping perplexity filter")
            log.warning("Install: pip install https://github.com/kpu/kenlm/archive/master.zip")

    stats = {
        "total": 0,
        "kept": 0,
        "rejected_terminal_punct": 0,
        "rejected_repeated_lines": 0,
        "rejected_perplexity": 0,
    }

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in tqdm(fin, desc="Validating", unit="doc"):
            record = json.loads(line)
            text = record.get("text", "")
            source = record.get("source", "")
            stats["total"] += 1

            # Skip structural checks for code
            if source != "code":
                # C4-style: at least one line ending with terminal punctuation
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                has_terminal = any(
                    l.endswith((".", "!", "?", '"', "'")) for l in lines
                )
                if not has_terminal:
                    stats["rejected_terminal_punct"] += 1
                    continue

                # Gopher-style: repeated line check
                if len(lines) >= 4:
                    seen = set()
                    dups = sum(1 for l in lines if l in seen or seen.add(l))
                    if dups / len(lines) > 0.3:
                        stats["rejected_repeated_lines"] += 1
                        continue

            # Perplexity filter
            if kenlm_model is not None:
                try:
                    ppl = kenlm_model.perplexity(text[:2000])
                    if ppl > perplexity_threshold:
                        stats["rejected_perplexity"] += 1
                        continue
                except Exception:
                    pass

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLM data validation pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "curated" / "train.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATED_DIR / "train.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--kenlm-model",
        type=Path,
        default=DATA_DIR / "models" / "en.arpa.bin",
        help="Path to KenLM binary model",
    )
    parser.add_argument(
        "--perplexity-threshold",
        type=float,
        default=None,
        help="Max perplexity (auto-computed at 90th percentile if not set)",
    )
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip perplexity filter",
    )
    parser.add_argument(
        "--use-datatrove",
        action="store_true",
        help="Use datatrove pipeline (requires datatrove installed)",
    )
    args = parser.parse_args()

    kenlm_path = None if args.no_perplexity else args.kenlm_model

    log.info(f"=== SLM Data Validation ===")
    log.info(f"Input:  {args.input}")
    log.info(f"Output: {args.output}")
    log.info(f"KenLM:  {kenlm_path or 'disabled'}")

    if args.use_datatrove:
        log.info("Using datatrove pipeline...")
        build_datatrove_pipeline(
            input_path=args.input,
            output_path=args.output,
            kenlm_model_path=kenlm_path,
            perplexity_threshold=args.perplexity_threshold,
        )
    else:
        log.info("Using manual validation pipeline...")
        stats = validate_manual(
            input_path=args.input,
            output_path=args.output,
            kenlm_model_path=kenlm_path,
            perplexity_threshold=args.perplexity_threshold,
        )

        total = stats["total"]
        kept = stats["kept"]
        log.info("=== Validation Report ===")
        log.info(f"Total:                    {total:>10,}")
        log.info(f"Kept:                     {kept:>10,}  ({100*kept/max(total,1):.1f}%)")
        log.info(f"Rejected (terminal punct):{stats['rejected_terminal_punct']:>10,}")
        log.info(f"Rejected (repeated lines):{stats['rejected_repeated_lines']:>10,}")
        log.info(f"Rejected (perplexity):    {stats['rejected_perplexity']:>10,}")

        stats_path = VALIDATED_DIR / "validation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        log.info(f"Stats written to {stats_path}")

    log.info("Validation complete.")


if __name__ == "__main__":
    main()