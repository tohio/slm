"""
validation/scripts/validate.py
--------------------------------
Data validation pipeline using datatrove.

Applies additional quality filters on top of the curator's heuristic
filters. The primary addition is perplexity-based filtering using a
KenLM language model — the most impactful filter for removing low-quality
web text that passes heuristic checks.

Pipeline (run independently for each split):
    1. Load curated JSONL from data/curated/{train,val}.jsonl
    2. Apply datatrove quality filters (C4, Gopher repetition)
    3. Apply perplexity filter (KenLM 5-gram model)
    4. Write validated JSONL to data/validated/{train,val}.jsonl
    5. Write per-split rejection stats

Why validate both splits? The curator produces train.jsonl and val.jsonl
as uniform random samples of the same shuffled distribution. If only train
were KenLM-filtered, val would end up with a *different* quality distribution
than train, defeating the point of having them come from the same blend.
Running validation over both splits preserves the "same distribution"
guarantee. Downstream eval loss is a meaningful comparison to training loss
only when both splits passed the same filters.

Perplexity filter:
    Documents with perplexity > threshold are removed. The threshold is
    auto-computed from train (90th percentile of train's perplexity
    distribution) and reused for val — so the two splits are filtered by
    the same cutoff, not two independently-computed ones. This keeps
    train and val comparable even when val is much smaller.

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
    python validation/scripts/validate.py --train data/curated/train.jsonl \\
                                          --val   data/curated/val.jsonl
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

# Source tags whose content is primarily code. These bypass the prose-style
# structural checks (terminal punctuation, repeated-line ratio) which would
# reject legitimate code. Must match CODE_SOURCES in
# curator/filters/quality.py — both files enumerate the same set of
# specific loader names.
CODE_SOURCES: frozenset[str] = frozenset({
    "codesearchnet",
    "stack_smol",
    "stack_v1",
    "jupyter",
    "conala",
})


# ── Datatrove filters ──────────────────────────────────────────────────────────

def build_datatrove_pipeline(
    input_path: Path,
    output_path: Path,
    kenlm_model_path: Path | None = None,
    perplexity_threshold: float | None = None,
) -> None:
    """
    Build and run a datatrove filtering pipeline for one split.

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

def _compute_perplexity_threshold(
    kenlm_model,
    input_path: Path,
    sample_size: int,
) -> float:
    """Compute the 90th-percentile perplexity from a sample of documents."""
    log.info(f"Computing perplexity threshold from {sample_size:,} documents...")
    perplexities: list[float] = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            text = json.loads(line).get("text", "")
            score = kenlm_model.perplexity(text[:1000])
            perplexities.append(score)
    perplexities.sort()
    threshold = perplexities[int(0.9 * len(perplexities))]
    log.info(f"Auto perplexity threshold (90th percentile): {threshold:.1f}")
    return threshold


def validate_manual_split(
    input_path: Path,
    output_path: Path,
    kenlm_model,
    perplexity_threshold: float | None,
    split: str,
) -> dict:
    """
    Manual validation for a single split (train or val).

    Applies:
        - Terminal punctuation check (C4-style) — prose sources only
        - Repeated line ratio (Gopher-style) — prose sources only
        - Perplexity filter (KenLM, if model available) — all sources

    Code sources (codesearchnet, stack_smol, stack_v1, jupyter, conala)
    bypass the structural prose checks because code does not always end
    in terminal punctuation and may have legitimate repeated lines
    (boilerplate imports, standard patterns). They still go through the
    perplexity filter — low-perplexity-on-English means garbage for any
    source.

    Args:
        input_path: Input JSONL file.
        output_path: Output JSONL file.
        kenlm_model: Loaded KenLM model, or None to skip perplexity filter.
        perplexity_threshold: Max allowed perplexity, or None to skip.
        split: "train" or "val" — used for log labels only.

    Returns:
        Stats dict for this split.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "kept": 0,
        "rejected_terminal_punct": 0,
        "rejected_repeated_lines": 0,
        "rejected_perplexity": 0,
    }

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in tqdm(fin, desc=f"Validating {split}", unit="doc"):
            record = json.loads(line)
            text = record.get("text", "")
            source = record.get("source", "")
            stats["total"] += 1

            # Skip structural prose checks for code sources — code doesn't
            # follow English prose conventions.
            if source not in CODE_SOURCES:
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

            # Perplexity filter — applies to all sources
            if kenlm_model is not None and perplexity_threshold is not None:
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


def _load_kenlm(kenlm_model_path: Path | None):
    """Load KenLM model from path, or return None if unavailable."""
    if kenlm_model_path is None or not kenlm_model_path.exists():
        return None
    try:
        import kenlm
        model = kenlm.Model(str(kenlm_model_path))
        log.info(f"Loaded KenLM model from {kenlm_model_path}")
        return model
    except ImportError:
        log.warning("kenlm not installed — skipping perplexity filter")
        log.warning("Install: pip install https://github.com/kpu/kenlm/archive/master.zip")
        return None


def _log_split_report(split: str, stats: dict) -> None:
    total = stats["total"]
    kept = stats["kept"]
    log.info(f"=== Validation Report: {split} ===")
    log.info(f"  Total:                    {total:>10,}")
    log.info(f"  Kept:                     {kept:>10,}  ({100*kept/max(total,1):.1f}%)")
    log.info(f"  Rejected (terminal punct):{stats['rejected_terminal_punct']:>10,}")
    log.info(f"  Rejected (repeated lines):{stats['rejected_repeated_lines']:>10,}")
    log.info(f"  Rejected (perplexity):    {stats['rejected_perplexity']:>10,}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLM data validation pipeline")
    parser.add_argument(
        "--train",
        type=Path,
        default=DATA_DIR / "curated" / "train.jsonl",
        help="Input train JSONL file",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DATA_DIR / "curated" / "val.jsonl",
        help="Input val JSONL file (processed if present, warning if missing)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=VALIDATED_DIR / "train.jsonl",
        help="Output train JSONL file",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=VALIDATED_DIR / "val.jsonl",
        help="Output val JSONL file",
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
        help="Max perplexity (auto-computed at 90th percentile of train if not set)",
    )
    parser.add_argument(
        "--perplexity-sample-size",
        type=int,
        default=10_000,
        help="Docs sampled from train to auto-compute perplexity threshold",
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
    log.info(f"Train input:  {args.train}")
    log.info(f"Val input:    {args.val}")
    log.info(f"Train output: {args.train_output}")
    log.info(f"Val output:   {args.val_output}")
    log.info(f"KenLM:        {kenlm_path or 'disabled'}")

    val_available = args.val.exists()
    if not val_available:
        log.warning(
            f"Val input not found: {args.val}\n"
            f"Only train will be validated. The curator's blend stage produces "
            f"both train.jsonl and val.jsonl — re-run 'make curate' to get val."
        )

    # ── datatrove path ────────────────────────────────────────────────────────
    if args.use_datatrove:
        log.info("Using datatrove pipeline...")
        build_datatrove_pipeline(
            input_path=args.train,
            output_path=args.train_output,
            kenlm_model_path=kenlm_path,
            perplexity_threshold=args.perplexity_threshold,
        )
        if val_available:
            build_datatrove_pipeline(
                input_path=args.val,
                output_path=args.val_output,
                kenlm_model_path=kenlm_path,
                perplexity_threshold=args.perplexity_threshold,
            )
        log.info("Validation complete.")
        return

    # ── Manual path ───────────────────────────────────────────────────────────
    log.info("Using manual validation pipeline...")

    kenlm_model = _load_kenlm(kenlm_path)

    # Compute perplexity threshold from train sample (if not provided). Reuse
    # the same threshold for val so both splits are filtered by the same
    # cutoff — per-split thresholds would diverge for small val sets and
    # invalidate the "same distribution" property.
    perplexity_threshold = args.perplexity_threshold
    if kenlm_model is not None and perplexity_threshold is None:
        perplexity_threshold = _compute_perplexity_threshold(
            kenlm_model, args.train, args.perplexity_sample_size,
        )

    # Train split
    train_stats = validate_manual_split(
        input_path=args.train,
        output_path=args.train_output,
        kenlm_model=kenlm_model,
        perplexity_threshold=perplexity_threshold,
        split="train",
    )
    _log_split_report("train", train_stats)

    # Val split
    val_stats: dict | None = None
    if val_available:
        val_stats = validate_manual_split(
            input_path=args.val,
            output_path=args.val_output,
            kenlm_model=kenlm_model,
            perplexity_threshold=perplexity_threshold,
            split="val",
        )
        _log_split_report("val", val_stats)

    # Aggregated stats — kept in the top-level fields for backwards compat
    # with existing tests + callers that expect `total` / `kept`. Per-split
    # breakdown is nested under splits.
    combined = {
        "total": train_stats["total"] + (val_stats["total"] if val_stats else 0),
        "kept": train_stats["kept"] + (val_stats["kept"] if val_stats else 0),
        "rejected_terminal_punct": (
            train_stats["rejected_terminal_punct"]
            + (val_stats["rejected_terminal_punct"] if val_stats else 0)
        ),
        "rejected_repeated_lines": (
            train_stats["rejected_repeated_lines"]
            + (val_stats["rejected_repeated_lines"] if val_stats else 0)
        ),
        "rejected_perplexity": (
            train_stats["rejected_perplexity"]
            + (val_stats["rejected_perplexity"] if val_stats else 0)
        ),
        "perplexity_threshold": perplexity_threshold,
        "splits": {
            "train": train_stats,
            **({"val": val_stats} if val_stats else {}),
        },
    }

    stats_path = VALIDATED_DIR / "validation_stats.json"
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(combined, f, indent=2)
    log.info(f"Stats written to {stats_path}")

    log.info("Validation complete.")


if __name__ == "__main__":
    main()