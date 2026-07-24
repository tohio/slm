"""
validation/scripts/validate.py
--------------------------------
Canonical source-aware data validation pipeline.

Applies additional quality filters on top of the curator's heuristic
filters. The primary addition is perplexity-based filtering using a
KenLM language model — the most impactful filter for removing low-quality
web text that passes heuristic checks.

Pipeline (run independently for each split):
    1. Load curated JSONL from data/runs/<size>/curated/{train,val}.jsonl
    2. Apply source-aware C4/Gopher-style checks
    3. Apply perplexity filter (KenLM 5-gram model)
    4. Write validated JSONL to data/runs/<size>/validated/{train,val}.jsonl
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
    python validation/scripts/validate.py --size 125m
    python validation/scripts/validate.py --train data/runs/125m/curated/train.jsonl \\
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

from config import PROSE_HEURISTIC_SKIP_SOURCES
from config.paths import curated_dir, validated_dir, BASE_DATA_DIR
from curator.state import (
    atomic_write_json,
    code_fingerprint,
    file_snapshot,
    manifest_matches,
    manifest_outputs_match,
    stable_digest,
    write_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = BASE_DATA_DIR

def _skip_prose_heuristics(record: dict) -> bool:
    """Return True when prose-only validation checks should be bypassed."""
    return record.get("source") in PROSE_HEURISTIC_SKIP_SOURCES


# ── Canonical validation ───────────────────────────────────────────────────────

def _compute_perplexity_threshold(
    kenlm_model,
    input_path: Path,
    sample_size: int,
) -> float | None:
    """Compute the 90th-percentile perplexity from prose-like train documents."""
    log.info(
        f"Computing perplexity threshold from up to {sample_size:,} prose-like documents..."
    )
    perplexities: list[float] = []

    with open(input_path) as f:
        for line in f:
            if len(perplexities) >= sample_size:
                break
            if not line.strip():
                continue

            record = json.loads(line)
            if _skip_prose_heuristics(record):
                continue

            text = record.get("text", "")
            if not text:
                continue

            score = kenlm_model.perplexity(text[:1000])
            perplexities.append(score)

    if not perplexities:
        log.warning(
            "No prose-like documents available for perplexity threshold; "
            "perplexity filtering will be skipped."
        )
        return None

    perplexities.sort()
    idx = min(int(0.9 * len(perplexities)), len(perplexities) - 1)
    threshold = perplexities[idx]
    log.info(
        f"Auto perplexity threshold (90th percentile, n={len(perplexities):,}): "
        f"{threshold:.1f}"
    )
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
        - Perplexity filter (KenLM, when enabled) — prose sources only

    Code sources (codesearchnet, stack_smol, stack_v1, jupyter, conala)
    bypass the structural prose checks because code does not always end
    in terminal punctuation and may have legitimate repeated lines
    (boilerplate imports, standard patterns). They also bypass prose KenLM,
    which is not meaningful for code, math templates, or symbol-heavy data.

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
        "skipped_prose_heuristics": 0,
    }

    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with open(input_path) as fin, open(tmp_path, "w") as fout:
            for line in tqdm(fin, desc=f"Validating {split}", unit="doc"):
                record = json.loads(line)
                text = record.get("text", "")
                stats["total"] += 1

                skip_prose = _skip_prose_heuristics(record)
                if skip_prose:
                    stats["skipped_prose_heuristics"] += 1

                lines = [l.strip() for l in text.split("\n") if l.strip()]

                # C4-style terminal punctuation is a prose-only heuristic.
                if not skip_prose:
                    has_terminal = any(
                        l.endswith((".", "!", "?", '"', "'")) for l in lines
                    )
                    if not has_terminal:
                        stats["rejected_terminal_punct"] += 1
                        continue

                # Gopher-style repeated line check can catch broken output for
                # any source, including code/template-like records.
                if len(lines) >= 4:
                    seen = set()
                    dups = 0
                    for text_line in lines:
                        if text_line in seen:
                            dups += 1
                        else:
                            seen.add(text_line)
                    if dups / len(lines) > 0.3:
                        stats["rejected_repeated_lines"] += 1
                        continue

                # KenLM perplexity is an English prose heuristic. It is not
                # meaningful for code/templates/symbol-heavy math.
                if (
                    not skip_prose
                    and kenlm_model is not None
                    and perplexity_threshold is not None
                ):
                    try:
                        ppl = kenlm_model.perplexity(text[:2000])
                        if ppl > perplexity_threshold:
                            stats["rejected_perplexity"] += 1
                            continue
                    except Exception as exc:
                        raise RuntimeError(
                            f"KenLM failed while scoring a {split} document"
                        ) from exc

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["kept"] += 1
            fout.flush()
            os.fsync(fout.fileno())
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return stats


def _load_kenlm(kenlm_model_path: Path | None):
    """Load the required KenLM model, or return None when explicitly disabled."""
    if kenlm_model_path is None:
        return None
    if not kenlm_model_path.exists():
        raise FileNotFoundError(
            f"KenLM model not found: {kenlm_model_path}. Download/configure it, "
            f"or pass --no-perplexity to explicitly record a no-KenLM run."
        )
    try:
        import kenlm
        model = kenlm.Model(str(kenlm_model_path))
        log.info(f"Loaded KenLM model from {kenlm_model_path}")
        return model
    except ImportError as exc:
        raise RuntimeError(
            "kenlm is required when perplexity filtering is enabled"
        ) from exc


def _log_split_report(split: str, stats: dict) -> None:
    total = stats["total"]
    kept = stats["kept"]
    log.info(f"=== Validation Report: {split} ===")
    log.info(f"  Total:                    {total:>10,}")
    log.info(f"  Kept:                     {kept:>10,}  ({100*kept/max(total,1):.1f}%)")
    log.info(f"  Rejected (terminal punct):{stats['rejected_terminal_punct']:>10,}")
    log.info(f"  Rejected (repeated lines):{stats['rejected_repeated_lines']:>10,}")
    log.info(f"  Rejected (perplexity):    {stats['rejected_perplexity']:>10,}")
    log.info(f"  Skipped prose heuristics: {stats.get('skipped_prose_heuristics', 0):>10,}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLM data validation pipeline")
    parser.add_argument("--size", default=os.environ.get("SIZE", "125m"), help="Run size: mini, 125m, 350m, or 1b")
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Input train JSONL file",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=None,
        help="Input val JSONL file (processed if present, warning if missing)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Output train JSONL file",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=None,
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
    args = parser.parse_args()

    run_curated_dir = curated_dir(args.size)
    run_validated_dir = validated_dir(args.size)
    args.train = args.train or (run_curated_dir / "train.jsonl")
    args.val = args.val or (run_curated_dir / "val.jsonl")
    args.train_output = args.train_output or (run_validated_dir / "train.jsonl")
    args.val_output = args.val_output or (run_validated_dir / "val.jsonl")

    kenlm_path = None if args.no_perplexity else args.kenlm_model

    log.info(f"=== SLM Data Validation ===")
    log.info(f"Train input:  {args.train}")
    log.info(f"Val input:    {args.val}")
    log.info(f"Train output: {args.train_output}")
    log.info(f"Val output:   {args.val_output}")
    log.info(f"KenLM:        {kenlm_path or 'disabled'}")

    if not args.train.exists():
        raise FileNotFoundError(f"Train input not found: {args.train}")
    if not args.val.exists():
        raise FileNotFoundError(
            f"Val input not found: {args.val}. Validation requires both splits "
            f"so train and val retain the same filtering contract."
        )
    val_available = True
    if (
        args.train.parent == run_curated_dir
        and args.val.parent == run_curated_dir
        and not manifest_outputs_match(
            run_curated_dir,
            output_pattern="*.json*",
        )
    ):
        raise RuntimeError(
            f"Curated inputs are not manifest-complete: {run_curated_dir}"
        )

    # ── Manual path ───────────────────────────────────────────────────────────
    log.info("Using canonical source-aware validation pipeline...")

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

    input_signature = stable_digest(
        {
            "train": file_snapshot([args.train], root=args.train.parent),
            "val": file_snapshot([args.val], root=args.val.parent),
        }
    )
    validation_contract = {
        "implementation_sha256": code_fingerprint(validate_manual_split),
        "prose_heuristic_skip_sources": PROSE_HEURISTIC_SKIP_SOURCES,
        "perplexity_enabled": kenlm_model is not None,
        "perplexity_threshold": perplexity_threshold,
        "perplexity_sample_size": args.perplexity_sample_size,
        "kenlm_model": (
            file_snapshot([kenlm_path], root=kenlm_path.parent)[0]
            if kenlm_path is not None
            else None
        ),
    }
    common_output_dir = (
        args.train_output.parent
        if args.train_output.parent == args.val_output.parent
        else None
    )
    if common_output_dir is not None and manifest_matches(
        common_output_dir,
        stage="validate",
        contract=validation_contract,
        input_signature=input_signature,
        output_pattern="*.json*",
    ):
        log.info("Verified validation manifest matches inputs/configuration — reusing")
        return

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
        "skipped_prose_heuristics": (
            train_stats.get("skipped_prose_heuristics", 0)
            + (val_stats.get("skipped_prose_heuristics", 0) if val_stats else 0)
        ),
        "perplexity_threshold": perplexity_threshold,
        "splits": {
            "train": train_stats,
            **({"val": val_stats} if val_stats else {}),
        },
    }

    stats_dir = common_output_dir or args.train_output.parent
    stats_path = stats_dir / "validation_stats.json"
    stats_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(stats_path, combined)
    if common_output_dir is not None:
        write_manifest(
            common_output_dir,
            stage="validate",
            contract=validation_contract,
            input_signature=input_signature,
            output_pattern="*.json*",
        )
    log.info(f"Stats written to {stats_path}")

    log.info("Validation complete.")


if __name__ == "__main__":
    main()
