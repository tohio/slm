"""
validation/scripts/validate.py
--------------------------------
Canonical source-aware data validation pipeline.

Applies additional source-aware checks on top of the curator's filters and
records KenLM perplexity measurements for eligible prose.

Pipeline (run independently for each split):
    1. Load curated JSONL from data/runs/<size>/curated/{train,val}.jsonl
    2. Apply source-aware C4/Gopher-style checks
    3. Measure perplexity with a KenLM 5-gram model
    4. Write validated JSONL to data/runs/<size>/validated/{train,val}.jsonl
    5. Write per-split rejection stats

Why validate both splits? The curator produces train.jsonl and val.jsonl
as uniform random samples of the same shuffled distribution. If only train
were KenLM-filtered, val would end up with a *different* quality distribution
than train, defeating the point of having them come from the same blend.
Running validation over both splits preserves the "same distribution"
guarantee. Downstream eval loss is a meaningful comparison to training loss
only when both splits passed the same filters.

Perplexity policy:
    KenLM is report-only by default. It records deterministic, bounded
    per-source distribution summaries without changing corpus membership.
    Documents are removed only when an explicit --perplexity-threshold is
    supplied. This avoids treating a self-derived corpus percentile as a
    validated quality boundary.

KenLM model:
    Requires the matched CCNet English model pair. CCNet's KenLM model was
    trained on normalized SentencePiece output, so raw text must be processed
    with en.sp.model before en.arpa.bin can score it:

    # Download the pretrained English CCNet pair:
    wget https://dl.fbaipublicfiles.com/cc_net/lm/en.arpa.bin
    wget https://dl.fbaipublicfiles.com/cc_net/lm/en.sp.model

    Use ``make download-kenlm-model`` so both files are installed together.

Usage:
    python validation/scripts/validate.py
    python validation/scripts/validate.py --size 125m
    python validation/scripts/validate.py --train data/runs/125m/curated/train.jsonl \\
    python validation/scripts/validate.py --perplexity-threshold 500
    python validation/scripts/validate.py --no-perplexity   # skip perplexity filter
"""

import argparse
import hashlib
import heapq
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
from validation.ccnet_perplexity import (
    CCNetPerplexityScorer,
    normalize_ccnet_text,
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

class PerplexityAudit:
    """Bounded deterministic per-source KenLM distribution summary."""

    def __init__(self, sample_size: int):
        if sample_size < 1:
            raise ValueError("perplexity sample size must be positive")
        self.sample_size = sample_size
        self._sources: dict[str, dict] = {}

    def observe(self, source: str, text: str, score: float) -> None:
        if not isinstance(score, (int, float)) or not (0.0 <= score < float("inf")):
            raise RuntimeError(
                f"KenLM returned invalid perplexity for source {source!r}: {score!r}"
            )
        row = self._sources.setdefault(
            source,
            {
                "documents": 0,
                "sum": 0.0,
                "min": float(score),
                "max": float(score),
                "sample": [],
            },
        )
        value = float(score)
        row["documents"] += 1
        row["sum"] += value
        row["min"] = min(row["min"], value)
        row["max"] = max(row["max"], value)

        digest = hashlib.sha256(
            source.encode("utf-8") + b"\0" + text.encode("utf-8")
        ).digest()
        priority = int.from_bytes(digest[:8], "big")
        candidate = (-priority, value)
        sample = row["sample"]
        if len(sample) < self.sample_size:
            heapq.heappush(sample, candidate)
        elif candidate > sample[0]:
            heapq.heapreplace(sample, candidate)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        index = round((len(values) - 1) * fraction)
        return values[index]

    def report(self, *, policy: str, threshold: float | None) -> dict:
        by_source = {}
        for source, row in sorted(self._sources.items()):
            values = sorted(value for _, value in row["sample"])
            by_source[source] = {
                "documents_scored": row["documents"],
                "sampled_documents": len(values),
                "mean": row["sum"] / row["documents"],
                "min": row["min"],
                "p50": self._percentile(values, 0.50),
                "p90": self._percentile(values, 0.90),
                "p95": self._percentile(values, 0.95),
                "p99": self._percentile(values, 0.99),
                "max": row["max"],
            }
        return {
            "policy": policy,
            "threshold": threshold,
            "sample_selection": "bottom_sha256_64_per_source",
            "sample_size_per_source": self.sample_size,
            "documents_scored": sum(
                row["documents"] for row in self._sources.values()
            ),
            "by_source": by_source,
        }


def validate_manual_split(
    input_path: Path,
    output_path: Path,
    kenlm_model,
    perplexity_threshold: float | None,
    split: str,
    perplexity_sample_size: int = 10_000,
) -> dict:
    """
    Manual validation for a single split (train or val).

    Applies:
        - Terminal punctuation check (C4-style) — prose sources only
        - Repeated line ratio (Gopher-style) — prose sources only
        - Perplexity measurement (KenLM, when enabled) — prose sources only
        - Perplexity filtering only with an explicit threshold

    Code sources (codesearchnet, stack_smol, stack_v1, jupyter, conala)
    bypass the structural prose checks because code does not always end
    in terminal punctuation and may have legitimate repeated lines
    (boilerplate imports, standard patterns). They also bypass prose KenLM,
    which is not meaningful for code, math templates, or symbol-heavy data.

    Args:
        input_path: Input JSONL file.
        output_path: Output JSONL file.
        kenlm_model: Loaded KenLM model, or None to skip KenLM measurement.
        perplexity_threshold: Explicit maximum perplexity, or None to report only.
        split: "train" or "val" — used for log labels only.
        perplexity_sample_size: Per-source bounded sample used for quantiles.

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
    perplexity_audit = (
        PerplexityAudit(perplexity_sample_size)
        if kenlm_model is not None
        else None
    )

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

                # KenLM is an English-prose distribution signal, not a
                # universal quality boundary. Always measure eligible prose;
                # remove only when the caller supplied an explicit threshold.
                if not skip_prose and kenlm_model is not None:
                    try:
                        ppl = kenlm_model.perplexity(text[:2000])
                        perplexity_audit.observe(
                            str(record.get("source", "unknown")),
                            text,
                            ppl,
                        )
                        if (
                            perplexity_threshold is not None
                            and ppl > perplexity_threshold
                        ):
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

    stats["perplexity_audit"] = (
        perplexity_audit.report(
            policy=(
                "explicit_threshold"
                if perplexity_threshold is not None
                else "report_only"
            ),
            threshold=perplexity_threshold,
        )
        if perplexity_audit is not None
        else {
            "policy": "disabled",
            "threshold": None,
            "sample_selection": None,
            "sample_size_per_source": 0,
            "documents_scored": 0,
            "by_source": {},
        }
    )
    return stats


def _load_kenlm(
    kenlm_model_path: Path | None,
    sentencepiece_model_path: Path | None,
):
    """Load a matched CCNet model pair, or return None when disabled."""
    if kenlm_model_path is None:
        if sentencepiece_model_path is not None:
            raise ValueError("SentencePiece model must be disabled with KenLM")
        return None
    if sentencepiece_model_path is None:
        raise ValueError("CCNet SentencePiece model is required with KenLM")
    missing = [
        path
        for path in (kenlm_model_path, sentencepiece_model_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"CCNet model file not found: {missing[0]}. Download the matched "
            "en.arpa.bin and en.sp.model pair, "
            f"or pass --no-perplexity to explicitly record a no-KenLM run."
        )
    scorer = CCNetPerplexityScorer(
        kenlm_model_path,
        sentencepiece_model_path,
    )
    log.info(f"Loaded CCNet KenLM model from {kenlm_model_path}")
    log.info(f"Loaded CCNet SentencePiece model from {sentencepiece_model_path}")
    return scorer


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
    audit = stats["perplexity_audit"]
    log.info(f"  KenLM policy:             {audit['policy']:>10}")
    log.info(f"  KenLM documents scored:   {audit['documents_scored']:>10,}")


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
        "--kenlm-sentencepiece-model",
        type=Path,
        default=DATA_DIR / "models" / "en.sp.model",
        help="Path to the matching CCNet SentencePiece model",
    )
    parser.add_argument(
        "--perplexity-threshold",
        type=float,
        default=None,
        help="Explicit maximum perplexity; omit for report-only KenLM scoring",
    )
    parser.add_argument(
        "--perplexity-sample-size",
        type=int,
        default=10_000,
        help="Deterministic scored-document sample retained per source for quantiles",
    )
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip KenLM measurement and filtering",
    )
    args = parser.parse_args()

    if args.perplexity_threshold is not None and args.perplexity_threshold <= 0:
        parser.error("--perplexity-threshold must be positive")
    if args.perplexity_sample_size < 1:
        parser.error("--perplexity-sample-size must be positive")
    if args.no_perplexity and args.perplexity_threshold is not None:
        parser.error(
            "--no-perplexity cannot be combined with --perplexity-threshold"
        )

    run_curated_dir = curated_dir(args.size)
    run_validated_dir = validated_dir(args.size)
    args.train = args.train or (run_curated_dir / "train.jsonl")
    args.val = args.val or (run_curated_dir / "val.jsonl")
    args.train_output = args.train_output or (run_validated_dir / "train.jsonl")
    args.val_output = args.val_output or (run_validated_dir / "val.jsonl")

    kenlm_path = None if args.no_perplexity else args.kenlm_model
    sentencepiece_path = (
        None if args.no_perplexity else args.kenlm_sentencepiece_model
    )

    log.info(f"=== SLM Data Validation ===")
    log.info(f"Train input:  {args.train}")
    log.info(f"Val input:    {args.val}")
    log.info(f"Train output: {args.train_output}")
    log.info(f"Val output:   {args.val_output}")
    log.info(f"KenLM:        {kenlm_path or 'disabled'}")
    log.info(f"SentencePiece:{sentencepiece_path or 'disabled'}")

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

    kenlm_model = _load_kenlm(kenlm_path, sentencepiece_path)

    # No threshold is inferred from the corpus. Without an explicit threshold,
    # KenLM remains report-only and cannot silently force percentile attrition.
    perplexity_threshold = args.perplexity_threshold

    input_signature = stable_digest(
        {
            "train": file_snapshot([args.train], root=args.train.parent),
            "val": file_snapshot([args.val], root=args.val.parent),
        }
    )
    validation_contract = {
        "implementation_sha256": code_fingerprint(
            validate_manual_split,
            CCNetPerplexityScorer,
            normalize_ccnet_text,
        ),
        "prose_heuristic_skip_sources": PROSE_HEURISTIC_SKIP_SOURCES,
        "perplexity_enabled": kenlm_model is not None,
        "perplexity_policy": (
            "disabled"
            if kenlm_model is None
            else (
                "explicit_threshold"
                if perplexity_threshold is not None
                else "report_only"
            )
        ),
        "perplexity_threshold": perplexity_threshold,
        "perplexity_sample_size": args.perplexity_sample_size,
        "kenlm_models": (
            {
                "language_model": file_snapshot(
                    [kenlm_path], root=kenlm_path.parent
                )[0],
                "sentencepiece_model": file_snapshot(
                    [sentencepiece_path], root=sentencepiece_path.parent
                )[0],
            }
            if kenlm_path is not None and sentencepiece_path is not None
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
        perplexity_sample_size=args.perplexity_sample_size,
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
            perplexity_sample_size=args.perplexity_sample_size,
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
        "perplexity_policy": (
            "disabled"
            if kenlm_model is None
            else (
                "explicit_threshold"
                if perplexity_threshold is not None
                else "report_only"
            )
        ),
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
