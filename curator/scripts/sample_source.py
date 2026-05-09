#!/usr/bin/env python3
"""
Print actual JSONL samples from one source and one pipeline stage.

Purpose:
    Human review of what the model is actually being trained on.

Examples:
    python curator/scripts/sample_source.py --stage raw --source wikipedia --limit 10
    python curator/scripts/sample_source.py --stage filtered --source wikipedia --limit 10
    python curator/scripts/sample_source.py --stage deduped --source wikipedia --limit 10
    python curator/scripts/sample_source.py --stage curated --source wikipedia --limit 10
    python curator/scripts/sample_source.py --stage validated --source wikipedia --limit 10

Notes:
    - This script intentionally does not score or summarize quality.
    - It prints actual text so we can inspect whether the source is useful.
    - DATA_DIR defaults to ./data and can be overridden with --data-dir.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator


VALID_STAGES = {"raw", "filtered", "deduped", "curated", "validated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print readable samples from a source/stage JSONL dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory. Default: data",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=sorted(VALID_STAGES),
        help="Pipeline stage to sample from.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source name, e.g. wikipedia, fineweb, stackexchange, codesearchnet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples to print. Default: 10",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum text characters to print per sample. Default: 2000",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample records instead of printing first matching records.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed when --random is used. Default: 13",
    )
    return parser.parse_args()


def stage_paths(data_dir: Path, stage: str, source: str) -> list[Path]:
    """
    Resolve JSONL files for a source/stage.

    Expected layouts:
        raw/<source>/*.jsonl
        filtered/<source>/*.jsonl
        filtered/<source>_deduped/*.jsonl
        curated/train.jsonl, curated/val.jsonl
        validated/train.jsonl, validated/val.jsonl
    """
    if stage == "raw":
        base = data_dir / "raw" / source
        return sorted(base.glob("*.jsonl"))

    if stage == "filtered":
        base = data_dir / "filtered" / source
        return sorted(base.glob("*.jsonl"))

    if stage == "deduped":
        base = data_dir / "filtered" / f"{source}_deduped"
        return sorted(base.glob("*.jsonl"))

    if stage == "curated":
        base = data_dir / "curated"
        return [p for p in [base / "train.jsonl", base / "val.jsonl"] if p.is_file()]

    if stage == "validated":
        base = data_dir / "validated"
        return [p for p in [base / "train.jsonl", base / "val.jsonl"] if p.is_file()]

    raise ValueError(f"Unknown stage: {stage}")


def read_jsonl(paths: Iterable[Path]) -> Iterator[tuple[Path, int, dict]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, line_no, json.loads(line)
                except json.JSONDecodeError:
                    yield path, line_no, {
                        "source": "<json_decode_error>",
                        "text": line,
                    }


def source_matches(record: dict, source: str, stage: str) -> bool:
    """
    For per-source directories, most rows should already belong to source.
    For curated/validated train.jsonl, filter by record['source'].
    """
    if stage in {"raw", "filtered", "deduped"}:
        return record.get("source", source) == source

    return record.get("source") == source


def collect_samples(
    paths: list[Path],
    source: str,
    stage: str,
    limit: int,
    random_sample: bool,
    seed: int,
) -> list[tuple[Path, int, dict]]:
    rows: list[tuple[Path, int, dict]] = []

    if not random_sample:
        for path, line_no, record in read_jsonl(paths):
            if source_matches(record, source, stage):
                rows.append((path, line_no, record))
                if len(rows) >= limit:
                    break
        return rows

    # Reservoir sample so we do not need to load huge files into memory.
    rng = random.Random(seed)
    seen = 0

    for path, line_no, record in read_jsonl(paths):
        if not source_matches(record, source, stage):
            continue

        seen += 1
        item = (path, line_no, record)

        if len(rows) < limit:
            rows.append(item)
        else:
            j = rng.randint(1, seen)
            if j <= limit:
                rows[j - 1] = item

    return rows


def compact_text(text: str, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [truncated] ..."


def print_sample(
    index: int,
    path: Path,
    line_no: int,
    record: dict,
    data_dir: Path,
    max_chars: int,
) -> None:
    text = record.get("text")
    if text is None:
        text = record.get("content")
    if text is None:
        text = json.dumps(record, ensure_ascii=False, indent=2)

    source = record.get("source", "<missing>")
    record_id = record.get("id", record.get("_id", "<missing>"))

    rel_path = path
    try:
        rel_path = path.relative_to(data_dir)
    except ValueError:
        pass

    print("=" * 100)
    print(f"SAMPLE {index}")
    print(f"source: {source}")
    print(f"id: {record_id}")
    print(f"path: {rel_path}")
    print(f"line: {line_no}")
    print(f"chars: {len(text):,}")
    print("-" * 100)
    print(compact_text(text, max_chars))
    print()


def main() -> None:
    args = parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.max_chars <= 0:
        raise SystemExit("--max-chars must be > 0")

    paths = stage_paths(args.data_dir, args.stage, args.source)

    if not paths:
        raise SystemExit(
            f"No JSONL files found for stage={args.stage!r}, "
            f"source={args.source!r}, data_dir={str(args.data_dir)!r}"
        )

    samples = collect_samples(
        paths=paths,
        source=args.source,
        stage=args.stage,
        limit=args.limit,
        random_sample=args.random,
        seed=args.seed,
    )

    if not samples:
        raise SystemExit(
            f"No records found for source={args.source!r} in stage={args.stage!r}. "
            f"Checked {len(paths)} file(s)."
        )

    print(
        f"Showing {len(samples)} sample(s) "
        f"from source={args.source!r}, stage={args.stage!r}"
    )
    print(f"data_dir: {args.data_dir}")
    print(f"files: {len(paths)}")
    print()

    for i, (path, line_no, record) in enumerate(samples, start=1):
        print_sample(i, path, line_no, record, args.data_dir, args.max_chars)


if __name__ == "__main__":
    main()
