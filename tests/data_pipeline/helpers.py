"""
Shared helpers for data-pipeline integration tests.
"""

import json
import os
from pathlib import Path

import pytest


def data_dir() -> Path:
    return Path(
        os.environ.get(
            "PIPELINE_TEST_DATA_DIR",
            os.environ.get("DATA_DIR", "data"),
        )
    )


def pipeline_size() -> str:
    return os.environ.get("PIPELINE_TEST_SIZE", os.environ.get("SIZE", "mini"))


def run_data_dir() -> Path:
    return data_dir() / "runs" / pipeline_size()


def pipeline_path(*parts: str) -> Path:
    """
    Resolve pipeline artifact paths.

    Most pipeline artifacts are size-scoped under:
        data/runs/<size>/<stage>/...

    The tokenizer is size-scoped with every other reusable artifact:
        data/runs/<size>/tokenizer/...
    """
    if not parts:
        return data_dir()

    stage, *rest = parts

    # Backward-compatible aliases used by older tests.
    if stage == "sft" and rest:
        variant, *tail = rest
        if variant == "chat":
            return run_data_dir().joinpath("sft_instruct", *tail)
        if variant == "code":
            return run_data_dir().joinpath("sft_code", *tail)

    return run_data_dir().joinpath(stage, *rest)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def requires_stage(stage: str):
    """
    Mark tests that require a pipeline stage's outputs.

    The tests still assert the concrete files they need; this marker is used
    for readability and optional pytest marker selection.
    """
    return pytest.mark.stage(stage)
