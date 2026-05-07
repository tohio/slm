"""
Shared helpers for data-pipeline integration tests.
"""

import json
import os
from pathlib import Path

import pytest


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def pipeline_path(*parts: str) -> Path:
    """Resolve a path under DATA_DIR for pipeline output tests."""
    return data_dir().joinpath(*parts)


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
