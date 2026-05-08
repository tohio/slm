"""
tests/conftest.py
-------------------------------
Shared test fixtures. Adds a --size pytest option so GPU pipeline tests
can validate any model size (mini, 125m, 350m, 1b) without code changes.

Default is mini. Full-size artifact checks are opt-in by passing
--size=125m, --size=350m, or --size=1b after the corresponding run completes.

Default is mini. Full-size artifact checks are opt-in by passing
--size=125m, --size=350m, or --size=1b after the corresponding run completes.

Usage:
    pytest tests/gpu_pipeline/ --size=mini
    pytest tests/gpu_pipeline/ --size=125m
"""

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--size",
        action="store",
        default="mini",
        help="Model size to validate: mini | 125m | 350m | 1b",
    )


@pytest.fixture(scope="session")
def model_size(request):
    return request.config.getoption("--size")


@pytest.fixture(scope="session")
def results_dir():
    return Path(os.environ.get("RESULTS_DIR", "results"))


@pytest.fixture(scope="session")
def pretrain_model_dir(results_dir, model_size):
    return results_dir / f"slm-{model_size}" / "final"


@pytest.fixture(scope="session")
def chat_sft_model_dir(results_dir, model_size):
    return results_dir / f"slm-{model_size}-chat" / "final"


@pytest.fixture(scope="session")
def code_sft_model_dir(results_dir, model_size):
    return results_dir / f"slm-{model_size}-chat-code" / "final"


@pytest.fixture(scope="session")
def dpo_model_dir(results_dir, model_size):
    return results_dir / f"slm-{model_size}-dpo" / "final"