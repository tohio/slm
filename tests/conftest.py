"""
tests/conftest.py
-------------------------------
Shared test fixtures. Adds a --size pytest option so GPU pipeline tests
can validate any model size (mini, 125m, 350m, 1b) without code changes.

Default is mini. Full-size artifact checks are opt-in by passing
--size=125m, --size=350m, or --size=1b after the corresponding run completes.

Usage:
    pytest tests/gpu_pipeline/ --size=mini
    pytest tests/gpu_pipeline/ --size=125m
"""

import os
from pathlib import Path

import pytest


def make_mini_config():
    """Return the small deterministic model config used by CPU unit tests."""
    from model.config import SLMConfig

    return SLMConfig(
        vocab_size=32_000,
        hidden_size=384,
        intermediate_size=1_024,
        num_hidden_layers=6,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=1_024,
        rope_theta=500_000.0,
    )


def pytest_addoption(parser):
    parser.addoption(
        "--size",
        action="store",
        default="mini",
        help="Model size to validate: mini | 125m | 350m | 1b",
    )
    parser.addoption(
        "--require-artifacts",
        action="store_true",
        default=False,
        help=(
            "Fail instead of skip when a requested pipeline artifact is "
            "missing. Makefile artifact-test targets enable this."
        ),
    )


@pytest.fixture(scope="session")
def model_size(request):
    return request.config.getoption("--size")


@pytest.fixture(scope="session")
def require_artifacts(request):
    return request.config.getoption("--require-artifacts")


def skip_or_fail_missing_artifact(
    path: Path,
    instruction: str,
    *,
    required: bool,
) -> None:
    """Keep exploratory pytest runs optional but make named gates strict."""
    message = f"Artifact not found at {path} — {instruction}"
    if required:
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="session")
def results_dir():
    return Path(os.environ.get("RESULTS_DIR", "results"))


@pytest.fixture(scope="session")
def pretrain_model_dir(results_dir, model_size):
    return results_dir / "runs" / model_size / "pretrain" / "final"


@pytest.fixture(scope="session")
def chat_sft_model_dir(results_dir, model_size):
    return results_dir / "runs" / model_size / "sft_instruct" / "final"


@pytest.fixture(scope="session")
def code_sft_model_dir(results_dir, model_size):
    return results_dir / "runs" / model_size / "sft_code" / "final"


@pytest.fixture(scope="session")
def dpo_model_dir(results_dir, model_size):
    return results_dir / "runs" / model_size / "dpo_chat" / "final"
