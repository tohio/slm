"""
tests/conftest.py
-----------------
Shared fixtures and configuration for all tests.

Two test modes:
    Pipeline tests  — validate real outputs in DATA_DIR after each make stage.
                      Require DATA_DIR to be set and the relevant stage to have run.
    Unit tests      — synthetic data, no DATA_DIR required.
                      Run on a fresh clone with no pipeline outputs.

DATA_DIR is resolved from the environment variable set by setup.sh.
If not set, tests that require real pipeline outputs are skipped.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Fasttext mock ──────────────────────────────────────────────────────────────
# Patch fasttext before any curator imports so tests never need lid.176.ftz.
# The mock always predicts English with high confidence.

_ft_mock = MagicMock()
_ft_mock.predict.return_value = (["__label__en"], [0.99])
sys.modules.setdefault("fasttext", MagicMock(
    load_model=MagicMock(return_value=_ft_mock),
    FastText=MagicMock(eprint=MagicMock()),
))


# ── DATA_DIR resolution ────────────────────────────────────────────────────────

def get_data_dir() -> Path | None:
    """Resolve DATA_DIR from environment. Returns None if not set."""
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        return None
    p = Path(data_dir)
    return p if p.exists() else None


DATA_DIR = get_data_dir()


def requires_stage(stage: str):
    """Skip marker for tests that need real pipeline outputs."""
    return pytest.mark.skipif(
        DATA_DIR is None,
        reason=f"DATA_DIR not set — run 'make {stage}' first",
    )


# ── Pipeline path helpers ──────────────────────────────────────────────────────

def pipeline_path(*parts) -> Path:
    """Return a path under DATA_DIR. Raises if DATA_DIR not set."""
    if DATA_DIR is None:
        pytest.skip("DATA_DIR not set")
    return DATA_DIR.joinpath(*parts)


# ── JSONL helpers ──────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ── Synthetic document factories ───────────────────────────────────────────────
#
# Factories produce documents matching the per-source output schema. Source
# tags use the specific loader names (codesearchnet, stack_smol, etc.) —
# the generic "code" tag is no longer valid.

GOOD_TEXT = (
    "This is a well-written encyclopedia article about a subject. "
    "It contains multiple sentences with good English prose. "
    "The article covers the topic in reasonable depth and contains "
    "enough content to pass all quality filters applied during curation. " * 15
)

GOOD_CODE = (
    "# Parse and validate the input configuration dictionary.\n"
    "def parse_config(config: dict) -> dict:\n"
    "    if not isinstance(config, dict):\n"
    "        raise TypeError('config must be a dict')\n"
    "    required = ['model', 'data_dir', 'output_dir']\n"
    "    for key in required:\n"
    "        if key not in config:\n"
    "            raise ValueError(f'Missing required key: {key}')\n"
    "    return config\n" * 8
)


# ── Non-code source factories ──────────────────────────────────────────────────

def make_common_crawl_doc(text=None, url="https://example.com/article", crawl="CC-MAIN-2024-10"):
    return {
        "text": text or GOOD_TEXT,
        "source": "common_crawl",
        "url": url,
        "crawl": crawl,
        "language": "en",
    }


# Backwards-compat alias — many existing tests still call make_cc_doc.
make_cc_doc = make_common_crawl_doc


def make_fineweb_doc(text=None, url="https://example.com/article", dump="CC-MAIN-2024-10"):
    return {
        "text": text or GOOD_TEXT,
        "source": "fineweb",
        "url": url,
        "dump": dump,
        "language": "en",
    }


def make_wikipedia_doc(text=None, title="Test Article", url="https://en.wikipedia.org/wiki/Test"):
    return {
        "text": text or GOOD_TEXT,
        "source": "wikipedia",
        "title": title,
        "url": url,
    }


def make_pg19_doc(text=None, title="Test Book", publication_date="1850", url=""):
    return {
        "text": text or GOOD_TEXT,
        "source": "pg19",
        "title": title,
        "publication_date": publication_date,
        "url": url,
    }


def make_pes2o_doc(text=None, paper_id="test_001", subset="s2orc"):
    return {
        "text": text or GOOD_TEXT,
        "source": "pes2o",
        "paper_id": paper_id,
        "subset": subset,
    }


def make_open_web_math_doc(text=None, url="https://example.com/math", date="2024-01-01"):
    return {
        "text": text or GOOD_TEXT,
        "source": "open_web_math",
        "url": url,
        "date": date,
        "subdomain": "",
    }


def make_stackexchange_doc(text=None, site="stackoverflow", question_id="12345"):
    return {
        "text": text or GOOD_TEXT,
        "source": "stackexchange",
        "site": site,
        "question_id": question_id,
    }


# ── Code source factories ──────────────────────────────────────────────────────

def make_codesearchnet_doc(text=None, language="python", repo="test/repo", path="test.py"):
    return {
        "text": text or GOOD_CODE,
        "source": "codesearchnet",
        "language": language,
        "repo": repo,
        "path": path,
    }


# Backwards-compat alias — the old make_code_doc produced source="code" which
# is no longer valid. Point existing callers at codesearchnet as the closest
# semantic match (docstring + code pattern).
make_code_doc = make_codesearchnet_doc


def make_stack_smol_doc(text=None, language="python", repo="test/repo", path="test.py"):
    return {
        "text": text or GOOD_CODE,
        "source": "stack_smol",
        "language": language,
        "repo": repo,
        "path": path,
    }


def make_stack_v1_doc(text=None, language="python", repo="test/repo", path="test.py"):
    return {
        "text": text or GOOD_CODE,
        "source": "stack_v1",
        "language": language,
        "repo": repo,
        "path": path,
    }


def make_jupyter_doc(text=None, repo="test/repo"):
    return {
        "text": text or GOOD_CODE,
        "source": "jupyter",
        "repo": repo,
    }


def make_conala_doc(text=None, question_id="12345"):
    return {
        "text": text or GOOD_CODE,
        "source": "conala",
        "language": "python",
        "question_id": question_id,
    }


# ── Mini model config ──────────────────────────────────────────────────────────

def make_mini_config():
    """Return SLMConfig matching gpt_mini.yaml for unit tests."""
    from model.config import SLMConfig
    return SLMConfig(
        vocab_size=32000,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )