from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
BASE_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
BASE_EXPORTS_DIR = Path(
    os.environ.get("EXPORTS_DIR", str(BASE_RESULTS_DIR / "exports"))
)


def data_run_dir(size: str) -> Path:
    return BASE_DATA_DIR / "runs" / size


def raw_dir(size: str) -> Path:
    return data_run_dir(size) / "raw"


def filtered_dir(size: str) -> Path:
    return data_run_dir(size) / "filtered"


def dedup_scratch_dir(size: str) -> Path:
    return data_run_dir(size) / "dedup_scratch"


def curated_dir(size: str) -> Path:
    return data_run_dir(size) / "curated"


def validated_dir(size: str) -> Path:
    return data_run_dir(size) / "validated"


def tokenizer_dir(size: str) -> Path:
    return data_run_dir(size) / "tokenizer"


def tokenized_dir(size: str) -> Path:
    return data_run_dir(size) / "tokenized"


def metadata_dir(size: str) -> Path:
    return data_run_dir(size) / "metadata"


def sft_instruct_data_dir(size: str) -> Path:
    return data_run_dir(size) / "sft_instruct"


def sft_chat_data_dir(size: str) -> Path:
    # Backwards-compatible alias. New code should use sft_instruct_data_dir().
    return sft_instruct_data_dir(size)


def sft_code_data_dir(size: str) -> Path:
    return data_run_dir(size) / "sft_code"


def code_completion_data_dir(size: str) -> Path:
    return data_run_dir(size) / "code_completion"


def dpo_chat_data_dir(size: str) -> Path:
    return data_run_dir(size) / "dpo_chat"


def dpo_data_dir(size: str) -> Path:
    # Backwards-compatible alias. New code should use dpo_chat_data_dir().
    return dpo_chat_data_dir(size)


def dpo_code_data_dir(size: str) -> Path:
    return data_run_dir(size) / "dpo_code"


def results_run_dir(size: str) -> Path:
    return BASE_RESULTS_DIR / "runs" / size


def pretrain_dir(size: str) -> Path:
    return results_run_dir(size) / "pretrain"


def sft_instruct_dir(size: str) -> Path:
    return results_run_dir(size) / "sft_instruct"


def sft_chat_dir(size: str) -> Path:
    # Backwards-compatible alias. New code should use sft_instruct_dir().
    return sft_instruct_dir(size)


def sft_code_dir(size: str) -> Path:
    return results_run_dir(size) / "sft_code"


def dpo_chat_dir(size: str) -> Path:
    return results_run_dir(size) / "dpo_chat"


def dpo_dir(size: str) -> Path:
    # Backwards-compatible alias. New code should use dpo_chat_dir().
    return dpo_chat_dir(size)


def dpo_code_dir(size: str) -> Path:
    return results_run_dir(size) / "dpo_code"


def eval_dir(size: str) -> Path:
    return results_run_dir(size) / "eval"


def export_dir(size: str) -> Path:
    return BASE_EXPORTS_DIR / size
