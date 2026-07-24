"""Pinned Hugging Face dataset loading for curation sources."""

from __future__ import annotations

from functools import lru_cache

from datasets import load_dataset as _load_dataset
from huggingface_hub import HfApi


@lru_cache(maxsize=None)
def resolve_dataset_revision(dataset_name: str) -> str:
    """Resolve a dataset repository to an immutable commit SHA."""
    info = HfApi().dataset_info(dataset_name)
    if not info.sha:
        raise RuntimeError(
            f"Hugging Face did not return a revision SHA for {dataset_name}"
        )
    return str(info.sha)


def load_dataset(path: str, *args, revision: str | None = None, **kwargs):
    """Call datasets.load_dataset at an immutable repository revision."""
    pinned_revision = revision or resolve_dataset_revision(path)
    return _load_dataset(
        path,
        *args,
        revision=pinned_revision,
        **kwargs,
    )
