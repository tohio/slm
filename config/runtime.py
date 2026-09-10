"""Shared PyTorch runtime configuration for training entry points."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_inductor_cache_env() -> str:
    """Set Inductor cache environment before importing PyTorch."""
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    cache_dir = os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        str(Path(os.environ.get("DATA_DIR", "data")) / "cache" / "torchinductor"),
    )
    return cache_dir


# This must happen before the first torch import. PyTorch/Inductor may otherwise
# establish its process default under /tmp, after which setdefault() is too late
# to select SLM's persistent DATA_DIR location. Explicit caller settings win.
configure_inductor_cache_env()

import torch


def configure_torch_runtime(log: logging.Logger | None = None) -> None:
    """Set CUDA defaults; SLM attention scopes its mixed-precision training policy."""
    if not torch.cuda.is_available():
        return

    # The cache environment is established at module import, before torch, by
    # supported training entry points. Re-read it here for logging and ensure
    # the selected directory exists.
    cache_dir = configure_inductor_cache_env()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Allows TF32 tensor-core matmuls where PyTorch considers them appropriate.
    # Model parameters and outputs remain in their configured dtypes.
    torch.set_float32_matmul_precision("high")

    # Keep reference/evaluation defaults available. SLM attention itself
    # excludes math for CUDA BF16/FP16 training, including compiled forwards.
    # Do not globally disable math: FP32 diagnostics need independent dispatch.
    cuda_backend = torch.backends.cuda
    for name in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_cudnn_sdp",
        "enable_math_sdp",
    ):
        setter = getattr(cuda_backend, name, None)
        if setter is not None:
            setter(True)

    if log is not None:
        log.info(
            "CUDA runtime: TF32=high; default SDPA dispatch=automatic. "
            "SLM CUDA BF16/FP16 training requires fused SDPA (math disabled "
            "per attention call); CPU/FP32/eval retain normal dispatch. "
            "This reports backend policy, not the kernel actually selected. "
            "Inductor cache=%s (FX graph + AOTAutograd enabled unless overridden).",
            cache_dir,
        )
