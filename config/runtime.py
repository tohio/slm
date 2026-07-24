"""Shared PyTorch runtime configuration for training entry points."""

from __future__ import annotations

import logging

import torch


def configure_torch_runtime(log: logging.Logger | None = None) -> None:
    """Enable safe CUDA fast paths while retaining dispatcher fallbacks."""
    if not torch.cuda.is_available():
        return

    # Allows TF32 tensor-core matmuls where PyTorch considers them appropriate.
    # Model parameters and outputs remain in their configured dtypes.
    torch.set_float32_matmul_precision("high")

    # Keep all SDPA implementations enabled. The dispatcher selects Flash,
    # memory-efficient, cuDNN, or math attention for each shape/mask. Disabling
    # math globally turns an unsupported fast-kernel shape into a hard failure.
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
            "CUDA runtime: TF32=high; Flash/memory-efficient/cuDNN/math "
            "SDPA dispatch enabled"
        )
