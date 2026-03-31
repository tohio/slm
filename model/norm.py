"""
model/norm.py
-------------
RMSNorm — Root Mean Square Layer Normalization.

RMSNorm simplifies LayerNorm by removing the mean subtraction step,
only normalizing by the root mean square of the activations. This is
faster and empirically matches LayerNorm quality in transformer LLMs.

Used by: LLaMA, Mistral, Qwen, Gemma, and most modern transformer models.

Reference:
    Zhang & Sennrich (2019) — "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes the input tensor by its RMS value and applies a learned
    per-dimension scale (weight). No bias term — consistent with the
    no-bias design of the SLM architecture.

    Formula:
        RMSNorm(x) = x / RMS(x) * weight
        where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        hidden_size (int): Dimension of the input tensor (last dimension).
        eps (float): Small constant for numerical stability. Default: 1e-5.

    Shape:
        Input:  (..., hidden_size)
        Output: (..., hidden_size)

    Example::

        norm = RMSNorm(768)
        x = torch.randn(2, 512, 768)
        out = norm(x)  # shape: (2, 512, 768)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability,
        # then cast back to the input dtype
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"