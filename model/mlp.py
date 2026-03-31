"""
model/mlp.py
------------
SwiGLU Feed-Forward Network (FFN).

SwiGLU replaces the standard FFN with a gated variant:
    FFN(x) = (xW1 * SiLU(xW3)) @ W2

where W1 is the "value" projection, W3 is the "gate" projection,
and W2 is the output projection. The gate controls information flow
through the network, improving gradient flow and model expressiveness.

The intermediate dimension follows LLaMA's formula:
    intermediate_size = round(8/3 * hidden_size) rounded to nearest 256

This ensures parameter count matches a standard 4x FFN expansion
despite using three projection matrices instead of two.

No bias terms — consistent with the SLM architecture.

References:
    SwiGLU: Shazeer (2020) — https://arxiv.org/abs/2002.05202
    LLaMA: Touvron et al. (2023) — https://arxiv.org/abs/2302.13971
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SLMConfig


class SwiGLUMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Three linear projections, no bias:
        gate_proj (W3): hidden_size → intermediate_size  — gate
        up_proj   (W1): hidden_size → intermediate_size  — value
        down_proj (W2): intermediate_size → hidden_size  — output

    Forward pass:
        gate = SiLU(gate_proj(x))
        value = up_proj(x)
        output = down_proj(gate * value)

    SiLU (Sigmoid Linear Unit) is the smooth activation used in SwiGLU:
        SiLU(x) = x * sigmoid(x)

    Args:
        config (SLMConfig): Model configuration.

    Shape:
        Input:  (batch, seq_len, hidden_size)
        Output: (batch, seq_len, hidden_size)

    Example::

        config = SLMConfig(hidden_size=768, intermediate_size=2048)
        mlp = SwiGLUMLP(config)
        x = torch.randn(2, 512, 768)
        out = mlp(x)  # shape: (2, 512, 768)
    """

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: element-wise product of gated and value projections
        # F.silu(x) = x * sigmoid(x)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )