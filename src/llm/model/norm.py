"""Normalization layers: LayerNorm and RMSNorm (used in LLaMA-family)."""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Standard LayerNorm with learnable affine parameters.

    y = gamma * (x - E[x]) / sqrt(Var[x] + eps) + beta
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        out = (x - mean) * torch.rsqrt(var + self.eps)
        out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm (Zhang & Sennrich, 2019).

    Used in LLaMA / Mistral. No mean centering, only scale normalization.
    y = x / sqrt(mean(x^2) + eps) * weight

    Roughly 10-50% faster than LayerNorm and empirically similar quality.
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x * rms * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out