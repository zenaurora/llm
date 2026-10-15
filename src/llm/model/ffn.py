"""Feed-forward network (FFN) blocks for Transformer layers.

The two main variants used in modern LLMs:
- Standard FFN:        x -> Linear -> activation -> Linear -> x
                       (GPT-2 / BERT, activation usually GELU)
- SwiGLU FFN (LLaMA):  x -> (silu(W1 x) * W3 x) -> W2 -> x

SwiGLU uses a gated linear unit with SiLU (swish) activation and is the
de-facto standard in LLaMA / Mistral / Qwen / Gemma.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """Standard 2-layer FFN with configurable activation.

    hidden_dim defaults to 4 * dim (the BERT/GPT-2 ratio).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.fc_in = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc_out = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        h = _ACTIVATE[self.activation_name](h)
        h = self.dropout(h)
        return self.fc_out(h)


class SwiGLU(nn.Module):
    """SwiGLU FFN as used in LLaMA / Mistral.

    Three projections: gate (w1), up (w3), down (w2).
        y = w2( silu(w1(x)) * w3(x) )

    Hidden dim is rounded up to a multiple of `multiple_of` (default 256) for
    efficient matmul kernels — this matches the LLaMA reference.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * dim * 4 / 3)
        
        # 向上取整为 multiple的倍数，方便 GPU 计算
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


_ACTIVATE = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "tanh": torch.tanh,
}