"""Placeholder encoding utilities."""
from __future__ import annotations

import torch
from torch import nn


def positional_encoding(num_positions: int, dim: int) -> torch.Tensor:
    position = torch.arange(0, num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(num_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
