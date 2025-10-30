"""Utility to construct optimisers."""
from __future__ import annotations

from typing import Iterable

import torch


def build_optimizer(parameters: Iterable[torch.nn.Parameter], lr: float = 1e-3) -> torch.optim.Optimizer:
    """Return a standard Adam optimiser for the provided parameters."""

    return torch.optim.Adam(parameters, lr=lr)
