"""Placeholder for community-guided routing losses."""
from __future__ import annotations

import torch


def routing_consistency_loss(S: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    # TODO: Incorporate community-aware routing constraints.
    return torch.zeros(1, device=alpha.device, dtype=alpha.dtype).sum()
