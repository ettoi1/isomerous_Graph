"""Routing related losses."""
from __future__ import annotations

import torch


def routing_sparsity_loss(alpha: torch.Tensor) -> torch.Tensor:
    return 1.0 - alpha.max(dim=-1).values.mean()


def expert_balance_loss(alpha: torch.Tensor) -> torch.Tensor:
    usage = alpha.mean(dim=0)
    return usage.var()
