"""Community-aware regularisers."""

from __future__ import annotations

from typing import Dict

import torch


def community_mask(atlas_meta: Dict[str, torch.Tensor]) -> torch.Tensor:
    return atlas_meta["community_mask"].bool()


def intra_community_consistency(h: torch.Tensor, mask: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    diffs = h.unsqueeze(1) - h.unsqueeze(0)
    squared = (diffs ** 2).sum(dim=-1)
    values = squared[mask]
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"Unknown reduction {reduction}")


def laplacian_reg(h: torch.Tensor, L_comm: torch.Tensor) -> torch.Tensor:
    return (h.T @ L_comm @ h).trace()


__all__ = ["community_mask", "intra_community_consistency", "laplacian_reg"]
