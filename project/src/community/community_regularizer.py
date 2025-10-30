"""Community regularisation losses."""
from __future__ import annotations

import torch


def community_consistency_loss(H: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Encourage nodes to stay close to their soft community centres."""

    weights = S / (S.sum(dim=1, keepdim=True) + 1e-6)
    centres = torch.einsum("bnm,bnd->bmd", weights, H)
    diff = H.unsqueeze(2) - centres.unsqueeze(1)
    sq_dist = (diff ** 2).sum(dim=-1)
    loss = (S * sq_dist).sum(dim=(1, 2)) / (S.sum(dim=(1, 2)) + 1e-6)
    return loss.mean()
