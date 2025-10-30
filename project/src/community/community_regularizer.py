"""Community regularisation losses."""
from __future__ import annotations

import torch


def community_consistency_loss(H: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Encourage nodes to stay close to their soft community centres."""

    community_mass = S.sum(dim=1, keepdim=True).clamp_min(1e-6)
    weights = S / community_mass
    centres = torch.einsum("bnm,bnd->bmd", weights, H)
    diff = H.unsqueeze(2) - centres.unsqueeze(1)
    sq_dist = diff.pow(2).sum(dim=-1)
    weighted_error = (S * sq_dist).sum(dim=(1, 2))
    normaliser = S.sum(dim=(1, 2)).clamp_min(1e-6)
    return (weighted_error / normaliser).mean()
