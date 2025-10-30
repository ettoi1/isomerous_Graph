"""Community assignment module."""
from __future__ import annotations

import torch
from torch import nn


class CommunityAssigner(nn.Module):
    def __init__(self, d_model: int, num_communities: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_communities, d_model))

    def forward(self, H: torch.Tensor):
        sim = torch.einsum("bnd,md->bnm", H, self.prototypes)
        S = torch.softmax(sim, dim=-1)
        return S, self.prototypes
