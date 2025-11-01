"""Edge gating network for mixture-of-experts routing."""
from __future__ import annotations

import torch
from torch import nn


class EdgeGatingNetwork(nn.Module):
    def __init__(self, in_dim: int, n_experts: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, n_experts),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(edge_features)
        return torch.softmax(logits, dim=-1)
