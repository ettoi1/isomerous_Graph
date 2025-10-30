"""Graph readout modules."""
from __future__ import annotations

import torch
from torch import nn


class ReadoutHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, H: torch.Tensor, z_comm: torch.Tensor):
        z_graph = H.mean(dim=1)
        concat = torch.cat([z_graph, z_comm], dim=-1)
        logits = self.mlp(concat)
        return logits, z_graph
