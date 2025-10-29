"""Graph-level readout and classifier heads."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .graph_transformer import GraphTransformer
from ..graph_build.metrics import SparseAdj


def graph_readout(node_emb: torch.Tensor, mode: Literal["mean", "max", "cls_token"] = "mean") -> torch.Tensor:
    if mode == "mean":
        return node_emb.mean(dim=0)
    if mode == "max":
        return node_emb.max(dim=0).values
    if mode == "cls_token":
        return node_emb[0]
    raise ValueError(f"Unknown readout mode: {mode}")


class GraphClassifier(nn.Module):
    def __init__(self, backbone: GraphTransformer, readout: str, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.readout_mode = readout
        self.head = nn.Linear(backbone.layers[0].self_attn.embed_dim, num_classes)

    def forward(self, node_feats: torch.Tensor, adj: SparseAdj, **encodings) -> torch.Tensor:
        h = self.backbone(node_feats, adj, **encodings)
        graph_repr = graph_readout(h, mode=self.readout_mode)
        return self.head(graph_repr)


__all__ = ["graph_readout", "GraphClassifier"]
