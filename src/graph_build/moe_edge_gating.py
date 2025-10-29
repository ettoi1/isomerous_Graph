"""Mixture-of-experts edge gating layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import SparseAdj


class EdgeMoEGater(nn.Module):
    """Mixture-of-experts gating operating on edge features."""

    def __init__(
        self,
        in_edgefeat_dim: int,
        n_experts: int,
        gating: str = "soft",
        temperature: float = 1.0,
        add_context: bool = False,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.temperature = temperature
        self.gating = gating
        self.add_context = add_context
        hidden_dim = max(32, in_edgefeat_dim // 2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edgefeat_dim + (context_dim if add_context else 0), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
        )
        self.expert_projection = nn.Linear(in_edgefeat_dim, n_experts, bias=False)

    def forward(
        self,
        edge_features: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_features.ndim != 2:
            raise ValueError("edge_features must have shape [E, d_e]")

        features = edge_features
        if self.add_context and context is not None and "edge" in context:
            features = torch.cat([features, context["edge"]], dim=-1)

        logits = self.edge_mlp(features) / self.temperature

        if self.gating == "soft":
            alpha = F.softmax(logits, dim=-1)
        elif self.gating == "top1":
            top_idx = logits.argmax(dim=-1, keepdim=True)
            alpha = torch.zeros_like(logits)
            alpha.scatter_(1, top_idx, 1.0)
        elif self.gating == "top2":
            top2 = logits.topk(k=2, dim=-1).indices
            alpha = torch.zeros_like(logits)
            alpha.scatter_(1, top2, 1.0 / 2.0)
        else:
            raise ValueError(f"Unsupported gating mode: {self.gating}")

        expert_scores = self.expert_projection(edge_features)
        fused_edge = (alpha * expert_scores).sum(dim=-1)
        return fused_edge, alpha


def fuse_views_with_alpha(views: list[SparseAdj], alpha: torch.Tensor) -> SparseAdj:
    if not views:
        raise ValueError("No views provided for fusion")
    base = views[0]
    stacked_weights = torch.stack([view.edge_weight for view in views], dim=-1)
    fused_weights = (stacked_weights * alpha.unsqueeze(-1)).sum(dim=-2)
    return SparseAdj(edge_index=base.edge_index, edge_weight=fused_weights, num_nodes=base.num_nodes)


__all__ = ["EdgeMoEGater", "fuse_views_with_alpha"]
