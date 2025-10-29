"""Attention map aggregation utilities."""

from __future__ import annotations

from typing import Dict

import torch

from ..graph_build.metrics import SparseAdj


def aggregate_attention_to_edges(attn_tensors: torch.Tensor, adj: SparseAdj) -> torch.Tensor:
    edge_index = adj.edge_index
    aggregated = attn_tensors.mean(dim=0)
    values = aggregated[edge_index[0], edge_index[1]]
    return values


def community_level_summary(edge_scores: torch.Tensor, community_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    total = edge_scores.sum()
    intra = edge_scores[community_mask].sum()
    inter = total - intra
    return {"intra": intra, "inter": inter}


__all__ = ["aggregate_attention_to_edges", "community_level_summary"]
