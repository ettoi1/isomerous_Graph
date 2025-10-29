"""Structural encodings for graph transformers."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..graph_build.metrics import SparseAdj


def edge_bias_from_weight(edge_weight: torch.Tensor) -> torch.Tensor:
    if edge_weight.ndim != 1:
        raise ValueError("edge_weight must be 1D")
    return torch.log1p(torch.relu(edge_weight))


def centrality_encoding(adj: SparseAdj) -> torch.Tensor:
    weights = adj.edge_weight
    index = adj.edge_index
    num_nodes = adj.num_nodes
    degree = torch.zeros(num_nodes, device=weights.device)
    degree.index_add_(0, index[0], weights.abs())
    return degree.unsqueeze(-1)


def pair_distance_encoding(adj: SparseAdj, max_dist: int = 5) -> Dict[str, torch.Tensor]:
    num_nodes = adj.num_nodes
    dense = adj.to_dense()
    graph = (dense > 0).float()
    dist = torch.full((num_nodes, num_nodes), float("inf"), device=dense.device)
    dist.fill_diagonal_(0)
    frontier = graph.clone()
    for step in range(1, max_dist + 1):
        new_reachable = (frontier > 0) & (dist > step)
        dist[new_reachable] = step
        frontier = frontier @ graph
    dist[torch.isinf(dist)] = max_dist + 1
    return {"shortest_path": dist}


__all__ = ["edge_bias_from_weight", "centrality_encoding", "pair_distance_encoding"]
