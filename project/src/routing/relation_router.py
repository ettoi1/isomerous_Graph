"""Route edges to expert-specific relation channels."""
from __future__ import annotations

from typing import Dict, List

import torch


def route_edges(edge_index: torch.Tensor, expert_outputs: List[Dict[str, torch.Tensor]], alpha: torch.Tensor, top_k: int = 1) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
    if edge_index.ndim != 2:
        raise ValueError("edge_index must have shape [2, E]")
    num_edges = edge_index.size(1)
    if alpha.size(0) != num_edges:
        raise ValueError("alpha must align with the number of edges")

    assigned = torch.argmax(alpha, dim=-1)
    n_experts = len(expert_outputs)
    edge_index_list: List[torch.Tensor] = []
    edge_attr_list: List[torch.Tensor] = []
    for expert_id in range(n_experts):
        mask = assigned == expert_id
        selected_edges = edge_index[:, mask]
        expert_repr = expert_outputs[expert_id]["edge_repr"][mask]
        edge_index_list.append(selected_edges)
        edge_attr_list.append(expert_repr)
    return {
        "edge_index_list": edge_index_list,
        "edge_attr_list": edge_attr_list,
        "alpha": alpha,
    }
