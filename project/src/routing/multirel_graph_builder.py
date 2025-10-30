"""Data structure for multi-relation graphs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class MultiRelGraphBatch:
    node_feat: torch.Tensor
    edge_index_list: List[torch.Tensor]
    edge_attr_list: List[torch.Tensor]
    alpha: torch.Tensor


def build_multirel_graph_batch(node_feat: torch.Tensor, routed: dict) -> MultiRelGraphBatch:
    return MultiRelGraphBatch(
        node_feat=node_feat,
        edge_index_list=routed["edge_index_list"],
        edge_attr_list=routed["edge_attr_list"],
        alpha=routed["alpha"],
    )
