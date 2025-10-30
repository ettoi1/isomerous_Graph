"""Simplified multi-relation graph transformer."""
from __future__ import annotations

from typing import List

import torch
from torch import nn

from project.src.routing.multirel_graph_builder import MultiRelGraphBatch


class MultiRelGraphTransformer(nn.Module):
    def __init__(self, d_node_init: int, d_model: int, n_layers: int, edge_attr_dims: List[int]):
        super().__init__()
        self.input_proj = nn.Linear(d_node_init, d_model)
        self.layers = nn.ModuleList()
        self.edge_attr_dims = edge_attr_dims
        self.n_relations = len(edge_attr_dims)
        for _ in range(n_layers):
            node_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in edge_attr_dims])
            edge_linears = nn.ModuleList([nn.Linear(dim, d_model) if dim > 0 else nn.Identity() for dim in edge_attr_dims])
            layer_norm = nn.LayerNorm(d_model)
            self.layers.append(nn.ModuleDict({
                "node": node_linears,
                "edge": edge_linears,
                "norm": layer_norm,
            }))
        self.activation = nn.ReLU()

    def forward(self, graph_batch: MultiRelGraphBatch) -> torch.Tensor:
        H = self.input_proj(graph_batch.node_feat)
        for layer in self.layers:
            node_linears: nn.ModuleList = layer["node"]
            edge_linears: nn.ModuleList = layer["edge"]
            norm: nn.LayerNorm = layer["norm"]
            residual = H
            agg = torch.zeros_like(H)
            B, N, _ = H.shape
            for relation_id in range(self.n_relations):
                edge_index = graph_batch.edge_index_list[relation_id]
                edge_attr = graph_batch.edge_attr_list[relation_id]
                if edge_index.numel() == 0:
                    continue
                src_idx = edge_index[0]
                dst_idx = edge_index[1]
                src_feat = torch.index_select(H, 1, src_idx)
                node_msg = node_linears[relation_id](src_feat)
                if edge_attr.numel() > 0:
                    edge_msg = edge_linears[relation_id](edge_attr).unsqueeze(0)
                    if edge_msg.shape[0] == 1 and B > 1:
                        edge_msg = edge_msg.expand(B, -1, -1)
                    node_msg = node_msg + edge_msg
                for b in range(B):
                    agg[b].index_add_(0, dst_idx, node_msg[b])
            H = norm(residual + self.activation(agg))
        return H
