"""Simplified Graph Transformer backbone."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph_build.metrics import SparseAdj


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_bias)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.linear2(F.gelu(self.linear1(x)))))
        return x


class GraphTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        edge_bias: bool = True,
        dist_enc: bool = True,
        centrality_enc: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([GraphTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.edge_bias = edge_bias
        self.dist_enc = dist_enc
        self.centrality_enc = centrality_enc

    def forward(
        self,
        node_feats: torch.Tensor,
        adj: SparseAdj,
        edge_bias: Optional[torch.Tensor] = None,
        pair_enc: Optional[Dict[str, torch.Tensor]] = None,
        centrality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = node_feats
        if self.centrality_enc and centrality is not None:
            h = h + centrality
        attn_bias = None
        if self.edge_bias and edge_bias is not None:
            num_nodes = adj.num_nodes
            attn_bias = torch.zeros((num_nodes, num_nodes), device=edge_bias.device)
            attn_bias[adj.edge_index[0], adj.edge_index[1]] = edge_bias
            attn_bias = attn_bias.unsqueeze(0)
        for layer in self.layers:
            h = layer(h.unsqueeze(0), attn_bias).squeeze(0)
        return h


__all__ = ["GraphTransformer"]
