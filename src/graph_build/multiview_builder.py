"""Multi-view graph construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from .metrics import SparseAdj, compute_correlation, normalize_connectivity, threshold_or_knn
from ..dataio.datasets import BrainGraphSample


@dataclass
class GraphView:
    adjacency: SparseAdj
    metric_name: str


@dataclass
class MultiviewGraph:
    views: List[GraphView]
    edge_features: torch.Tensor
    node_features: Optional[torch.Tensor]
    meta: Dict[str, object]


def build_multiview_graph(
    sample: BrainGraphSample,
    metrics_cfg: List[Dict[str, object]],
    sparsify_cfg: Dict[str, object],
    norm_cfg: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Create a multi-view sparse graph for a single subject."""

    ts = sample["timeseries"]
    if isinstance(ts, torch.Tensor):
        ts_np = ts.cpu().numpy()
    else:
        ts_np = np.load(ts) if not isinstance(ts, np.ndarray) else ts

    views: List[GraphView] = []
    edge_weights: List[torch.Tensor] = []

    for metric in metrics_cfg:
        name = metric["name"]
        dense = compute_correlation(ts_np, method=name, **{k: v for k, v in metric.items() if k != "name"})
        if norm_cfg:
            dense = normalize_connectivity(dense, norm=norm_cfg.get("mode", "fisherz"))
        adj = threshold_or_knn(dense.copy(), mode=sparsify_cfg.get("mode", "topk"), k=sparsify_cfg.get("k"), q=sparsify_cfg.get("q"))
        views.append(GraphView(adjacency=adj, metric_name=name))
        edge_weights.append(adj.edge_weight.unsqueeze(-1))

    # Align the edges across views by concatenating weights along the last dimension.
    # Assumes identical edge ordering which holds for deterministic sparsification.
    stacked_edge_weights = torch.cat(edge_weights, dim=-1)
    node_features = None
    meta = {"subject_id": sample["subject_id"], "site": sample.get("site")}

    return {
        "views": [v.adjacency for v in views],
        "edge_feats": stacked_edge_weights,
        "node_feats": node_features,
        "meta": meta,
    }


__all__ = ["build_multiview_graph", "MultiviewGraph", "GraphView"]
