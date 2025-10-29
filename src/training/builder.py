"""High level assembly of the training components."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from ..dataio.datasets import BrainGraphSample
from ..graph_build.moe_edge_gating import EdgeMoEGater, fuse_views_with_alpha
from ..graph_build.multiview_builder import build_multiview_graph
from ..models.encodings import centrality_encoding, edge_bias_from_weight, pair_distance_encoding
from ..models.graph_transformer import GraphTransformer
from ..models.classifier import GraphClassifier


def prepare_batch(batch: List[BrainGraphSample]) -> Dict[str, torch.Tensor]:
    sample = batch[0]
    timeseries = sample["timeseries"]
    if isinstance(timeseries, torch.Tensor):
        return {"timeseries": timeseries}
    return {"timeseries": torch.from_numpy(timeseries).float()}


def forward_pass(batch_dict: Dict[str, torch.Tensor], modules: Dict[str, object], configs: Dict[str, object]) -> Dict[str, object]:
    sample = {"subject_id": "tmp", "timeseries": batch_dict["timeseries"]}
    graph = build_multiview_graph(sample, configs["metrics"], configs["sparsify"], configs.get("norm"))
    gater: EdgeMoEGater = modules["gater"]
    backbone: GraphTransformer = modules["backbone"]
    classifier: GraphClassifier = modules["classifier"]

    fused_edge, alpha = gater(graph["edge_feats"])
    fused_adj = fuse_views_with_alpha(graph["views"], alpha)

    encodings = {
        "edge_bias": edge_bias_from_weight(fused_edge),
        "centrality": centrality_encoding(fused_adj),
        "pair_enc": pair_distance_encoding(fused_adj),
    }

    node_feats = graph["node_feats"] or torch.ones((fused_adj.num_nodes, backbone.layers[0].self_attn.embed_dim))
    logits = classifier(node_feats, fused_adj, **encodings)

    return {"logits": logits, "alpha": alpha, "A_star": fused_adj, "node_emb": logits}


__all__ = ["prepare_batch", "forward_pass"]
