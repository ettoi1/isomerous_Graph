"""Bundle assembling the full BrainGraph model."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from project.src.community.community_assign import CommunityAssigner
from project.src.community.community_graph import build_community_graph
from project.src.experts.expert_factory import build_experts
from project.src.models.multirel_graph_transformer import MultiRelGraphTransformer
from project.src.models.readout_and_heads import ReadoutHead
from project.src.routing.edge_feature_builder import build_edge_feature_table
from project.src.routing.gating_network import EdgeGatingNetwork
from project.src.routing.multirel_graph_builder import build_multirel_graph_batch
from project.src.routing.relation_router import route_edges


class BrainGraphModel(nn.Module):
    def __init__(self, cfg_model: Dict, cfg_experts: Dict, num_classes: int):
        super().__init__()
        self.experts = nn.ModuleList(build_experts(cfg_experts))
        edge_attr_dims = [expert.out_dim for expert in self.experts]
        self.node_encoder = nn.Linear(1, cfg_model.get("d_node_init", 16))
        self.gate = EdgeGatingNetwork(sum(edge_attr_dims), len(self.experts))
        self.transformer = MultiRelGraphTransformer(
            d_node_init=cfg_model.get("d_node_init", 16),
            d_model=cfg_model.get("d_model", 32),
            n_layers=cfg_model.get("n_layers", 2),
            edge_attr_dims=edge_attr_dims,
        )
        self.comm_assign = CommunityAssigner(
            d_model=cfg_model.get("d_model", 32),
            num_communities=cfg_model.get("num_communities", 4),
        )
        self.readout = ReadoutHead(cfg_model.get("d_model", 32), num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]):
        ts = batch["ts"]
        if not self.experts:
            raise RuntimeError("No experts configured")

        expert_outputs = []
        primary_output = self.experts[0](ts)
        expert_outputs.append(primary_output)
        shared_edge_index = primary_output["edge_index"].detach()
        for expert in self.experts[1:]:
            expert.set_edge_index(shared_edge_index)
            expert_outputs.append(expert(ts))
        merged = build_edge_feature_table(expert_outputs)
        alpha = self.gate(merged["edge_features"])
        routed = route_edges(merged["edge_index"], expert_outputs, alpha)
        ts_mean = ts.mean(dim=1, keepdim=True)
        node_feat_init = self.node_encoder(ts_mean.transpose(1, 2))
        graph_batch = build_multirel_graph_batch(node_feat_init, routed)
        H = self.transformer(graph_batch)
        S, prototypes = self.comm_assign(H)
        A_comm, z_comm = build_community_graph(S, H)
        logits, z_graph = self.readout(H, z_comm)
        aux = {
            "H": H,
            "S": S,
            "alpha": alpha,
            "A_comm": A_comm,
            "z_comm": z_comm,
            "z_graph": z_graph,
            "prototypes": prototypes,
        }
        return logits, aux
