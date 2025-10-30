"""Utilities to merge expert features."""
from __future__ import annotations

from typing import Dict, List

import torch


def build_edge_feature_table(expert_outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
    if not expert_outputs:
        raise ValueError("expert_outputs must not be empty")

    edge_index = expert_outputs[0]["edge_index"]
    for expert in expert_outputs[1:]:
        if not torch.equal(edge_index, expert["edge_index"]):
            raise ValueError("All experts are expected to share the same edge_index in this version")
    features = torch.cat([expert["edge_repr"] for expert in expert_outputs], dim=-1)
    return {
        "edge_index": edge_index,
        "edge_features": features,
        "expert_outputs": expert_outputs,
    }
