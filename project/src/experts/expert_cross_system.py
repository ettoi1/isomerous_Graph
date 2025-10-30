"""Cross-system expert focusing on spatially distant ROI pairs."""
from __future__ import annotations

from typing import Dict

import torch

from project.src.experts.base_expert import BaseExpert
from project.src.utils.connectivity import compute_pearson


class CrossSystemExpert(BaseExpert):
    """Expert E4 approximating long-range modulatory couplings."""

    def __init__(self, top_k: int = 200, expert_id: int = 3):
        super().__init__(expert_id=expert_id, out_dim=2, top_k=top_k)

    def forward(self, ts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Augment shared edges with Pearson strength and index distance.

        Args:
            ts: Tensor[B, T, N] time series. Currently assumes ``B = 1``.

        Returns:
            Dictionary with ``edge_index``, ``edge_repr`` and ``expert_id``.
        """
        if ts.ndim != 3:
            raise ValueError("Expected ts with shape [B, T, N]")
        if ts.size(0) != 1:
            raise ValueError("CrossSystemExpert currently assumes B=1")

        edge_index = self._get_edge_index()
        ts_single = ts[0]
        pearson = compute_pearson(ts_single)
        sel_i, sel_j = edge_index[0].long(), edge_index[1].long()

        N = pearson.shape[0]
        norm_dist = (sel_j - sel_i).abs().float() / float(N)
        edge_repr = torch.stack([pearson[sel_i, sel_j], norm_dist], dim=-1)

        return {
            "edge_index": edge_index,
            "edge_repr": edge_repr,
            "expert_id": self.expert_id,
        }
