"""Direct coupling expert."""
from __future__ import annotations

from typing import Dict

import torch

from project.src.experts.base_expert import BaseExpert
from project.src.utils.connectivity import compute_partial_correlation


class DirectCouplingExpert(BaseExpert):
    """Expert E2 capturing conditional (partial) correlations."""

    def __init__(self, top_k: int = 200, expert_id: int = 1):
        super().__init__(expert_id=expert_id, out_dim=1, top_k=top_k)

    def forward(self, ts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract partial-correlation features for shared candidate edges.

        Args:
            ts: Tensor[B, T, N] time series. Currently assumes ``B = 1``.

        Returns:
            Dictionary with ``edge_index``, ``edge_repr`` and ``expert_id``.
        """
        if ts.ndim != 3:
            raise ValueError("Expected ts with shape [B, T, N]")
        if ts.size(0) != 1:
            raise ValueError("DirectCouplingExpert currently assumes B=1")

        edge_index = self._get_edge_index()
        ts_single = ts[0]
        pcorr = compute_partial_correlation(ts_single)
        sel_i, sel_j = edge_index[0].long(), edge_index[1].long()
        edge_repr = pcorr[sel_i, sel_j].unsqueeze(-1)

        return {
            "edge_index": edge_index,
            "edge_repr": edge_repr,
            "expert_id": self.expert_id,
        }
