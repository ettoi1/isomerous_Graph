"""Linear synchrony expert."""
from __future__ import annotations

from typing import Dict

import torch

from project.src.experts.base_expert import BaseExpert
from project.src.utils.connectivity import compute_pearson, compute_spearman


class LinearSyncExpert(BaseExpert):
    """Expert E1 capturing linear and monotonic co-fluctuations."""

    def __init__(self, top_k: int = 200, expert_id: int = 0):
        super().__init__(expert_id=expert_id, out_dim=2, top_k=top_k)

    def forward(self, ts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Pearson/Spearman features for the top-K correlated edges.

        Args:
            ts: Tensor[B, T, N] time series. Currently assumes ``B = 1``.

        Returns:
            Dictionary with ``edge_index``, ``edge_repr`` and ``expert_id``.
        """

        if ts.ndim != 3:
            raise ValueError("Expected ts with shape [B, T, N]")
        if ts.size(0) != 1:
            raise ValueError("LinearSyncExpert currently assumes B=1")

        ts_single = ts[0]
        pearson = compute_pearson(ts_single)
        spearman = compute_spearman(ts_single)
        N = pearson.shape[0]
        if N < 2:
            raise ValueError("Need at least two ROIs to form edges")

        iu, ju = torch.triu_indices(N, N, offset=1)
        scores = pearson[iu, ju].abs()
        k = min(self.top_k, scores.numel())
        _, idx = torch.topk(scores, k=k, largest=True)
        sel_i = iu[idx]
        sel_j = ju[idx]

        edge_repr = torch.stack(
            [pearson[sel_i, sel_j], spearman[sel_i, sel_j]], dim=-1
        )
        edge_index = torch.stack([sel_i, sel_j], dim=0).long()
        self.set_edge_index(edge_index)

        return {
            "edge_index": edge_index,
            "edge_repr": edge_repr,
            "expert_id": self.expert_id,
        }
