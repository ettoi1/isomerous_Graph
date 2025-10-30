"""Base expert definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class BaseExpert(nn.Module, ABC):
    """Abstract base class for edge feature experts.

    Each expert works with a shared :class:`edge_index` supplied externally so
    that their edge representations are aligned. The simplified pipeline still
    assumes ``B = 1`` batches, but keeps the batch dimension in the public
    interface for future extensibility.
    """

    def __init__(self, expert_id: int, out_dim: int, top_k: int = 200):
        super().__init__()
        self.expert_id = expert_id
        self.out_dim = out_dim
        self.top_k = top_k
        self._edge_index: torch.Tensor | None = None

    def set_edge_index(self, edge_index: torch.Tensor) -> None:
        """Register the candidate edges that this expert must describe."""

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(dtype=torch.long)
        self._edge_index = edge_index.detach().clone()

    def _get_edge_index(self) -> torch.Tensor:
        if self._edge_index is None:
            raise RuntimeError(
                "Edge index has not been set for this expert. "
                "Ensure LinearSyncExpert runs first and shares its selection."
            )
        return self._edge_index

    @abstractmethod
    def forward(self, ts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return edge representations for the shared edge set."""
