"""Graph construction utilities for multiple connectivity metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import torch


@dataclass
class SparseAdj:
    """Sparse adjacency matrix representation."""

    edge_index: torch.LongTensor
    edge_weight: torch.FloatTensor
    num_nodes: int

    def to_dense(self) -> torch.Tensor:
        matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=self.edge_weight.dtype)
        matrix[self.edge_index[0], self.edge_index[1]] = self.edge_weight
        return matrix


def _validate_timeseries(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts)
    if ts.ndim != 2:
        raise ValueError("Timeseries must be a 2D array of shape [time, nodes]")
    if np.isnan(ts).any():  # pragma: no cover - sanity check
        raise ValueError("Timeseries contains NaNs")
    return ts


def compute_correlation(
    ts: np.ndarray,
    method: Literal["pearson", "partial"],
    **kwargs,
) -> np.ndarray:
    """Compute a connectivity matrix using the requested statistical metric."""

    ts = _validate_timeseries(ts)
    if method == "pearson":
        return np.corrcoef(ts, rowvar=False)
    if method == "partial":
        covariance = np.cov(ts, rowvar=False)
        precision = np.linalg.pinv(covariance)
        d = np.sqrt(np.diag(precision))
        partial = -precision / np.outer(d, d)
        np.fill_diagonal(partial, 1.0)
        return partial
    raise NotImplementedError(f"Correlation method '{method}' is not implemented")


def threshold_or_knn(
    A: np.ndarray,
    mode: Literal["topk", "quantile", "abs"],
    k: Optional[int] = None,
    q: Optional[float] = None,
) -> SparseAdj:
    """Sparsify a dense connectivity matrix."""

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Connectivity matrix must be square")

    num_nodes = A.shape[0]
    np.fill_diagonal(A, 0.0)

    if mode == "topk":
        if k is None:
            raise ValueError("Parameter k is required for topk sparsification")
        topk_idx = np.argpartition(-A, kth=k, axis=1)[:, :k]
        rows = np.repeat(np.arange(num_nodes), k)
        cols = topk_idx.reshape(-1)
        weights = A[rows, cols]
    elif mode == "quantile":
        if q is None:
            raise ValueError("Parameter q is required for quantile sparsification")
        threshold = np.quantile(A[A > 0], q)
        mask = A >= threshold
        rows, cols = np.where(mask)
        weights = A[rows, cols]
    elif mode == "abs":
        if q is None:
            raise ValueError("Parameter q is required for abs sparsification")
        mask = np.abs(A) >= q
        rows, cols = np.where(mask)
        weights = A[rows, cols]
    else:
        raise ValueError(f"Unknown sparsification mode: {mode}")

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return SparseAdj(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)


def normalize_connectivity(
    A: np.ndarray,
    norm: Literal["fisherz", "zscore", "minmax"],
) -> np.ndarray:
    """Normalise connectivity scores to a comparable range."""

    if norm == "fisherz":
        A = np.clip(A, -0.999999, 0.999999)
        return np.arctanh(A)
    if norm == "zscore":
        mean = A.mean()
        std = A.std() + 1e-8
        return (A - mean) / std
    if norm == "minmax":
        min_val = A.min()
        max_val = A.max()
        denom = (max_val - min_val) or 1.0
        return (A - min_val) / denom
    raise ValueError(f"Unknown normalisation '{norm}'")


__all__ = [
    "SparseAdj",
    "compute_correlation",
    "threshold_or_knn",
    "normalize_connectivity",
]
