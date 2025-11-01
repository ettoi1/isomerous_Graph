"""Compute pairwise edge features from ROI time-series."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np

from src.utils.torch_import import torch


def compute_edge_features(ts: np.ndarray, config: Dict[str, Any]) -> torch.Tensor:
    """Compute pairwise connectivity metrics between ROIs.

    Parameters
    ----------
    ts: np.ndarray
        Time-series data with shape ``(T, N)`` where ``T`` is the number of time
        points and ``N`` is the number of ROIs.
    config: Dict[str, Any]
        Configuration dictionary for additional hyperparameters. Currently unused
        but kept for API compatibility.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N, N, d_edge_raw)`` containing pairwise features.
    """

    T, N = ts.shape
    ts_centered = ts - ts.mean(axis=0, keepdims=True)
    cov = ts_centered.T @ ts_centered / max(T - 1, 1)
    var = np.diag(cov)
    std = np.sqrt(np.maximum(var, 1e-8))

    pearson = cov / (std[:, None] * std[None, :])
    pearson = np.clip(pearson, -1.0, 1.0)

    # TODO: Implement real partial correlation calculation.
    partial_corr = pearson.copy()

    # TODO: Implement real dynamic functional connectivity variance.
    dfc_var = np.zeros_like(pearson)

    # TODO: Implement real mutual information estimation instead of proxy.
    mutual_info = np.abs(pearson)

    features = np.stack([pearson, partial_corr, dfc_var, mutual_info], axis=-1)
    np.fill_diagonal(features[:, :, 0], 0.0)
    np.fill_diagonal(features[:, :, 1], 0.0)
    np.fill_diagonal(features[:, :, 2], 0.0)
    np.fill_diagonal(features[:, :, 3], 0.0)

    edge_feat_tensor = torch.from_numpy(features.astype(np.float32))
    return edge_feat_tensor


if __name__ == "__main__":
    dummy_ts = np.random.randn(10, 5).astype(np.float32)
    feats = compute_edge_features(dummy_ts, {})
    print("Edge feature tensor shape:", feats.shape)
