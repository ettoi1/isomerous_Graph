"""Connectivity metrics computed from ROI time series."""
from __future__ import annotations

import torch


def _check_ts(ts: torch.Tensor) -> torch.Tensor:
    if ts.ndim != 2:
        raise ValueError("ts must have shape [T, N]")
    if ts.size(0) < 1 or ts.size(1) < 1:
        raise ValueError("ts must contain at least one time point and one ROI")
    return ts.to(torch.float64)


def compute_pearson(ts: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation between all ROI time series.

    Args:
        ts: Tensor[T, N], time series for one subject (no batch dim).
            T = time points, N = number of ROIs.

    Returns:
        corr: Tensor[N, N], Pearson correlation matrix in [-1,1].
        Diagonal should be 1.0.
        Implementation detail:
            - subtract mean over time
            - corr_ij = cov_ij / (std_i * std_j)
    """

    ts64 = _check_ts(ts)
    T = ts64.size(0)
    demeaned = ts64 - ts64.mean(dim=0, keepdim=True)
    if T > 1:
        denom = T - 1
    else:
        denom = T
    cov = (demeaned.t() @ demeaned) / denom
    var = cov.diagonal().clamp_min(1e-12)
    std = torch.sqrt(var)
    denom_matrix = std.unsqueeze(0) * std.unsqueeze(1)
    corr = cov / denom_matrix.clamp_min(1e-12)
    corr = corr.clamp(-1.0, 1.0)
    corr.fill_diagonal_(1.0)
    return corr.to(ts.dtype)


def _average_rank_transform(ts: torch.Tensor) -> torch.Tensor:
    T, N = ts.shape
    ranks = torch.empty((T, N), dtype=torch.float64, device=ts.device)
    for j in range(N):
        column = ts[:, j]
        sorted_vals, order = torch.sort(column)
        ranks_col = torch.empty(T, dtype=torch.float64, device=ts.device)
        start = 0
        while start < T:
            end = start + 1
            while end < T and torch.isclose(sorted_vals[end], sorted_vals[start]):
                end += 1
            avg_rank = (start + end - 1) / 2.0 + 1.0
            ranks_col[order[start:end]] = avg_rank
            start = end
        ranks[:, j] = ranks_col
    return ranks


def compute_spearman(ts: torch.Tensor) -> torch.Tensor:
    """
    Compute Spearman rank correlation (rank-based Pearson).
    Steps:
        1. For each ROI signal ts[:, i], replace with its rank over time
           (ties can be handled by average rank).
        2. Call compute_pearson() on the ranked signals.

    Args:
        ts: Tensor[T, N]

    Returns:
        spear: Tensor[N, N], in [-1,1].
    """

    ts64 = _check_ts(ts)
    ranks = _average_rank_transform(ts64)
    return compute_pearson(ranks.to(ts.dtype))


def compute_partial_correlation(ts: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute partial correlation matrix from ROI time series using
    inverse covariance (precision matrix).

    Args:
        ts: Tensor[T, N]
        eps: small value added to covariance diagonal for numerical stability.

    Returns:
        pcorr: Tensor[N, N], where
            pcorr_ij = -prec_ij / sqrt(prec_ii * prec_jj)
        Diagonal should be 1.0.
    Implementation detail:
        - compute covariance (N x N)
        - add eps * I
        - invert (or pseudo-inverse)
        - apply formula above
    """

    ts64 = _check_ts(ts)
    T = ts64.size(0)
    demeaned = ts64 - ts64.mean(dim=0, keepdim=True)
    if T > 1:
        denom = T - 1
    else:
        denom = T
    cov = (demeaned.t() @ demeaned) / denom
    N = cov.size(0)
    cov = cov + eps * torch.eye(N, dtype=torch.float64, device=cov.device)
    try:
        precision = torch.linalg.inv(cov)
    except RuntimeError:
        precision = torch.linalg.pinv(cov)
    diag = precision.diagonal().clamp_min(1e-12)
    denom = torch.sqrt(diag).unsqueeze(0) * torch.sqrt(diag).unsqueeze(1)
    pcorr = -precision / denom.clamp_min(1e-12)
    pcorr.fill_diagonal_(1.0)
    return pcorr.to(ts.dtype)


def compute_mutual_information(ts: torch.Tensor, num_bins: int = 16) -> torch.Tensor:
    """
    Approximate pairwise mutual information between ROI time series by
    histogram-based discrete MI.

    Args:
        ts: Tensor[T, N]
        num_bins: number of histogram bins for discretization.

    Returns:
        mi: Tensor[N, N], mi[i,j] >= 0
        Symmetric. Diagonal can be set to 0.
    Procedure (simple, non-differentiable is OK for now):
        - Discretize each ROI signal into num_bins using uniform binning.
        - Estimate p(i), p(j), p(i,j) as empirical frequencies over T.
        - MI(i,j) = sum_{a,b} p(a,b) * log(p(a,b)/(p(a)*p(b))).
    """

    if num_bins <= 1:
        raise ValueError("num_bins must be greater than 1")
    ts64 = _check_ts(ts)
    T, N = ts64.shape
    device = ts64.device
    discrete = torch.zeros((T, N), dtype=torch.long, device=device)
    for j in range(N):
        col = ts64[:, j]
        min_v = col.min()
        max_v = col.max()
        if torch.isclose(max_v, min_v):
            discrete[:, j] = 0
            continue
        bins = torch.linspace(min_v, max_v, num_bins + 1, device=device)
        idx = torch.bucketize(col, bins, right=False) - 1
        idx = idx.clamp(0, num_bins - 1)
        discrete[:, j] = idx

    marginals = torch.zeros((N, num_bins), dtype=torch.float64, device=device)
    for j in range(N):
        counts = torch.bincount(discrete[:, j], minlength=num_bins).to(torch.float64)
        marginals[j] = counts / T

    mi = torch.zeros((N, N), dtype=torch.float64, device=device)
    ones = torch.ones(T, dtype=torch.float64, device=device)
    for i in range(N):
        for j in range(i, N):
            joint = torch.zeros((num_bins, num_bins), dtype=torch.float64, device=device)
            idx_i = discrete[:, i]
            idx_j = discrete[:, j]
            joint.index_put_((idx_i, idx_j), ones, accumulate=True)
            joint /= T
            p_i = marginals[i]
            p_j = marginals[j]
            denom = p_i[:, None] * p_j[None, :]
            mask = (joint > 0) & (denom > 0)
            contrib = joint[mask] * torch.log(joint[mask] / denom[mask])
            mi_val = contrib.sum()
            mi[i, j] = mi_val
            mi[j, i] = mi_val
    mi = mi.clamp_min(0.0)
    mi.fill_diagonal_(0.0)
    return mi.to(ts.dtype)
