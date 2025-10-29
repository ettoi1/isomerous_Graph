"""Regularisation terms for MoE gating."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F


def alpha_sparsity(alpha: torch.Tensor, mode: str = "entropy", target_k: int | None = None) -> torch.Tensor:
    if mode == "l1":
        return alpha.abs().mean()
    if mode == "entropy":
        return -(alpha * (alpha.clamp_min(1e-8).log())).sum(dim=-1).mean()
    raise ValueError(f"Unknown sparsity mode: {mode}")


def alpha_stability(alpha_seq: List[torch.Tensor]) -> torch.Tensor:
    if len(alpha_seq) < 2:
        return torch.tensor(0.0, device=alpha_seq[0].device)
    diffs = [F.mse_loss(alpha_seq[i], alpha_seq[i - 1]) for i in range(1, len(alpha_seq))]
    return torch.stack(diffs).mean()


__all__ = ["alpha_sparsity", "alpha_stability"]
