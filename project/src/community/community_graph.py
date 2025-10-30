"""Community graph construction."""
from __future__ import annotations

import torch


def build_community_graph(S: torch.Tensor, H: torch.Tensor):
    weights = S / (S.sum(dim=1, keepdim=True) + 1e-6)
    C = torch.einsum("bnm,bnd->bmd", weights, H)
    A = torch.einsum("bmd,bnd->bmn", C, C)
    z_comm = C.mean(dim=1)
    return A, z_comm
