"""Community graph construction."""
from __future__ import annotations

import torch


def build_community_graph(S: torch.Tensor, H: torch.Tensor):
    community_mass = S.sum(dim=1, keepdim=True).clamp_min(1e-6)
    weights = S / community_mass
    C = torch.einsum("bnm,bnd->bmd", weights, H)
    A_comm = torch.matmul(C, C.transpose(-1, -2))
    z_comm = C.mean(dim=1)
    return A_comm, z_comm
