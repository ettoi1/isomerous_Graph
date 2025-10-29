"""Classification losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_with_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


__all__ = ["cross_entropy_with_logits"]
