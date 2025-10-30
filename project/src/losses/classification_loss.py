"""Classification loss helpers."""
from __future__ import annotations

import torch
from torch import nn


_cross_entropy = nn.CrossEntropyLoss()


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return _cross_entropy(logits, labels)
