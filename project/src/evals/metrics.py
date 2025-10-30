"""Evaluation metrics placeholders."""
from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean()
