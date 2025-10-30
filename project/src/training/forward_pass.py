"""Forward pass helpers for the demo training loop."""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from project.src.losses.total_loss import total_loss_main


def forward_pass(model: torch.nn.Module, batch: Dict[str, torch.Tensor], loss_weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run a full forward pass and compute the composite loss."""

    logits, aux = model(batch)
    losses = total_loss_main(logits=logits, labels=batch["label"], aux=aux, loss_weights=loss_weights)
    total = losses.pop("total_loss")
    return total, losses
