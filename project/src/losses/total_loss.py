"""Total loss aggregation."""
from __future__ import annotations

from typing import Dict

import torch

from project.src.losses.classification_loss import classification_loss
from project.src.losses.community_losses import community_aux_losses
from project.src.losses.routing_losses import expert_balance_loss, routing_sparsity_loss


def total_loss_main(logits: torch.Tensor, labels: torch.Tensor, aux: Dict[str, torch.Tensor], loss_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    cls = classification_loss(logits, labels)
    losses["classification"] = cls
    community_losses = community_aux_losses(aux["H"], aux["S"], aux["alpha"])
    losses.update(community_losses)
    losses["routing_sparsity"] = routing_sparsity_loss(aux["alpha"])
    losses["expert_balance"] = expert_balance_loss(aux["alpha"])

    total = cls
    total = total + loss_weights.get("lambda_comm", 0.1) * community_losses["comm_consistency"]
    total = total + loss_weights.get("lambda_route_sparse", 0.1) * losses["routing_sparsity"]
    total = total + loss_weights.get("lambda_route_balance", 0.1) * losses["expert_balance"]
    losses["total_loss"] = total
    return losses
