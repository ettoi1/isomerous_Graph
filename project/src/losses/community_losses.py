"""Community related losses."""
from __future__ import annotations

from typing import Dict

import torch

from project.src.community.community_guided_routing import routing_consistency_loss
from project.src.community.community_regularizer import community_consistency_loss


def community_aux_losses(H: torch.Tensor, S: torch.Tensor, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
    comm_loss = community_consistency_loss(H, S)
    route_loss = routing_consistency_loss(S, alpha)
    return {
        "comm_consistency": comm_loss,
        "routing_consistency": route_loss,
    }
