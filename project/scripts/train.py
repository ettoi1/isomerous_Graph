"""Minimal training entry point to validate the model pipeline."""
from __future__ import annotations

import torch

from project.src.models.model_bundle import BrainGraphModel
from project.src.training.forward_pass import forward_pass


def main():
    B, T, N = 1, 50, 30
    num_classes = 3
    ts = torch.randn(B, T, N)
    label = torch.randint(0, num_classes, (B,))
    batch = {"ts": ts, "label": label}

    cfg_model = {"d_node_init": 16, "d_model": 32, "n_layers": 2, "num_communities": 4}
    cfg_experts = {"top_k": 200}
    loss_weights = {"lambda_comm": 0.1, "lambda_route_sparse": 0.1, "lambda_route_balance": 0.1}

    model = BrainGraphModel(cfg_model=cfg_model, cfg_experts=cfg_experts, num_classes=num_classes)
    total_loss, loss_dict = forward_pass(model, batch, loss_weights)
    print("total_loss", float(total_loss.item()))
    print("loss_terms", sorted(loss_dict.keys()))


if __name__ == "__main__":
    main()
