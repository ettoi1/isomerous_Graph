"""Toy training loop utilities."""
from __future__ import annotations

from typing import Dict, Iterable

import torch

from project.src.training.forward_pass import forward_pass


def run_epoch(model: torch.nn.Module, batches: Iterable[Dict[str, torch.Tensor]], loss_weights: Dict[str, float]) -> Dict[str, float]:
    """Run a single optimisation epoch over ``batches``.

    The implementation is intentionally minimal and intended for smoke tests.
    """

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    log_sums: Dict[str, float] = {}
    num_batches = 0
    for batch in batches:
        optimiser.zero_grad()
        total_loss, loss_dict = forward_pass(model, batch, loss_weights)
        total_loss.backward()
        optimiser.step()
        num_batches += 1
        for name, value in loss_dict.items():
            log_sums[name] = log_sums.get(name, 0.0) + float(value.detach())
    if num_batches == 0:
        return {}
    return {name: value / num_batches for name, value in log_sums.items()}
