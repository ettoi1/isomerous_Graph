"""Dataset helpers for toy examples."""
from __future__ import annotations

from typing import Dict, List

import torch


def generate_random_dataset(num_samples: int, T: int, N: int, num_classes: int) -> Dict[str, List]:
    data = []
    labels = []
    for _ in range(num_samples):
        data.append(torch.randn(T, N))
        labels.append(int(torch.randint(0, num_classes, (1,)).item()))
    return {"data": data, "labels": labels}
