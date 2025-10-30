"""Simple split utilities."""
from __future__ import annotations

from typing import Dict, Tuple


def train_val_split(dataset: Dict[str, list], val_fraction: float = 0.2) -> Tuple[Dict[str, list], Dict[str, list]]:
    size = len(dataset["data"])
    split = int(size * (1 - val_fraction))
    train = {"data": dataset["data"][:split], "labels": dataset["labels"][:split]}
    val = {"data": dataset["data"][split:], "labels": dataset["labels"][split:]}
    return train, val
