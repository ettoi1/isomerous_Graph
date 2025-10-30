"""Simple logging helpers for training metrics."""
from __future__ import annotations

from typing import Dict


def format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
