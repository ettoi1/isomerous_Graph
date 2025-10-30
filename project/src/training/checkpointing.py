"""Checkpoint helpers (placeholders for the demo)."""
from __future__ import annotations

from typing import Any, Dict

import torch


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    torch.save(state, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")
