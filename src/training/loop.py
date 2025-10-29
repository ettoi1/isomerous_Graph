"""Training and evaluation loops."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from ..losses.classification import cross_entropy_with_logits


def train_one_epoch(model_bundle: Dict[str, object], dataloader: DataLoader, optim: torch.optim.Optimizer, cfg: Dict[str, object]) -> Dict[str, float]:
    model_bundle["classifier"].train()
    total_loss = 0.0
    for batch in dataloader:
        logits = model_bundle["classifier"](batch["node_feats"], batch["adj"])
        loss = cross_entropy_with_logits(logits, batch["labels"])
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return {"loss": total_loss / len(dataloader)}


def validate(model_bundle: Dict[str, object], dataloader: DataLoader) -> Dict[str, float]:
    model_bundle["classifier"].eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            logits = model_bundle["classifier"](batch["node_feats"], batch["adj"])
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].numel()
    return {"accuracy": correct / max(total, 1)}


def inference(model_bundle: Dict[str, object], dataloader: DataLoader) -> torch.Tensor:
    model_bundle["classifier"].eval()
    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            logits = model_bundle["classifier"](batch["node_feats"], batch["adj"])
            outputs.append(logits.softmax(dim=-1))
    return torch.cat(outputs, dim=0)


__all__ = ["train_one_epoch", "validate", "inference"]
