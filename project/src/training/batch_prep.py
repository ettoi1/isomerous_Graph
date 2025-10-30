"""Batch preparation utilities for toy data generation."""
from __future__ import annotations

from typing import List, Sequence

import torch


def prepare_batch(samples: Sequence[torch.Tensor], labels: Sequence[int]) -> dict:
    """Construct a minimal batch dictionary for the BrainGraph model.

    Parameters
    ----------
    samples:
        Sequence of tensors shaped ``[T, N]`` where ``T`` is the number of
        time-points and ``N`` is the number of regions of interest (ROIs).
    labels:
        Sequence of integer class labels.

    Returns
    -------
    dict
        A batch dictionary with entries ``ts`` and ``label`` as required by the
        training pipeline. ``ts`` has shape ``[B, T, N]`` and ``label`` has shape
        ``[B]``.
    """

    if len(samples) != len(labels):
        raise ValueError("samples and labels must have matching length")

    batch_ts: List[torch.Tensor] = []
    for ts in samples:
        if ts.ndim != 2:
            raise ValueError("each sample must have shape [T, N]")
        batch_ts.append(ts)

    stacked_ts = torch.stack(batch_ts, dim=0)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    return {"ts": stacked_ts, "label": batch_labels}
