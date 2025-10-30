"""Cross-validation helpers."""
from __future__ import annotations


def k_fold_indices(num_samples: int, k: int):
    fold_size = max(1, num_samples // k)
    for i in range(k):
        start = i * fold_size
        end = min(num_samples, start + fold_size)
        yield list(range(start, end))
