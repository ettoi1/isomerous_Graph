"""Evaluation metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn import metrics


def roc_auc(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    return metrics.roc_auc_score(y_true, y_prob)


def pr_auc(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    return metrics.auc(recall, precision)


def balanced_accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    return metrics.balanced_accuracy_score(y_true, y_pred)


def ece_calibration(y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10) -> float:
    prob = np.asarray(y_prob)
    true = np.asarray(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (prob >= bins[i]) & (prob < bins[i + 1])
        if mask.any():
            accuracy = true[mask].mean()
            confidence = prob[mask].mean()
            ece += np.abs(accuracy - confidence) * mask.mean()
    return ece


__all__ = ["roc_auc", "pr_auc", "balanced_accuracy", "ece_calibration"]
