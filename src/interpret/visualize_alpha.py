"""Utilities for inspecting mixture-of-experts gate outputs."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_expert_distribution(alpha: torch.Tensor, graph_index: List[Tuple[int, int]], atlas_meta) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    mean_alpha = alpha.mean(dim=0).cpu().numpy()
    ax.bar(np.arange(len(mean_alpha)), mean_alpha)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Mean selection probability")
    ax.set_title("MoE expert usage")
    return fig


def edge_importance_by_expert(alpha: torch.Tensor, e_values: torch.Tensor) -> pd.DataFrame:
    data = alpha.cpu().numpy() * e_values.unsqueeze(-1).cpu().numpy()
    importance = data.mean(axis=0)
    return pd.DataFrame({"expert": np.arange(len(importance)), "importance": importance})


__all__ = ["plot_expert_distribution", "edge_importance_by_expert"]
