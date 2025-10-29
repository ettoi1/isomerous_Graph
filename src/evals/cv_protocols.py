"""Cross-validation split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold


@dataclass
class Split:
    train_idx: List[int]
    test_idx: List[int]


def make_splits(index_df: pd.DataFrame, strategy: Literal["stratified_kfold", "group_site_kfold", "leave_one_site_out"], seed: int = 42, folds: int = 5) -> List[Split]:
    y = index_df["label"]
    if strategy == "stratified_kfold":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return [Split(train_idx=list(train), test_idx=list(test)) for train, test in splitter.split(index_df, y)]
    if strategy == "group_site_kfold":
        groups = index_df["site"]
        splitter = GroupKFold(n_splits=folds)
        return [Split(train_idx=list(train), test_idx=list(test)) for train, test in splitter.split(index_df, y, groups)]
    if strategy == "leave_one_site_out":
        splits: List[Split] = []
        for site, site_df in index_df.groupby("site"):
            test_idx = site_df.index.tolist()
            train_idx = index_df.index.difference(site_df.index).tolist()
            splits.append(Split(train_idx=train_idx, test_idx=test_idx))
        return splits
    raise ValueError(f"Unknown CV strategy: {strategy}")


__all__ = ["Split", "make_splits"]
