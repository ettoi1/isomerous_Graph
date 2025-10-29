"""Dataset utilities for graph-based rs-fMRI classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BrainGraphSample(TypedDict, total=False):
    """Container describing a single subject-level sample."""

    subject_id: str
    timeseries: torch.Tensor | np.ndarray | Path | str
    label: int
    site: str | int
    confounds: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass(slots=True)
class DatasetIndex:
    """Wrapper around the CSV index used by :class:`StaticFMRIDataset`."""

    frame: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Path | str) -> "DatasetIndex":
        df = pd.read_csv(path)
        required_columns = {"subject_id", "label"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Index CSV missing required columns: {missing}")
        return cls(frame=df)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.frame)

    def row(self, i: int) -> Dict[str, Any]:
        return self.frame.iloc[i].to_dict()


class StaticFMRIDataset(Dataset[BrainGraphSample]):
    """Dataset that reads cached ROI time-series from disk.

    The class assumes that the preprocessing stage has already extracted ROI
    time-series for each subject and stored them as ``.npy`` files inside a
    cache directory.  The CSV index is expected to provide at least the subject
    identifier, the path to the cached time-series (relative to ``cache_dir``) or
    absolute, a label, and optionally site information.
    """

    def __init__(
        self,
        index_csv: Path | str,
        cache_dir: Path | str,
        transforms: Optional[Callable[[BrainGraphSample], BrainGraphSample]] = None,
        load_timeseries: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.index = DatasetIndex.from_csv(index_csv)
        self.transforms = transforms
        self.load_timeseries = load_timeseries

    def _resolve_timeseries_path(self, row: Dict[str, Any]) -> Path:
        ts_path = row.get("timeseries_path")
        if ts_path is None:
            ts_path = self.cache_dir / f"{row['subject_id']}_timeseries.npy"
        else:
            ts_path = Path(ts_path)
            if not ts_path.is_absolute():
                ts_path = self.cache_dir / ts_path
        return ts_path

    def __getitem__(self, index: int) -> BrainGraphSample:
        row = self.index.row(index)
        sample: BrainGraphSample = {
            "subject_id": row["subject_id"],
            "label": int(row["label"]),
            "site": row.get("site", ""),
            "confounds": {key: row[key] for key in row.keys() if key.startswith("conf_")},
            "meta": {key: row[key] for key in row.keys() if key not in {"subject_id", "label"}},
        }

        ts_path = self._resolve_timeseries_path(row)
        if not ts_path.exists():
            raise FileNotFoundError(f"Time-series file not found for subject {sample['subject_id']}: {ts_path}")

        sample["timeseries"] = ts_path
        if self.load_timeseries:
            sample["timeseries"] = torch.from_numpy(np.load(ts_path)).float()

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.index)


__all__ = [
    "BrainGraphSample",
    "DatasetIndex",
    "StaticFMRIDataset",
]
