"""Utilities for preprocessing resting-state fMRI data.

This module centralises a thin layer of wrappers around nibabel and nilearn so
that the rest of the code base can rely on a consistent set of tensor shapes
and metadata.  The actual heavy lifting is expected to happen in the upstream
processing pipeline; the helpers here simply orchestrate that work, cache the
results to ``data/interim`` and expose them as NumPy arrays suitable for graph
construction.

None of the functions depend on a concrete dataset layout – paths are passed in
explicitly – which keeps the logic testable and makes it possible to plug in a
synthetic dataset during development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

try:  # Optional dependency, importing lazily keeps unit-tests light.
    import nibabel as nib
except Exception:  # pragma: no cover - nibabel is optional in CI.
    nib = None


@dataclass
class NiftiLike:
    """Light-weight container for the data we care about from a NIfTI image.

    The full :class:`nib.Nifti1Image` object carries affine matrices and header
    metadata.  For preprocessing we often only need the dense array and the
    affine, hence this stripped-down representation.
    """

    data: np.ndarray
    affine: Optional[np.ndarray] = None


def load_raw_fmri(subject_id: str, paths: Dict[str, Path]) -> NiftiLike:
    """Load a raw 4D fMRI volume.

    Parameters
    ----------
    subject_id:
        Identifier of the subject to load.
    paths:
        Mapping that must include a ``"fmri"`` entry pointing to the file on
        disk.  Additional entries (e.g. anatomical or confound files) are
        ignored but allowed so the same dictionary can be shared with other
        helpers.
    """

    fmri_path = Path(paths["fmri"]).expanduser()
    if not fmri_path.exists():
        raise FileNotFoundError(f"fMRI volume for {subject_id} not found: {fmri_path}")

    if nib is None:  # pragma: no cover - handled during runtime when nibabel missing.
        raise ImportError("nibabel is required to load NIfTI files but is not installed")

    image = nib.load(str(fmri_path))
    return NiftiLike(data=image.get_fdata(), affine=image.affine)


def extract_roi_timeseries(
    img: NiftiLike,
    atlas: str | Path,
    confounds_cfg: Optional[Dict[str, Iterable[str]]] = None,
) -> np.ndarray:
    """Extract the ROI-wise time-series matrix from a 4D fMRI volume.

    The routine delegates to :mod:`nilearn.signal` if the package is available
    and falls back to a basic mean pooling implementation otherwise.  Confound
    regression is intentionally conservative: only well-known nuisance
    regressors are supported out of the box, but the function can be extended by
    providing the necessary design matrices inside ``confounds_cfg``.
    """

    atlas_path = Path(atlas).expanduser()
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas image not found: {atlas_path}")

    data = np.asarray(img.data)
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D fMRI volume, got shape {data.shape}")

    # Import nilearn lazily to keep the dependency optional.
    masker = None
    if confounds_cfg is None:
        confounds_cfg = {}

    try:  # pragma: no cover - nilearn may be missing in minimal environments.
        from nilearn import input_data, signal

        masker = input_data.NiftiLabelsMasker(
            labels_img=str(atlas_path),
            standardize=confounds_cfg.get("standardize", True),
            detrend=confounds_cfg.get("detrend", True),
            low_pass=confounds_cfg.get("bandpass", (None, None))[1],
            high_pass=confounds_cfg.get("bandpass", (None, None))[0],
            t_r=confounds_cfg.get("tr"),
        )
    except Exception:
        masker = None

    if masker is not None:  # pragma: no cover - depends on nilearn being present.
        confounds = confounds_cfg.get("design_matrix")
        return masker.fit_transform(img.data, confounds=confounds)

    # Fallback: naïve per-label averaging.
    atlas_img = load_raw_fmri("atlas", {"fmri": atlas_path}).data
    unique_labels = np.unique(atlas_img)
    unique_labels = unique_labels[unique_labels > 0]
    timepoints = data.shape[-1]
    roi_count = len(unique_labels)
    ts = np.zeros((timepoints, roi_count), dtype=np.float32)
    for idx, label in enumerate(unique_labels):
        mask = atlas_img == label
        roi_signal = data[mask]
        ts[:, idx] = roi_signal.mean(axis=0)

    return ts


def save_timeseries(subject_id: str, ts: np.ndarray, meta: Dict[str, object], out_dir: Path | str = "data/interim") -> Path:
    """Persist ROI time-series alongside metadata.

    The time-series is stored as ``.npy`` for efficiency, while metadata is
    written to a companion ``.json`` file.  The function returns the path to the
    ``.npy`` file so that datasets can reference it directly.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_path = out_dir / f"{subject_id}_timeseries.npy"
    meta_path = out_dir / f"{subject_id}_meta.json"

    np.save(ts_path, ts)

    import json

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return ts_path


__all__ = [
    "NiftiLike",
    "load_raw_fmri",
    "extract_roi_timeseries",
    "save_timeseries",
]
