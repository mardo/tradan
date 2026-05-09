"""Per-feature mean/std persistence for DataFeed normalization.

Computed once at training time from the training feature array, then reused
at eval and live inference so the model sees the same input distribution it
was trained against. Stored as `<model>.mean.npy` and `<model>.std.npy` next
to the SB3 model `.zip` file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray  # shape (num_features,), dtype float32
    std: np.ndarray   # shape (num_features,), dtype float32, no zeros


def fit_stats(features: np.ndarray) -> NormalizationStats:
    """Compute per-feature mean and std; clamp zero std to 1.0."""
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return NormalizationStats(mean=mean, std=std)


def save_stats(stats: NormalizationStats, model_path_no_ext: Path) -> None:
    """Save mean.npy and std.npy next to a model file (without extension)."""
    base = Path(model_path_no_ext)
    base.parent.mkdir(parents=True, exist_ok=True)
    np.save(base.with_suffix(".mean.npy"), stats.mean)
    np.save(base.with_suffix(".std.npy"), stats.std)


def load_stats(model_path_no_ext: Path) -> NormalizationStats:
    """Load saved stats. Raises FileNotFoundError if either file missing."""
    base = Path(model_path_no_ext)
    mean_path = base.with_suffix(".mean.npy")
    std_path = base.with_suffix(".std.npy")
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {mean_path}, {std_path}"
        )
    return NormalizationStats(
        mean=np.load(mean_path).astype(np.float32),
        std=np.load(std_path).astype(np.float32),
    )
