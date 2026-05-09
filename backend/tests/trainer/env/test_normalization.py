from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trainer.env.normalization import (
    NormalizationStats,
    fit_stats,
    load_stats,
    save_stats,
)


def test_fit_stats_returns_per_feature_mean_std():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(1000, 7)).astype(np.float32)
    stats = fit_stats(features)
    assert stats.mean.shape == (7,)
    assert stats.std.shape == (7,)
    np.testing.assert_allclose(stats.mean, features.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(stats.std, features.std(axis=0), atol=1e-6)


def test_fit_stats_replaces_zero_std_with_one():
    features = np.ones((100, 3), dtype=np.float32)
    stats = fit_stats(features)
    np.testing.assert_array_equal(stats.std, np.ones(3, dtype=np.float32))


def test_save_and_load_roundtrip(tmp_path: Path):
    stats = NormalizationStats(
        mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        std=np.array([0.5, 1.0, 2.0], dtype=np.float32),
    )
    save_stats(stats, tmp_path / "model")
    loaded = load_stats(tmp_path / "model")
    np.testing.assert_array_equal(loaded.mean, stats.mean)
    np.testing.assert_array_equal(loaded.std, stats.std)


def test_load_stats_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_stats(tmp_path / "nope")
