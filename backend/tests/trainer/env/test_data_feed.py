from __future__ import annotations

import numpy as np

from trainer.env.data_feed import DataFeed
from trainer.env.normalization import NormalizationStats


def test_data_feed_uses_provided_stats():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3)).astype(np.float32)
    timestamps = np.arange(200, dtype=np.int64)

    custom_stats = NormalizationStats(
        mean=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        std=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    feed = DataFeed(
        timestamps=timestamps,
        features=features,
        lookback=10,
        stats=custom_stats,
    )

    obs = feed.get_observation(0)
    expected = (features[:10] - custom_stats.mean) / custom_stats.std
    np.testing.assert_allclose(obs, expected.astype(np.float32), atol=1e-6)


def test_data_feed_fits_stats_when_none_provided():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(200, 3)).astype(np.float32)
    timestamps = np.arange(200, dtype=np.int64)

    feed = DataFeed(timestamps=timestamps, features=features, lookback=10)

    # Stats should match fit_stats() output.
    np.testing.assert_allclose(feed.stats.mean, features.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(feed.stats.std, features.std(axis=0), atol=1e-6)
