import numpy as np
import pytest

from trainer.env.data_feed import DataFeed


@pytest.fixture
def sample_data() -> dict:
    n = 600
    timestamps = list(range(1000, 1000 + n))
    base = np.random.default_rng(42)
    prices = 50000 + base.normal(0, 100, n).cumsum()
    return {
        "timestamps": np.array(timestamps, dtype=np.int64),
        "features": np.column_stack([
            prices,
            prices + base.uniform(10, 200, n),
            prices - base.uniform(10, 200, n),
            prices + base.normal(0, 50, n),
            base.uniform(100, 1000, n),
        ]),
        "columns": ["open", "high", "low", "close", "volume"],
    }


def test_feed_shape(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
    )
    assert feed.total_steps == 100
    obs = feed.get_observation(0)
    assert obs.shape == (500, 5)


def test_feed_normalization(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
    )
    obs = feed.get_observation(0)
    for col in range(obs.shape[1]):
        assert abs(obs[:, col].mean()) < 1.0
        assert obs[:, col].std() < 5.0


def test_feed_stepping(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
    )
    obs0 = feed.get_observation(0)
    obs1 = feed.get_observation(1)
    assert not np.array_equal(obs0, obs1)
    np.testing.assert_array_equal(obs0[1:, :], obs1[:-1, :])


def test_get_candle_prices(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
        price_columns={"open": 0, "high": 1, "low": 2, "close": 3},
    )
    prices = feed.get_candle_prices(0)
    assert "open" in prices
    assert "high" in prices
    assert "low" in prices
    assert "close" in prices


def test_get_timestamp(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
    )
    ts = feed.get_timestamp(0)
    assert ts == sample_data["timestamps"][500]
