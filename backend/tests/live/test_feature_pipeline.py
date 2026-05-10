from __future__ import annotations

import numpy as np
import pytest

from live.exchange.base import Balance, Kline
from live.feature_pipeline import build_live_observation, klines_to_features
from trainer.env.observation import ObservationConfig


_TRAINER_DEFAULT_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
]


def _full_kline(t: int = 0) -> Kline:
    return Kline(
        open_time_ms=t,
        open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0,
        quote_volume=15.0, num_trades=42,
        taker_buy_base_vol=4.0, taker_buy_quote_vol=6.0,
    )


def test_klines_to_features_handles_all_9_columns():
    arr = klines_to_features([_full_kline(0), _full_kline(60_000)], _TRAINER_DEFAULT_COLUMNS)
    assert arr.shape == (2, 9)
    np.testing.assert_allclose(arr[0], [1.0, 2.0, 0.5, 1.5, 10.0, 15.0, 42.0, 4.0, 6.0])


def test_klines_to_features_handles_ohlcv_subset():
    k = Kline(open_time_ms=0, open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0)
    arr = klines_to_features([k], ["open", "high", "low", "close", "volume"])
    assert arr.shape == (1, 5)


def test_klines_to_features_raises_for_missing_extended_column():
    k = Kline(open_time_ms=0, open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0)
    with pytest.raises(ValueError):
        klines_to_features([k], _TRAINER_DEFAULT_COLUMNS)


def test_klines_to_features_raises_for_unknown_column():
    k = _full_kline()
    with pytest.raises(KeyError):
        klines_to_features([k], ["open", "weather"])


def _make_obs_cfg(lookback: int = 50, num_features: int = 5) -> ObservationConfig:
    return ObservationConfig(
        lookback=lookback, num_features=num_features,
        max_open_orders=20, max_open_positions=20,
        max_leverage=125.0, initial_balance=10_000.0,
    )


def _make_klines(n: int, close: float = 100.0) -> list[Kline]:
    return [
        Kline(
            open_time_ms=i * 60_000,
            open=close, high=close + 1.0, low=close - 1.0,
            close=close, volume=1.0,
        )
        for i in range(n)
    ]


def test_build_live_observation_fits_stats_from_lookback_when_no_context():
    """Without context_features, stats are fit from the kline window itself.
    Verifies the live (no-context) code path."""
    cols = ["open", "high", "low", "close", "volume"]
    klines = [
        Kline(open_time_ms=i * 60_000,
              open=100.0 + i, high=101.0 + i, low=99.0 + i,
              close=100.0 + i, volume=1.0)
        for i in range(50)
    ]
    obs_cfg = _make_obs_cfg(lookback=50, num_features=5)
    obs = build_live_observation(
        klines=klines, columns=cols,
        balance=Balance(10_000.0, 10_000.0, 0.0),
        positions=[], open_orders=[],
        obs_cfg=obs_cfg,
    )
    # market is normalized from window stats — mean across rows should be ~0
    np.testing.assert_allclose(obs["market"].mean(axis=0), np.zeros(5), atol=1e-6)


def test_build_live_observation_uses_context_when_provided():
    """With context_features, stats are fit from the context, not the klines.
    Verifies the replay (passes full historical context) code path."""
    cols = ["open", "high", "low", "close", "volume"]
    rng = np.random.default_rng(0)
    # Big context (200 rows) with wide range
    context = rng.normal(loc=100.0, scale=30.0, size=(200, 5)).astype(np.float32)
    # Small kline window (50 rows) with narrow range (constant ~100)
    klines = _make_klines(50, close=100.0)
    obs_cfg = _make_obs_cfg(lookback=50, num_features=5)
    obs = build_live_observation(
        klines=klines, columns=cols,
        balance=Balance(10_000.0, 10_000.0, 0.0),
        positions=[], open_orders=[],
        obs_cfg=obs_cfg,
        context_features=context,
    )
    # With wide-stddev context, the constant ~100 kline window should
    # normalize to a small magnitude (close to 0).
    market = obs["market"]
    assert abs(market.mean()) < 1.0  # demonstrates context controlled normalization
