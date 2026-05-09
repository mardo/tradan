from __future__ import annotations

import numpy as np
import pytest

from live.exchange.base import Kline
from live.feature_pipeline import klines_to_features


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
