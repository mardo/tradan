from __future__ import annotations

import numpy as np

from live.exchange.replay import ReplayAdapter


def test_fetch_klines_returns_recent_window():
    timestamps = np.arange(0, 200, dtype=np.int64) * 60_000
    features = np.column_stack([
        np.linspace(100, 120, 200),     # open
        np.linspace(101, 121, 200),     # high
        np.linspace(99, 119, 200),      # low
        np.linspace(100, 120, 200),     # close
        np.full(200, 1000.0),           # volume
    ]).astype(np.float32)
    price_columns = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}

    adapter = ReplayAdapter.from_arrays(
        timestamps=timestamps,
        features=features,
        price_columns=price_columns,
        symbol="BTC/USDT:USDT",
        interval="4h",
        starting_balance=10_000.0,
    )
    # Advance once so cursor > 0 — fetch_klines returns klines [0:cursor].
    adapter.advance()
    klines = adapter.fetch_klines("BTC/USDT:USDT", "4h", limit=50)
    # We've advanced once, so cursor=1; fetching limit=50 returns klines[max(0,1-50):1] = klines[0:1].
    assert len(klines) == 1
    assert klines[0].open_time_ms == int(timestamps[0])


def test_advance_step_processes_candle():
    timestamps = np.arange(0, 50, dtype=np.int64)
    features = np.tile(
        np.array([100.0, 101.0, 99.0, 100.0, 1.0]), (50, 1),
    ).astype(np.float32)
    price_columns = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}

    adapter = ReplayAdapter.from_arrays(
        timestamps=timestamps, features=features, price_columns=price_columns,
        symbol="BTC/USDT:USDT", interval="4h", starting_balance=10_000.0,
    )
    initial_balance = adapter.fetch_balance()
    assert initial_balance.available == 10_000.0
    adapter.advance()
    after = adapter.fetch_balance()
    # No orders/positions; balance unchanged.
    assert after.available == 10_000.0
