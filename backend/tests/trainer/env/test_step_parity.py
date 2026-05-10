"""Drift-detection regression test for TradingEnv.

Runs the env on a small synthetic kline window with a deterministic
action sequence; asserts the equity series matches a frozen snapshot.

If this test fails, the refactor has changed observable behavior. Either
the change is intentional (update the snapshot) or it's a bug (revert).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trainer.config import ExchangeConfig, ModelConfig
from trainer.env.data_feed import DataFeed
from trainer.env.trading_env import TradingEnv


SNAPSHOT_PATH = Path(__file__).parent / "step_parity_snapshot.json"


def _build_env() -> TradingEnv:
    n = 200
    lookback = 50
    # Synthetic OHLCV with a gentle uptrend, deterministic.
    closes = 100.0 + np.arange(n, dtype=np.float32) * 0.1
    highs = closes + 1.0
    lows = closes - 1.0
    opens = closes - 0.05
    volumes = np.full(n, 1000.0, dtype=np.float32)
    features = np.column_stack([opens, highs, lows, closes, volumes]).astype(np.float32)
    timestamps = (np.arange(n, dtype=np.int64) * 4 * 3600 * 1000)

    feed = DataFeed(
        timestamps=timestamps,
        features=features,
        lookback=lookback,
        price_columns={"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4},
    )
    cfg = ModelConfig(
        name="parity",
        symbols=["BTCUSDT"],
        intervals=["4h"],
        columns=["open", "high", "low", "close", "volume"],
        exchange=ExchangeConfig(max_open_orders=5, max_open_positions=5),
        lookback_window=lookback,
        initial_balance=10_000.0,
        num_tp_levels=3,
    )
    return TradingEnv(cfg, feed)


def _action_sequence(env: TradingEnv) -> list[np.ndarray]:
    """Deterministic action sequence: alternate open + no-op."""
    size = env.action_space.shape[0]
    actions = []
    for i in range(env.data_feed.total_steps):
        a = np.full(size, -1.0, dtype=np.float32)
        if i % 5 == 0:
            # open: confidence > 0.5, long, conservative SL/margin
            a[0] = 1.0   # open confidence
            a[1] = 0.5   # direction (>0 → long)
            a[2] = 0.0   # trigger offset
            a[3] = -0.5  # sl_pct (maps to ~min sl)
            # tp levels: indices 4, 5, 6 (num_tp_levels=3)
            a[4] = 0.3
            a[5] = 0.5
            a[6] = 0.7
            # tp weights: indices 7, 8, 9
            a[7] = 0.33
            a[8] = 0.33
            a[9] = 0.34
            # margin: index 10
            a[10] = -0.5  # margin ~0.25 * available
        actions.append(a)
    return actions


def _run_and_collect(env: TradingEnv) -> list[float]:
    obs, _ = env.reset()
    equities: list[float] = []
    for action in _action_sequence(env):
        obs, _r, terminated, truncated, _info = env.step(action)
        equities.append(env.pnl_history[-1]["equity"])
        if terminated or truncated:
            break
    return equities


def test_step_parity_matches_snapshot():
    env = _build_env()
    equities = _run_and_collect(env)

    if not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.write_text(json.dumps({"equities": [float(e) for e in equities]}, indent=2))
        pytest.skip(
            f"snapshot created at {SNAPSHOT_PATH}; commit it and re-run to enable parity check"
        )

    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    expected = snapshot["equities"]
    assert len(equities) == len(expected), (
        f"length mismatch: got {len(equities)}, snapshot {len(expected)}"
    )
    np.testing.assert_allclose(equities, expected, rtol=1e-6, atol=1e-6)
