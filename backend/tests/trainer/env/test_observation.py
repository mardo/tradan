from __future__ import annotations

import numpy as np

from trainer.config import ExchangeConfig
from trainer.env.observation import (
    ObservationConfig,
    ObservationInputs,
    build_observation,
)
from trainer.env.exchange_sim import Order, Position


def _cfg() -> ObservationConfig:
    return ObservationConfig(
        lookback=50,
        num_features=7,
        max_open_orders=20,
        max_open_positions=20,
        max_leverage=125.0,
        initial_balance=10_000.0,
    )


def test_build_observation_shape_and_keys():
    cfg = _cfg()
    market = np.zeros((50, 7), dtype=np.float32)
    inputs = ObservationInputs(
        market=market,
        balance=10_000.0, equity=10_000.0, unrealized_pnl=0.0,
        margin_used=0.0, available_balance=10_000.0,
        open_orders=[], open_positions=[], close=100.0,
    )
    obs = build_observation(inputs, cfg)
    assert set(obs.keys()) == {"market", "account", "orders", "positions"}
    assert obs["market"].shape == (50, 7)
    assert obs["account"].shape == (5,)
    assert obs["orders"].shape == (20, 11)
    assert obs["positions"].shape == (20, 6)
    assert obs["market"].dtype == np.float32
    assert obs["account"].dtype == np.float32


def test_build_observation_account_state_normalized_by_initial_balance():
    cfg = _cfg()
    inputs = ObservationInputs(
        market=np.zeros((50, 7), dtype=np.float32),
        balance=8_000.0, equity=12_000.0, unrealized_pnl=4_000.0,
        margin_used=2_000.0, available_balance=6_000.0,
        open_orders=[], open_positions=[], close=100.0,
    )
    obs = build_observation(inputs, cfg)
    np.testing.assert_allclose(
        obs["account"],
        np.array([0.8, 1.2, 0.4, 0.2, 0.6], dtype=np.float32),
    )


def test_build_observation_encodes_open_order_row():
    cfg = _cfg()
    order = Order(
        id=0, direction=1, trigger_price=100.0, sl_price=98.0,
        tp_prices=[105.0, 110.0, 115.0],
        tp_size_pcts=[0.5, 0.3, 0.2], margin=500.0,
    )
    inputs = ObservationInputs(
        market=np.zeros((50, 7), dtype=np.float32),
        balance=10_000.0, equity=10_000.0, unrealized_pnl=0.0,
        margin_used=500.0, available_balance=9_500.0,
        open_orders=[order], open_positions=[], close=100.0,
    )
    obs = build_observation(inputs, cfg)
    row = obs["orders"][0]
    assert row[0] == 1.0
    assert row[1] == 1.0                        # direction
    assert row[2] == 1.0                        # trigger / close = 1.0
    assert row[3] == 0.98                       # sl / close
    np.testing.assert_allclose(row[4:7], [1.05, 1.10, 1.15], atol=1e-6)
    np.testing.assert_allclose(row[7:10], [0.5, 0.3, 0.2], atol=1e-6)
    assert row[10] == 0.05                      # margin / initial_balance
