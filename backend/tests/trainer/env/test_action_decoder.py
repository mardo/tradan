from __future__ import annotations

import numpy as np
import pytest

from trainer.config import ExchangeConfig, ModelConfig
from trainer.env.action_decoder import (
    DecoderState,
    OpenIntent,
    OrderIntent,
    decode_action,
)


def _cfg() -> ModelConfig:
    return ModelConfig(
        name="t",
        symbols=["BTCUSDT"],
        intervals=["4h"],
        num_tp_levels=3,
        exchange=ExchangeConfig(max_open_orders=20, max_open_positions=20),
    )


def _state(close: float = 100.0, available: float = 1000.0,
           num_open_orders: int = 0, num_open_positions: int = 0) -> DecoderState:
    return DecoderState(
        close=close, available_balance=available,
        num_open_orders=num_open_orders, num_open_positions=num_open_positions,
    )


def _zero_action(cfg: ModelConfig) -> np.ndarray:
    size = (
        1 + 1 + 1 + 1
        + cfg.num_tp_levels + cfg.num_tp_levels + 1
        + cfg.exchange.max_open_orders + cfg.exchange.max_open_positions
    )
    # All -1 → no opens (open_conf = 0), no cancels, no closes (frac = 0)
    return np.full(size, -1.0, dtype=np.float32)


def test_decode_action_no_open_when_open_confidence_below_threshold():
    cfg = _cfg()
    action = _zero_action(cfg)
    intent = decode_action(action, _state(), cfg)
    assert intent.open is None
    assert intent.cancels == []
    assert intent.closes == []


def test_decode_action_emits_open_with_long_direction():
    cfg = _cfg()
    action = _zero_action(cfg)
    action[0] = 1.0      # open_conf = 1.0 > 0.5 → open
    action[1] = 1.0      # direction = +1 (long)
    action[2] = 0.0      # trigger offset = 0 → trigger == close
    action[3] = 0.0      # SL distance: action=0 → sl_raw=(0+1)/2=0.5 → midpoint
    # Default TP distances/sizes (=-1 → all 0 → clamped to 0.001 dist)
    action[4 + 2 * cfg.num_tp_levels] = 1.0  # margin = available_balance

    intent = decode_action(action, _state(close=100.0, available=1000.0), cfg)
    assert intent.open is not None
    assert intent.open.direction == 1
    assert intent.open.trigger_price == pytest.approx(100.0)
    # sl_raw=0.5: min_sl_pct=0.1, max_sl_pct=10.0 → 0.1 + 0.5*9.9 = 5.05% below 100
    assert intent.open.sl_price == pytest.approx(100.0 * (1 - 0.0505), rel=1e-3)
    assert intent.open.margin == pytest.approx(1000.0)


def test_decode_action_picks_indices_to_cancel():
    cfg = _cfg()
    action = _zero_action(cfg)
    cancel_start = 1 + 1 + 1 + 1 + cfg.num_tp_levels + cfg.num_tp_levels + 1
    action[cancel_start + 0] = 0.5     # cancel index 0
    action[cancel_start + 3] = 0.7     # cancel index 3

    intent = decode_action(action, _state(num_open_orders=5), cfg)
    assert intent.cancels == [0, 3]


def test_decode_action_emits_close_intents_above_threshold():
    cfg = _cfg()
    action = _zero_action(cfg)
    close_start = (
        1 + 1 + 1 + 1
        + cfg.num_tp_levels + cfg.num_tp_levels + 1
        + cfg.exchange.max_open_orders
    )
    # close index 1 with frac > 0.05: action ∈ [-1,1] → frac=(a+1)/2; need (a+1)/2 > 0.05 → a > -0.9
    action[close_start + 1] = 0.0   # → frac=0.5

    intent = decode_action(action, _state(num_open_positions=3), cfg)
    assert len(intent.closes) == 1
    assert intent.closes[0].position_index == 1
    assert intent.closes[0].fraction == pytest.approx(0.5)


def test_decode_action_skips_open_below_min_order_size():
    cfg = _cfg()
    action = _zero_action(cfg)
    action[0] = 1.0
    action[1] = 1.0
    action[4 + 2 * cfg.num_tp_levels] = -0.99   # margin ≈ 0.005 * available
    intent = decode_action(action, _state(close=100.0, available=1000.0), cfg)
    # margin = 0.005 * 1000 = 5; min_order_size_usd = 10 → no open
    assert intent.open is None
