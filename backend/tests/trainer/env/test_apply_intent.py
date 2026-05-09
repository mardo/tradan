from __future__ import annotations

import pytest

from trainer.config import ExchangeConfig
from trainer.env.account import Account
from trainer.env.action_decoder import CloseIntent, OpenIntent, OrderIntent
from trainer.env.exchange_sim import ExchangeSim


def _sim() -> ExchangeSim:
    return ExchangeSim(config=ExchangeConfig(), account=Account(initial_balance=10_000.0))


def test_apply_intent_opens_order_when_intent_has_open():
    sim = _sim()
    intent = OrderIntent(
        open=OpenIntent(
            direction=1, trigger_price=100.0, sl_price=99.0,
            tp_prices=[101.0, 102.0, 103.0],
            tp_size_pcts=[1/3, 1/3, 1/3], margin=500.0,
        ),
        cancels=[], closes=[],
    )
    info = sim.apply_intent(intent, current_price=100.0)
    assert info["orders_placed"] == 1
    assert len(sim.open_orders) == 1


def test_apply_intent_cancels_in_descending_order():
    sim = _sim()
    # Place 3 orders so we have something to cancel
    for _ in range(3):
        sim.place_order(
            direction=1, trigger_price=100.0, sl_price=99.0,
            tp_prices=[101.0, 102.0, 103.0],
            tp_size_pcts=[1/3, 1/3, 1/3], margin=500.0,
        )
    intent = OrderIntent(open=None, cancels=[0, 2], closes=[])
    info = sim.apply_intent(intent, current_price=100.0)
    assert info["orders_cancelled"] == 2
    assert len(sim.open_orders) == 1


def test_apply_intent_returns_zero_counts_for_empty_intent():
    sim = _sim()
    info = sim.apply_intent(OrderIntent(open=None, cancels=[], closes=[]), current_price=100.0)
    assert info == {"orders_placed": 0, "orders_cancelled": 0, "positions_closed": 0}
