from __future__ import annotations

import pytest

from live.exchange.base import Balance, Order, Position
from live.reconciliation import reconcile, ReconciliationOutcome


def _state(positions=(), open_orders=(), balance_total=10_000.0):
    return {
        "balance": {"total": balance_total, "available": balance_total, "used": 0.0},
        "positions": [_pos_dict(p) for p in positions],
        "open_orders": [_order_dict(o) for o in open_orders],
    }


def _pos(id="P1", side="long", size=0.01) -> Position:
    return Position(
        id=id, symbol="BTC/USDT:USDT", side=side,
        entry_price=100.0, size=size, leverage=3.0,
        unrealized_pnl=0.0, margin=300.0, liquidation_price=80.0,
    )


def _pos_dict(p):
    return {"id": p.id, "side": p.side, "size": p.size}


def _order(id="O1") -> Order:
    return Order(
        id=id, symbol="BTC/USDT:USDT", side="buy", type="limit",
        price=100.0, amount=0.01, status="open",
    )


def _order_dict(o):
    return {"id": o.id, "side": o.side, "amount": o.amount}


def test_resume_clean_when_states_match():
    last_logged = _state(positions=[_pos()], open_orders=[_order()])
    outcome = reconcile(
        last_logged_account_state=last_logged,
        exchange_balance=Balance(10_000.0, 10_000.0, 0.0),
        exchange_positions=[_pos()],
        exchange_orders=[_order()],
    )
    assert outcome.action == "resume"


def test_refuse_when_unknown_position_at_exchange():
    last_logged = _state(positions=[_pos(id="P1")], open_orders=[])
    outcome = reconcile(
        last_logged_account_state=last_logged,
        exchange_balance=Balance(10_000.0, 10_000.0, 0.0),
        exchange_positions=[_pos(id="P1"), _pos(id="P-UNKNOWN")],
        exchange_orders=[],
    )
    assert outcome.action == "refuse"
    assert "unknown" in outcome.diff_notes.lower()


def test_refuse_when_unknown_order_at_exchange():
    last_logged = _state(positions=[], open_orders=[_order(id="O1")])
    outcome = reconcile(
        last_logged_account_state=last_logged,
        exchange_balance=Balance(10_000.0, 10_000.0, 0.0),
        exchange_positions=[],
        exchange_orders=[_order(id="O1"), _order(id="O-NEW")],
    )
    assert outcome.action == "refuse"
    assert "unknown" in outcome.diff_notes.lower()


def test_resume_when_logged_position_no_longer_at_exchange():
    """A position we logged but the exchange doesn't have means it was
    closed (SL/TP/liquidation) while we were down. That's expected; resume."""
    last_logged = _state(positions=[_pos(id="P1")], open_orders=[])
    outcome = reconcile(
        last_logged_account_state=last_logged,
        exchange_balance=Balance(9_500.0, 9_500.0, 0.0),
        exchange_positions=[],
        exchange_orders=[],
    )
    assert outcome.action == "resume"
