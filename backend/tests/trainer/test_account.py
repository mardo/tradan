import pytest

from trainer.env.account import Account


def test_initial_state():
    acc = Account(initial_balance=10_000.0)
    assert acc.balance == 10_000.0
    assert acc.margin_used == 0.0
    assert acc.available_balance == 10_000.0


def test_allocate_margin():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(1_000.0)
    assert acc.margin_used == 1_000.0
    assert acc.available_balance == 9_000.0


def test_allocate_margin_insufficient():
    acc = Account(initial_balance=10_000.0)
    with pytest.raises(ValueError, match="Insufficient"):
        acc.allocate_margin(11_000.0)


def test_release_margin():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(2_000.0)
    acc.release_margin(500.0)
    assert acc.margin_used == 1_500.0
    assert acc.available_balance == 8_500.0


def test_realize_pnl_profit():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(1_000.0)
    acc.realize_pnl(200.0)
    assert acc.balance == 10_200.0
    assert acc.margin_used == 1_000.0


def test_realize_pnl_loss():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(1_000.0)
    acc.realize_pnl(-300.0)
    assert acc.balance == 9_700.0


def test_apply_fee():
    acc = Account(initial_balance=10_000.0)
    acc.apply_fee(50.0)
    assert acc.balance == 9_950.0


def test_equity_with_unrealized():
    acc = Account(initial_balance=10_000.0)
    assert acc.equity(unrealized_pnl=500.0) == 10_500.0
    assert acc.equity(unrealized_pnl=-200.0) == 9_800.0


def test_reset():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(2_000.0)
    acc.realize_pnl(500.0)
    acc.reset()
    assert acc.balance == 10_000.0
    assert acc.margin_used == 0.0
