"""Tests for the Phase 4 env risk-control caps.

Each fix is independently testable so we can A/B them later.
"""
from __future__ import annotations

import numpy as np
import pytest

from trainer.config import ExchangeConfig, ModelConfig
from trainer.env.account import Account
from trainer.env.data_feed import DataFeed
from trainer.env.exchange_sim import ExchangeSim
from trainer.env.trading_env import TradingEnv


# ---------- Fix 1: max_leverage default -------------------------------------


def test_exchange_config_default_max_leverage_is_10():
    """ExchangeConfig() with no overrides must cap leverage at 10x by default.

    The previous default (125.0) auto-leveraged any tight stop-loss to ~100x,
    which is the proximate cause of the >100% drawdowns seen in 4A and 4C.
    """
    cfg = ExchangeConfig()
    assert cfg.max_leverage == 10.0


def test_compute_leverage_clamps_to_default_max():
    """A tight SL that would compute >10x leverage must be clamped to 10."""
    sim = ExchangeSim(config=ExchangeConfig(), account=Account(initial_balance=10_000))
    # SL=0.1% would yield ~100x via the formula; expect clamp at 10.
    leverage = sim.compute_leverage(entry_price=50_000.0, sl_price=49_950.0, direction=1)
    assert leverage == pytest.approx(10.0)


# ---------- Fix 2: max_position_size_pct ------------------------------------


@pytest.fixture
def minimal_env() -> TradingEnv:
    """A TradingEnv with a tiny synthetic DataFeed, just enough to call step()
    and exercise _process_actions. 2 orders / 2 positions max keeps the action
    vector small; 3 TP levels matches the env default."""
    n_candles = 60
    rng = np.random.default_rng(0)
    closes = 50_000.0 + rng.normal(0, 100, n_candles).cumsum()
    features = np.stack(
        [
            closes,                                          # open
            closes + 50.0,                                   # high
            closes - 50.0,                                   # low
            closes,                                          # close
            np.full(n_candles, 1_000.0),                     # volume
        ],
        axis=1,
    ).astype(np.float32)
    feed = DataFeed(
        timestamps=np.arange(n_candles, dtype=np.int64),
        features=features,
        lookback=10,
        price_columns={"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4},
    )
    config = ModelConfig(
        name="risk_caps_test",
        symbols=["BTCUSDT"],
        intervals=["4h"],
        columns=["open", "high", "low", "close", "volume"],
        exchange=ExchangeConfig(max_open_orders=2, max_open_positions=2),
        lookback_window=10,
        num_tp_levels=3,
    )
    return TradingEnv(config=config, data_feed=feed)


def _make_open_action(env: TradingEnv, *, size_raw: float) -> np.ndarray:
    """Build an action vector that requests opening a long position with the
    given size_raw (in [0, 1]). Other action components are neutral."""
    a = np.zeros(env.config.action_size, dtype=np.float32)
    a[0] = 1.0   # open_conf > 0.5
    a[1] = 1.0   # direction long
    a[2] = 0.0   # offset 0
    a[3] = 0.0   # mid SL (neutral)
    # TP prices and sizes default to neutral (zero) — env clamps to safe values
    # action[4..4+num_tp-1] = tp prices
    # action[4+num_tp..4+2*num_tp-1] = tp sizes
    size_idx = 4 + 2 * env.config.num_tp_levels
    a[size_idx] = float(size_raw * 2.0 - 1.0)  # invert (action+1)/2 mapping
    return a


def test_position_size_capped_to_max_position_size_pct(minimal_env: TradingEnv):
    """size_raw=1.0 must NOT result in a position margined at full available_balance.
    The env caps it at max_position_size_pct * available_balance."""
    env = minimal_env
    env.reset()
    initial_avail = env.account.available_balance
    expected_cap = env.config.exchange.max_position_size_pct * initial_avail

    action = _make_open_action(env, size_raw=1.0)  # request full balance
    env.step(action)

    assert len(env.exchange.open_orders) == 1, (
        "expected the order to be placed (at the cap, not rejected)"
    )
    placed_margin = env.exchange.open_orders[0].margin
    # The cap clamps margin to max_position_size_pct * available_balance.
    # Equality with a small tolerance (fees may be applied to balance first).
    # rel=1e-5 absorbs float32 round-trip noise from the action vector.
    assert placed_margin == pytest.approx(expected_cap, rel=1e-5), (
        f"expected margin {expected_cap}, got {placed_margin}"
    )


def test_position_size_below_cap_is_unchanged(minimal_env: TradingEnv):
    """size_raw=0.1 (below the 25% cap) must place at exactly that requested
    fraction — the cap doesn't squeeze trades that are already small."""
    env = minimal_env
    env.reset()
    initial_avail = env.account.available_balance

    action = _make_open_action(env, size_raw=0.1)
    env.step(action)

    assert len(env.exchange.open_orders) == 1
    placed_margin = env.exchange.open_orders[0].margin
    assert placed_margin == pytest.approx(0.1 * initial_avail, rel=1e-5)


def test_exchange_config_default_max_position_size_pct_is_quarter():
    cfg = ExchangeConfig()
    assert cfg.max_position_size_pct == 0.25


# ---------- Fix 3: max_drawdown_pct -----------------------------------------


def test_exchange_config_default_max_drawdown_pct_is_half():
    cfg = ExchangeConfig()
    assert cfg.max_drawdown_pct == 0.5


def test_step_terminates_when_equity_drops_below_drawdown_threshold(
    minimal_env: TradingEnv,
):
    """If equity falls to peak * (1 - max_drawdown_pct), the env must
    terminate the episode. This gives the policy a sharp negative signal
    well before the account hits zero."""
    env = minimal_env
    assert env.config.exchange.max_drawdown_pct == 0.5
    env.reset()
    # Set the peak high so the threshold is unambiguous.
    env._peak_equity = 10_000.0
    # Drain account directly to simulate a large loss.
    env.account.balance = 4_999.0  # equity will be 4999, below threshold 5000
    no_op = np.zeros(env.config.action_size, dtype=np.float32)
    no_op[0] = -1.0  # open_conf < 0.5 so no order placed
    _, _, terminated, _, _ = env.step(no_op)
    assert terminated is True


def test_step_does_not_terminate_just_above_drawdown_threshold(
    minimal_env: TradingEnv,
):
    """Equity exactly at the threshold should not yet terminate."""
    env = minimal_env
    env.reset()
    env._peak_equity = 10_000.0
    env.account.balance = 5_001.0  # just above 50% drawdown
    no_op = np.zeros(env.config.action_size, dtype=np.float32)
    no_op[0] = -1.0
    _, _, terminated, _, _ = env.step(no_op)
    assert terminated is False


def test_peak_equity_is_trailing(minimal_env: TradingEnv):
    """peak_equity must rise with new highs but never fall, so the threshold
    is a TRAILING drawdown rather than initial-balance-based."""
    env = minimal_env
    env.reset()
    initial_peak = env._peak_equity
    # Simulate the account growing.
    env.account.balance = 20_000.0
    no_op = np.zeros(env.config.action_size, dtype=np.float32)
    no_op[0] = -1.0
    env.step(no_op)
    assert env._peak_equity == pytest.approx(20_000.0)
    # Then shrinking (but not enough to cross the new threshold of 10_000).
    env.account.balance = 11_000.0
    env.step(no_op)
    # Peak must not retreat.
    assert env._peak_equity == pytest.approx(20_000.0)
    assert env._peak_equity > initial_peak


# ---------- Idle-step penalty (Phase 4E) ------------------------------------


def test_exchange_config_default_idle_step_penalty_is_zero():
    """The penalty defaults to 0 so existing configs (4A/4C/4D) are unaffected.
    Phase 4E sweeps opt in by setting it to a positive value."""
    cfg = ExchangeConfig()
    assert cfg.idle_step_penalty_usd == 0.0


def test_exchange_config_idle_step_penalty_round_trip():
    cfg = ExchangeConfig(idle_step_penalty_usd=0.5)
    d = cfg.to_dict()
    assert d["idle_step_penalty_usd"] == 0.5
    restored = ExchangeConfig.from_dict(d)
    assert restored.idle_step_penalty_usd == 0.5


def test_exchange_config_idle_step_penalty_omitted_in_legacy_dict():
    """Old persisted ExchangeConfig JSON predating this field must round-trip
    without crashing (from_dict already filters by __dataclass_fields__)."""
    legacy = {"max_leverage": 10.0}
    restored = ExchangeConfig.from_dict(legacy)
    assert restored.idle_step_penalty_usd == 0.0


def test_idle_step_penalty_subtracted_when_no_positions_or_orders(
    minimal_env: TradingEnv,
):
    """With penalty=1.0 and an empty book, step() must subtract 1.0 from the
    reward computed as Δ equity. The penalty is additive on top of the existing
    reward formula — never replaces it."""
    env = minimal_env
    env.config.exchange.idle_step_penalty_usd = 1.0
    env.reset()
    # No positions, no orders, no equity change: reward should be -1.0.
    no_op = np.zeros(env.config.action_size, dtype=np.float32)
    no_op[0] = -1.0  # open_conf < 0.5 → no order placed this step
    _, reward, _, _, _ = env.step(no_op)
    assert reward == pytest.approx(-1.0)


def test_idle_step_penalty_not_applied_when_orders_open(
    minimal_env: TradingEnv,
):
    """If the model has live orders (or positions), it is not idle and the
    penalty must not fire. Place an order on step 1 so the exchange's order
    book is non-empty; step 2 with a no-op must NOT subtract the penalty."""
    env = minimal_env
    env.config.exchange.idle_step_penalty_usd = 1.0
    env.reset()
    open_action = _make_open_action(env, size_raw=0.1)
    env.step(open_action)
    assert len(env.exchange.open_orders) >= 1, (
        "expected an open order to be present so the next step is non-idle"
    )
    prev_equity = env._prev_equity
    no_op = np.zeros(env.config.action_size, dtype=np.float32)
    no_op[0] = -1.0
    _, reward, _, _, _ = env.step(no_op)
    expected = float(env._prev_equity - prev_equity)
    assert reward == pytest.approx(expected)


def test_idle_step_penalty_default_zero_does_not_change_reward(
    minimal_env: TradingEnv,
):
    """Default idle_step_penalty_usd=0 must leave reward exactly equal to
    Δ equity, even when the env is idle. Guards against accidentally enabling
    the penalty for any non-4E sweep."""
    env = minimal_env
    assert env.config.exchange.idle_step_penalty_usd == 0.0
    env.reset()
    no_op = np.zeros(env.config.action_size, dtype=np.float32)
    no_op[0] = -1.0
    _, reward, _, _, _ = env.step(no_op)
    assert reward == pytest.approx(0.0)
