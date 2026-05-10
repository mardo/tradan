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
#
# Implemented in trading_env._process_actions: margin is clamped to
# max_position_size_pct * available_balance regardless of the action vector.


# ---------- Fix 3: max_drawdown_pct -----------------------------------------
#
# Implemented in trading_env.step(): trailing peak_equity tracked; episode
# terminates when equity drops below peak * (1 - max_drawdown_pct).
