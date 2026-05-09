"""Pure observation builder shared between TradingEnv and live runner.

Returns the same Dict observation that TradingEnv.observation_space describes.
No I/O, no DB, no exchange access.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trainer.env.exchange_sim import Order, Position


@dataclass(frozen=True)
class ObservationConfig:
    lookback: int
    num_features: int
    max_open_orders: int
    max_open_positions: int
    max_leverage: float
    initial_balance: float


@dataclass
class ObservationInputs:
    """Snapshot of state at observation time. Caller is responsible for
    normalizing `market` (i.e. applying mean/std)."""
    market: np.ndarray              # shape (lookback, num_features), already normalized
    balance: float
    equity: float
    unrealized_pnl: float
    margin_used: float
    available_balance: float
    open_orders: list[Order]
    open_positions: list[Position]
    close: float                    # current close (for ratio normalization)


def build_observation(
    inputs: ObservationInputs,
    cfg: ObservationConfig,
) -> dict[str, np.ndarray]:
    init = cfg.initial_balance
    close = inputs.close if inputs.close > 0 else 1.0

    account_state = np.array([
        inputs.balance / init,
        inputs.equity / init,
        inputs.unrealized_pnl / init,
        inputs.margin_used / init,
        inputs.available_balance / init,
    ], dtype=np.float32)

    orders = np.zeros((cfg.max_open_orders, 11), dtype=np.float32)
    for i, order in enumerate(inputs.open_orders[:cfg.max_open_orders]):
        orders[i, 0] = 1.0
        orders[i, 1] = float(order.direction)
        orders[i, 2] = order.trigger_price / close
        orders[i, 3] = order.sl_price / close
        for j, tp in enumerate(order.tp_prices[:3]):
            orders[i, 4 + j] = tp / close
        for j, pct in enumerate(order.tp_size_pcts[:3]):
            orders[i, 7 + j] = pct
        orders[i, 10] = order.margin / init

    positions = np.zeros((cfg.max_open_positions, 6), dtype=np.float32)
    for i, pos in enumerate(inputs.open_positions[:cfg.max_open_positions]):
        positions[i, 0] = 1.0
        positions[i, 1] = float(pos.direction)
        positions[i, 2] = pos.entry_price / close
        positions[i, 3] = pos.size * pos.entry_price / init
        positions[i, 4] = pos.unrealized_pnl(close) / init
        positions[i, 5] = pos.leverage / cfg.max_leverage

    return {
        "market": inputs.market.astype(np.float32),
        "account": account_state,
        "orders": orders,
        "positions": positions,
    }
