"""Pure action decoder shared between TradingEnv and live runner.

Translates the model's float vector (shape (action_size,), values in [-1, 1])
into a structured OrderIntent. No I/O, no mutation, no exchange access.

Used by:
- trainer.env.trading_env.TradingEnv.step (training/eval)
- live.action_decoder.to_exchange_intent (production)

Both code paths must produce identical intents from identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trainer.config import ModelConfig


@dataclass(frozen=True)
class DecoderState:
    """Snapshot of state needed to decode an action."""
    close: float                # current close price
    available_balance: float    # for margin sizing
    num_open_orders: int        # to truncate cancel signals
    num_open_positions: int     # to truncate close signals


@dataclass(frozen=True)
class OpenIntent:
    direction: int                  # +1 long, -1 short
    trigger_price: float
    sl_price: float
    tp_prices: list[float]
    tp_size_pcts: list[float]       # sums to 1.0
    margin: float                   # in account currency (USD)


@dataclass(frozen=True)
class CloseIntent:
    position_index: int
    fraction: float                 # 0 < fraction <= 1


@dataclass(frozen=True)
class OrderIntent:
    open: OpenIntent | None
    cancels: list[int]              # indices into open_orders
    closes: list[CloseIntent]


def _action_size(cfg: ModelConfig) -> int:
    n_tp = cfg.num_tp_levels
    exc = cfg.exchange
    return 1 + 1 + 1 + 1 + n_tp + n_tp + 1 + exc.max_open_orders + exc.max_open_positions


def decode_action(
    action: np.ndarray,
    state: DecoderState,
    cfg: ModelConfig,
) -> OrderIntent:
    """Translate a 51-float action vector into a structured OrderIntent.

    Layout (matches the original TradingEnv._process_actions):
      [0]                        : open confidence (-1..1; >0 → open)
      [1]                        : direction (+/-)
      [2]                        : trigger price offset
      [3]                        : SL distance
      [4 .. 4+n_tp-1]            : TP distances
      [4+n_tp .. 4+2*n_tp-1]     : TP size weights
      [4+2*n_tp]                 : margin size
      [next max_open_orders]     : cancel signals
      [next max_open_positions]  : close fractions
    """
    expected = _action_size(cfg)
    if action.shape[0] != expected:
        raise ValueError(
            f"action shape {action.shape}, expected ({expected},)"
        )

    n_tp = cfg.num_tp_levels
    exc = cfg.exchange
    cancel_start = 1 + 1 + 1 + 1 + n_tp + n_tp + 1
    cancel_end = cancel_start + exc.max_open_orders
    close_start = cancel_end
    close_end = close_start + exc.max_open_positions

    cancels = [
        i for i in range(state.num_open_orders)
        if i < exc.max_open_orders and action[cancel_start + i] > 0.0
    ]

    closes: list[CloseIntent] = []
    for i in range(min(state.num_open_positions, exc.max_open_positions)):
        frac = float(max(0.0, min(1.0, (action[close_start + i] + 1.0) / 2.0)))
        if frac > 0.05:
            closes.append(CloseIntent(position_index=i, fraction=frac))

    open_conf = (action[0] + 1.0) / 2.0
    open_intent: OpenIntent | None = None
    if open_conf > 0.5:
        direction = 1 if action[1] > 0.0 else -1
        offset_pct = float(action[2]) * cfg.max_trigger_offset_pct / 100.0
        trigger_price = state.close * (1.0 + offset_pct)

        sl_raw = (action[3] + 1.0) / 2.0
        sl_dist_pct = (
            cfg.min_sl_pct
            + sl_raw * (cfg.max_sl_pct - cfg.min_sl_pct)
        ) / 100.0
        if direction == 1:
            sl_price = trigger_price * (1.0 - sl_dist_pct)
        else:
            sl_price = trigger_price * (1.0 + sl_dist_pct)

        tp_prices: list[float] = []
        raw_tp_sizes: list[float] = []
        for j in range(n_tp):
            tp_raw = (action[4 + j] + 1.0) / 2.0
            tp_dist_pct = max(tp_raw * cfg.max_tp_pct / 100.0, 0.001)
            if direction == 1:
                tp_price = trigger_price * (1.0 + tp_dist_pct)
            else:
                tp_price = trigger_price * (1.0 - tp_dist_pct)
            tp_prices.append(tp_price)
            raw_tp_sizes.append(max((action[4 + n_tp + j] + 1.0) / 2.0, 0.01))

        total = sum(raw_tp_sizes)
        tp_size_pcts = [s / total for s in raw_tp_sizes]

        size_raw = (action[4 + 2 * n_tp] + 1.0) / 2.0
        margin = float(size_raw * state.available_balance)

        if margin >= cfg.exchange.min_order_size_usd:
            open_intent = OpenIntent(
                direction=direction,
                trigger_price=trigger_price,
                sl_price=sl_price,
                tp_prices=tp_prices,
                tp_size_pcts=tp_size_pcts,
                margin=margin,
            )

    return OrderIntent(open=open_intent, cancels=cancels, closes=closes)
