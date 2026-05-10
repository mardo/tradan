"""Convert exchange-DTO klines into the trainer's normalized observation.

Thin wrapper:
  ccxt-style klines → numpy feature array
  account / positions / orders DTOs → ObservationInputs
  → trainer.env.observation.build_observation
"""
from __future__ import annotations

import numpy as np

from live.exchange.base import Balance, Kline
from live.exchange.base import Order as LiveOrder
from live.exchange.base import Position as LivePosition
from trainer.env.exchange_sim import Order as SimOrder
from trainer.env.exchange_sim import Position as SimPosition
from trainer.env.normalization import NormalizationStats
from trainer.env.observation import (
    ObservationConfig,
    ObservationInputs,
    build_observation,
)


def klines_to_features(
    klines: list[Kline],
    columns: list[str],
) -> np.ndarray:
    """Project Kline DTOs into a (N, len(columns)) float32 array.

    `columns` is the list the trainer used. The 9 supported columns
    correspond 1:1 to the trainer's ALL_KLINE_COLUMNS in trainer.config.
    """
    name_to_attr = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "quote_volume": "quote_volume",
        "num_trades": "num_trades",
        "taker_buy_base_vol": "taker_buy_base_vol",
        "taker_buy_quote_vol": "taker_buy_quote_vol",
    }
    rows: list[list[float]] = []
    for k in klines:
        row: list[float] = []
        for c in columns:
            attr = name_to_attr.get(c)
            if attr is None:
                raise KeyError(f"unsupported kline column: {c!r}")
            val = getattr(k, attr)
            if val is None:
                raise ValueError(
                    f"kline at {k.open_time_ms} missing required column {c!r}; "
                    f"adapter must populate this field for models that use it"
                )
            row.append(float(val))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def normalize(features: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((features - stats.mean) / stats.std).astype(np.float32)


def build_live_observation(
    *,
    klines: list[Kline],
    columns: list[str],
    balance: Balance,
    positions: list[LivePosition],
    open_orders: list[LiveOrder],
    stats: NormalizationStats,
    obs_cfg: ObservationConfig,
) -> dict[str, np.ndarray]:
    raw = klines_to_features(klines, columns)
    if raw.shape[0] != obs_cfg.lookback:
        raise ValueError(
            f"got {raw.shape[0]} klines, need {obs_cfg.lookback}"
        )
    market = normalize(raw, stats)
    close = float(klines[-1].close) if klines and klines[-1].close > 0 else 1.0

    inputs = ObservationInputs(
        market=market,
        balance=balance.available + balance.used,
        equity=balance.total,
        unrealized_pnl=balance.total - (balance.available + balance.used),
        margin_used=balance.used,
        available_balance=balance.available,
        open_orders=[_to_sim_order(o) for o in open_orders],
        open_positions=[_to_sim_position(p) for p in positions],
        close=close,
    )
    return build_observation(inputs, obs_cfg)


def _to_sim_order(o: LiveOrder) -> SimOrder:
    """Best-effort projection: live Order DTO → trainer's SimOrder.
    Uses the new optional SL/TP fields when the adapter populates them
    (ReplayAdapter does; live BingX adapter will once D.1/E.1 land)."""
    return SimOrder(
        id=int(o.id) if o.id.isdigit() else 0,
        direction=1 if o.side == "buy" else -1,
        trigger_price=o.price or 0.0,
        sl_price=o.stop_loss if o.stop_loss is not None else (o.price or 0.0),
        tp_prices=list(o.take_profit_prices) if o.take_profit_prices else [],
        tp_size_pcts=list(o.take_profit_size_pcts) if o.take_profit_size_pcts else [],
        margin=o.amount,
    )


def _to_sim_position(p: LivePosition) -> SimPosition:
    return SimPosition(
        id=int(p.id) if p.id.isdigit() else 0,
        direction=1 if p.side == "long" else -1,
        entry_price=p.entry_price,
        size=p.size,
        leverage=p.leverage,
        sl_price=0.0,
        tp_prices=[],
        tp_size_pcts=[],
        margin=p.margin,
        liquidation_price=p.liquidation_price or 0.0,
    )
