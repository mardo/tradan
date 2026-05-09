"""Replay adapter — drives the live code path with historical klines.

Wraps ExchangeSim so fills, fees, leverage, and liquidations match the
trainer's eval exactly. The cursor advances one candle per `advance()` call.

Used only by scripts/live_replay.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from live.exchange.base import (
    Balance,
    ExchangeAdapter,
    Kline,
    Order,
    OrderRequest,
    Position,
)
from trainer.config import ExchangeConfig
from trainer.env.account import Account
from trainer.env.exchange_sim import ExchangeSim


@dataclass
class _ReplayState:
    timestamps: np.ndarray
    features: np.ndarray
    price_columns: dict[str, int]
    interval: str
    symbol: str
    cursor: int                # index into features (current candle)
    sim: ExchangeSim


class ReplayAdapter(ExchangeAdapter):
    """Constructed via from_arrays(). Do not instantiate directly."""

    def __init__(self, state: _ReplayState):
        self._state = state

    @classmethod
    def from_arrays(
        cls,
        *,
        timestamps: np.ndarray,
        features: np.ndarray,
        price_columns: dict[str, int],
        symbol: str,
        interval: str,
        starting_balance: float,
        exchange_config: ExchangeConfig | None = None,
    ) -> "ReplayAdapter":
        cfg = exchange_config or ExchangeConfig()
        sim = ExchangeSim(config=cfg, account=Account(initial_balance=starting_balance))
        state = _ReplayState(
            timestamps=timestamps, features=features,
            price_columns=price_columns, interval=interval, symbol=symbol,
            cursor=0, sim=sim,
        )
        return cls(state)

    @property
    def cursor(self) -> int:
        return self._state.cursor

    @property
    def sim(self) -> ExchangeSim:
        return self._state.sim

    def advance(self) -> None:
        st = self._state
        idx = st.cursor
        if idx >= len(st.features):
            return
        row = st.features[idx]
        high = float(row[st.price_columns["high"]])
        low = float(row[st.price_columns["low"]])
        close = float(row[st.price_columns["close"]])
        st.sim.process_candle(high=high, low=low, close=close)
        st.cursor += 1

    # -- ExchangeAdapter interface ------------------------------------------

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[Kline]:
        st = self._state
        end = max(0, st.cursor)
        start = max(0, end - limit)
        out: list[Kline] = []
        for i in range(start, end):
            row = st.features[i]
            out.append(Kline(
                open_time_ms=int(st.timestamps[i]),
                open=float(row[st.price_columns["open"]]),
                high=float(row[st.price_columns["high"]]),
                low=float(row[st.price_columns["low"]]),
                close=float(row[st.price_columns["close"]]),
                volume=float(row[st.price_columns["volume"]]),
                quote_volume=_lookup_optional(row, st.price_columns, "quote_volume"),
                num_trades=_lookup_optional_int(row, st.price_columns, "num_trades"),
                taker_buy_base_vol=_lookup_optional(row, st.price_columns, "taker_buy_base_vol"),
                taker_buy_quote_vol=_lookup_optional(row, st.price_columns, "taker_buy_quote_vol"),
            ))
        return out

    def fetch_balance(self) -> Balance:
        sim = self._state.sim
        unrealized = sim.total_unrealized_pnl(self._current_close())
        return Balance(
            total=sim.account.equity(unrealized),
            available=sim.account.available_balance,
            used=sim.account.margin_used,
        )

    def fetch_positions(self, symbol: str) -> list[Position]:
        out: list[Position] = []
        close = self._current_close()
        for p in self._state.sim.open_positions:
            out.append(Position(
                id=str(p.id), symbol=symbol,
                side="long" if p.direction == 1 else "short",
                entry_price=p.entry_price, size=p.size,
                leverage=p.leverage, unrealized_pnl=p.unrealized_pnl(close),
                margin=p.margin, liquidation_price=p.liquidation_price,
            ))
        return out

    def fetch_open_orders(self, symbol: str) -> list[Order]:
        out: list[Order] = []
        for o in self._state.sim.open_orders:
            out.append(Order(
                id=str(o.id), symbol=symbol,
                side="buy" if o.direction == 1 else "sell",
                type="limit", price=o.trigger_price, amount=o.margin,
                status="open",
            ))
        return out

    def place_order(self, symbol: str, request: OrderRequest) -> Order:
        # ReplayAdapter doesn't accept raw OrderRequest from the runner —
        # it uses the trainer's apply_intent path. This method exists to
        # satisfy the ABC and is unused in the replay flow.
        raise NotImplementedError(
            "ReplayAdapter does not implement place_order; runner uses sim.apply_intent"
        )

    def cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError(
            "ReplayAdapter does not implement cancel_order; runner uses sim.apply_intent"
        )

    def close_position(self, symbol: str, position_id: str, fraction: float) -> Order:
        raise NotImplementedError(
            "ReplayAdapter does not implement close_position; runner uses sim.apply_intent"
        )

    def set_leverage(self, symbol: str, leverage: float) -> None:
        # No-op — leverage is computed per-order by ExchangeSim.
        return

    # -- internal helpers ---------------------------------------------------

    def _current_close(self) -> float:
        st = self._state
        idx = max(0, st.cursor - 1)
        if idx >= len(st.features):
            idx = len(st.features) - 1
        if idx < 0 or idx >= len(st.features):
            return 1.0
        return float(st.features[idx][st.price_columns["close"]])


def _lookup_optional(row, price_columns: dict[str, int], key: str) -> float | None:
    idx = price_columns.get(key)
    if idx is None:
        return None
    return float(row[idx])


def _lookup_optional_int(row, price_columns: dict[str, int], key: str) -> int | None:
    idx = price_columns.get(key)
    if idx is None:
        return None
    return int(row[idx])
