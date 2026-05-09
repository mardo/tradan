"""Stub — implementation in Phase C."""
from __future__ import annotations

from live.exchange.base import (
    Balance, ExchangeAdapter, Kline, Order, OrderRequest, Position,
)


class ReplayAdapter(ExchangeAdapter):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("ReplayAdapter is implemented in Phase C")

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[Kline]:
        raise NotImplementedError

    def fetch_balance(self) -> Balance:
        raise NotImplementedError

    def fetch_positions(self, symbol: str) -> list[Position]:
        raise NotImplementedError

    def fetch_open_orders(self, symbol: str) -> list[Order]:
        raise NotImplementedError

    def place_order(self, symbol: str, request: OrderRequest) -> Order:
        raise NotImplementedError

    def cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError

    def close_position(self, symbol: str, position_id: str, fraction: float) -> Order:
        raise NotImplementedError

    def set_leverage(self, symbol: str, leverage: float) -> None:
        raise NotImplementedError
