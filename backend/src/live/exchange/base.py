"""Abstract ExchangeAdapter and exchange-agnostic DTOs.

Concrete adapters (BingX, Replay, future Binance/Bybit) implement this
interface. The runner does not import any concrete adapter directly — it
goes through live.exchange.registry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Kline:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    num_trades: int | None = None
    taker_buy_base_vol: float | None = None
    taker_buy_quote_vol: float | None = None


@dataclass(frozen=True)
class Balance:
    total: float            # equity-equivalent (USDT)
    available: float        # free margin
    used: float             # margin in use


@dataclass(frozen=True)
class Position:
    id: str                 # exchange position id (or symbol-side composite)
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    size: float             # base units
    leverage: float
    unrealized_pnl: float
    margin: float
    liquidation_price: float | None


@dataclass(frozen=True)
class Order:
    id: str
    symbol: str
    side: Literal["buy", "sell"]
    type: Literal["limit", "market", "stop", "take_profit"]
    price: float | None
    amount: float
    status: Literal["open", "filled", "cancelled", "rejected"]
    fill_price: float | None = None
    fill_amount: float | None = None


@dataclass(frozen=True)
class OrderRequest:
    side: Literal["buy", "sell"]
    type: Literal["limit", "market", "stop", "take_profit"]
    amount: float           # base units
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


class ExchangeAdapter(ABC):
    @abstractmethod
    def fetch_klines(
        self, symbol: str, interval: str, limit: int
    ) -> list[Kline]: ...

    @abstractmethod
    def fetch_balance(self) -> Balance: ...

    @abstractmethod
    def fetch_positions(self, symbol: str) -> list[Position]: ...

    @abstractmethod
    def fetch_open_orders(self, symbol: str) -> list[Order]: ...

    @abstractmethod
    def place_order(self, symbol: str, request: OrderRequest) -> Order: ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> None: ...

    @abstractmethod
    def close_position(self, symbol: str, position_id: str, fraction: float) -> Order: ...

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: float) -> None: ...
