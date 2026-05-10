"""BingX VST adapter via ccxt.

VST = Virtual Simulation Trading: real prices, fake balance, real API surface.
The unified ccxt symbol for BTCUSDT perp is 'BTC/USDT:USDT'.

Construction:
  BingXAdapter.from_env(api_key_env=..., api_secret_env=..., mode='demo')

Mode 'demo' enables ccxt sandbox routing for VST. Mode 'live' targets the
production endpoint — only flip when ready for real money.
"""
from __future__ import annotations

import os
from typing import Any

import ccxt

from live.exchange.base import (
    Balance,
    ExchangeAdapter,
    Kline,
    Order,
    OrderRequest,
    Position,
)


class BingXAdapter(ExchangeAdapter):
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        mode: str,                  # "demo" | "live"
    ):
        if mode not in ("demo", "live"):
            raise ValueError(f"unsupported mode: {mode}")
        self._mode = mode
        self._client = ccxt.bingx({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if mode == "demo":
            # ccxt unified flag for demo trading.
            self._client.set_sandbox_mode(True)

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str,
        api_secret_env: str,
        mode: str,
    ) -> "BingXAdapter":
        try:
            key = os.environ[api_key_env]
            secret = os.environ[api_secret_env]
        except KeyError as e:
            raise RuntimeError(
                f"missing env var {e.args[0]!r} for BingX adapter"
            ) from e
        return cls(api_key=key, api_secret=secret, mode=mode)

    # -- read methods -------------------------------------------------------

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[Kline]:
        # BingX's unified fetch_ohlcv only returns OHLCV (timestamp + 5 fields).
        # The trainer's picks were trained on 9 columns (Binance archive). The
        # 4 extras (quote_volume, num_trades, taker_buy_base_vol, taker_buy_quote_vol)
        # are populated as 0.0 here so feature_pipeline does not raise. The model
        # will see zeros in those slots, which is a known divergence from replay
        # (which uses real Binance values from the klines table). The 4-week paper
        # test on BingX VST is what surfaces whether this matters in practice.
        rows = self._client.fetch_ohlcv(symbol, interval, limit=limit)
        return [
            Kline(
                open_time_ms=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
                quote_volume=0.0,
                num_trades=0,
                taker_buy_base_vol=0.0,
                taker_buy_quote_vol=0.0,
            )
            for r in rows
        ]

    def fetch_balance(self) -> Balance:
        bal = self._client.fetch_balance()
        # BingX VST accounts denominate in "VST" instead of "USDT".
        # In live mode we want USDT; in demo we want VST. Try the
        # mode-appropriate key first, then fall back to whichever
        # asset has a nonzero total (graceful for either setup).
        asset = "VST" if self._mode == "demo" else "USDT"
        entry = bal.get(asset, {}) if isinstance(bal, dict) else {}
        if not entry:
            for k, v in (bal or {}).items():
                if isinstance(v, dict) and (v.get("total") or 0):
                    entry = v
                    break
        total = float(entry.get("total", 0.0) or 0.0)
        free = float(entry.get("free", 0.0) or 0.0)
        used = float(entry.get("used", 0.0) or 0.0)
        return Balance(total=total, available=free, used=used)

    def fetch_positions(self, symbol: str) -> list[Position]:
        rows = self._client.fetch_positions([symbol])
        out: list[Position] = []
        for r in rows:
            contracts = float(r.get("contracts") or 0.0)
            if contracts == 0:
                continue
            side = "long" if r.get("side") == "long" else "short"
            out.append(Position(
                id=str(r.get("id") or f"{symbol}-{side}"),
                symbol=symbol,
                side=side,
                entry_price=float(r.get("entryPrice") or 0.0),
                size=contracts,
                leverage=float(r.get("leverage") or 1.0),
                unrealized_pnl=float(r.get("unrealizedPnl") or 0.0),
                margin=float(r.get("initialMargin") or 0.0),
                liquidation_price=(
                    float(r["liquidationPrice"])
                    if r.get("liquidationPrice") is not None
                    else None
                ),
            ))
        return out

    def fetch_open_orders(self, symbol: str) -> list[Order]:
        rows = self._client.fetch_open_orders(symbol)
        out: list[Order] = []
        for r in rows:
            info: dict[str, Any] = r.get("info") or {}
            out.append(Order(
                id=str(r["id"]),
                symbol=symbol,
                side="buy" if r["side"] == "buy" else "sell",
                type=_map_order_type(r.get("type")),
                price=(float(r["price"]) if r.get("price") is not None else None) or None,
                amount=float(r["amount"]),
                status="open",
                stop_loss=_optional_float(info.get("stopLoss") or info.get("stop_loss")),
                take_profit_prices=_optional_tp_list(info.get("takeProfit") or info.get("take_profit")),
                take_profit_size_pcts=None,   # BingX doesn't surface partial-TP weights
            ))
        return out

    # -- write methods ------------------------------------------------------

    def place_order(self, symbol: str, request: OrderRequest) -> Order:
        params: dict[str, Any] = {}
        if request.stop_loss is not None:
            params["stopLoss"] = {
                "type": "STOP_MARKET",
                "stopPrice": float(request.stop_loss),
            }
        if request.take_profit is not None:
            params["takeProfit"] = {
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": float(request.take_profit),
            }

        if request.type == "market":
            r = self._client.create_market_order(
                symbol, request.side, float(request.amount), params=params,
            )
        elif request.type == "limit":
            if request.price is None:
                raise ValueError("limit order requires price")
            r = self._client.create_limit_order(
                symbol, request.side, float(request.amount),
                float(request.price), params=params,
            )
        else:
            raise ValueError(
                f"unsupported order type for placement: {request.type}"
            )

        return Order(
            id=str(r["id"]),
            symbol=symbol,
            side=request.side,
            type=request.type,
            price=(
                float(r.get("price") or request.price or 0.0)
                if (r.get("price") or request.price) is not None
                else None
            ),
            amount=float(r["amount"]),
            status=_map_status(r.get("status")),
            stop_loss=request.stop_loss,
            take_profit_prices=[request.take_profit] if request.take_profit is not None else None,
            take_profit_size_pcts=None,
        )

    def cancel_order(self, symbol: str, order_id: str) -> None:
        self._client.cancel_order(order_id, symbol)

    def close_position(
        self, symbol: str, position_id: str, fraction: float,
    ) -> Order:
        positions = self.fetch_positions(symbol)
        target = next((p for p in positions if p.id == position_id), None)
        if target is None:
            raise ValueError(f"position {position_id} not found at exchange")
        qty = float(target.size) * float(fraction)
        side = "sell" if target.side == "long" else "buy"
        r = self._client.create_market_order(
            symbol, side, qty, params={"reduceOnly": True},
        )
        return Order(
            id=str(r["id"]),
            symbol=symbol,
            side=side,
            type="market",
            price=None,
            amount=qty,
            status=_map_status(r.get("status")),
        )

    def set_leverage(self, symbol: str, leverage: float) -> None:
        # BingX requires a side argument for setLeverage. The picks may go
        # long or short, so set the same leverage for both. We try BOTH
        # first (one-way / hedge-disabled mode); if rejected, fall back to
        # setting LONG and SHORT individually (hedge mode).
        try:
            self._client.set_leverage(int(leverage), symbol, params={"side": "BOTH"})
        except Exception:
            self._client.set_leverage(int(leverage), symbol, params={"side": "LONG"})
            self._client.set_leverage(int(leverage), symbol, params={"side": "SHORT"})


# -- helpers ------------------------------------------------------------------


def _map_order_type(ccxt_type: str | None) -> str:
    mapping = {
        "limit": "limit",
        "market": "market",
        "stop_market": "stop",
        "stop": "stop",
        "take_profit_market": "take_profit",
        "take_profit": "take_profit",
    }
    return mapping.get((ccxt_type or "").lower(), "limit")


def _map_status(s: str | None) -> str:
    s = (s or "").lower()
    if s in ("open", "pending"):
        return "open"
    if s in ("closed", "filled"):
        return "filled"
    if s in ("canceled", "cancelled"):
        return "cancelled"
    return "rejected"


def _optional_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f or None


def _optional_tp_list(v: Any) -> list[float] | None:
    """BingX's takeProfit field may be a single price, a dict with stopPrice,
    or a list. Normalize to a list of prices when possible; None otherwise."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, dict):
        sp = v.get("stopPrice") or v.get("stop_price") or v.get("price")
        if sp is None:
            return None
        return [float(sp)]
    if isinstance(v, list):
        out: list[float] = []
        for item in v:
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, dict):
                sp = item.get("stopPrice") or item.get("stop_price") or item.get("price")
                if sp is not None:
                    out.append(float(sp))
        return out or None
    return None
