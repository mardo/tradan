"""
Fetch historical klines for a full calendar month via ccxt (Binance USDT-M futures).
Used as a fallback when the Binance data.vision zip is unavailable (404).

Note: ccxt standard OHLCV returns [timestamp, open, high, low, close, volume].
The Binance-specific fields (quote_volume, num_trades, taker volumes) are not
available through the standard ccxt interface, so they are stored as 0.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterator

import ccxt

from .logutil import get_logger

_log = get_logger("ccxt_fetcher")

_MAX_VALID_MS = 9_999_999_999_999


def _to_ms(ts: int) -> int:
    """Normalise a timestamp to milliseconds, dividing by 1000 if it looks like µs."""
    return ts // 1000 if ts > _MAX_VALID_MS else ts

# Map ingester interval names → ccxt timeframe strings
INTERVAL_TO_TF: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1mo": "1M",
}

# Fixed candle duration in milliseconds (used to compute close_time and advance cursor).
# "1mo" is absent because months have variable length and is handled separately.
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

_MAX_LIMIT = 1500  # Binance OHLCV max candles per request


def _month_range_ms(year: int, month: int) -> tuple[int, int]:
    """Return (start_ms inclusive, end_ms inclusive) for a UTC calendar month."""
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    last_day = calendar.monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 59, 59, 999000, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _to_ccxt_symbol(symbol: str) -> str:
    """Convert BTCUSDT (Binance REST style) → BTC/USDT:USDT (ccxt USDT-M futures)."""
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"
    raise ValueError(
        f"Cannot map symbol to ccxt USDT-M format: {symbol!r}. "
        "Expected a USDT-M symbol ending in 'USDT'."
    )


def _close_time(open_time: int, interval: str, candle_year: int, candle_month: int) -> int:
    """Compute close_time (ms) for a candle given its open_time and interval."""
    if interval == "1mo":
        last_day = calendar.monthrange(candle_year, candle_month)[1]
        end_dt = datetime(
            candle_year, candle_month, last_day, 23, 59, 59, 999000,
            tzinfo=timezone.utc,
        )
        return int(end_dt.timestamp() * 1000)
    ms = INTERVAL_MS.get(interval)
    if ms is None:
        raise ValueError(f"Unknown interval: {interval!r}")
    return open_time + ms - 1


def fetch_month_klines(
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> Iterator[dict]:
    """
    Fetch all OHLCV klines for a full calendar month via ccxt Binance USDT-M futures.

    Yields row dicts compatible with the klines table schema.
    Fields unavailable via ccxt (quote_volume, num_trades, taker_buy_base_vol,
    taker_buy_quote_vol) are stored as 0.

    Raises:
        ValueError: if the interval is unsupported or the symbol cannot be mapped.
        ccxt.BaseError: on exchange or network errors.
    """
    tf = INTERVAL_TO_TF.get(interval)
    if tf is None:
        raise ValueError(f"Unsupported interval for ccxt fetch: {interval!r}")

    ccxt_sym = _to_ccxt_symbol(symbol)
    start_ms, end_ms = _month_range_ms(year, month)
    exchange = ccxt.binanceusdm({"enableRateLimit": True})

    _log.info(
        "ccxt fetch start sym=%s tf=%s %04d-%02d start_ms=%s end_ms=%s",
        ccxt_sym, tf, year, month, start_ms, end_ms,
    )

    since = start_ms
    total = 0

    while True:
        _log.debug("ccxt fetch_ohlcv since=%s limit=%s", since, _MAX_LIMIT)
        ohlcv = exchange.fetch_ohlcv(ccxt_sym, tf, since=since, limit=_MAX_LIMIT)

        if not ohlcv:
            break

        yielded_in_batch = 0
        for candle in ohlcv:
            open_time: int = _to_ms(candle[0])
            if open_time > end_ms:
                _log.info(
                    "ccxt fetch done (past month end) sym=%s interval=%s total=%s",
                    symbol, interval, total,
                )
                return

            # Derive the calendar month of this candle's open_time for close_time calc
            candle_dt = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc)
            ct = _close_time(open_time, interval, candle_dt.year, candle_dt.month)

            yield {
                "symbol": symbol,
                "interval": interval,
                "open_time": open_time,
                "open": Decimal(str(candle[1])),
                "high": Decimal(str(candle[2])),
                "low": Decimal(str(candle[3])),
                "close": Decimal(str(candle[4])),
                "volume": Decimal(str(candle[5])),
                "close_time": ct,
                "quote_volume": Decimal("0"),
                "num_trades": 0,
                "taker_buy_base_vol": Decimal("0"),
                "taker_buy_quote_vol": Decimal("0"),
            }
            yielded_in_batch += 1
            total += 1

        if yielded_in_batch == 0:
            break

        last_open: int = _to_ms(ohlcv[-1][0])
        if interval == "1mo":
            since = last_open + 32 * 86_400_000  # advance ~1 month safely
        else:
            since = last_open + INTERVAL_MS[interval]

        if since > end_ms:
            break

    _log.info(
        "ccxt fetch complete sym=%s interval=%s %04d-%02d total_candles=%s",
        symbol, interval, year, month, total,
    )
