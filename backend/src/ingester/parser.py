"""
Parse a Binance monthly kline zip (in memory) into row dicts.

CSV column order (no header row):
  0  open_time            – ms epoch
  1  open
  2  high
  3  low
  4  close
  5  volume               – base asset
  6  close_time           – ms epoch
  7  quote_volume
  8  num_trades
  9  taker_buy_base_vol
  10 taker_buy_quote_vol
  11 ignore               – always 0, discarded
"""
from __future__ import annotations

import csv
import io
import zipfile
from decimal import Decimal
from typing import Iterator

# Binance switched some 2025+ CSV files to microsecond precision.
# Any ms timestamp above this value is ~year 2286, well beyond any real candle;
# values this large were written in µs and need to be divided by 1000.
_MAX_VALID_MS = 9_999_999_999_999


def _to_ms(ts: int) -> int:
    """Normalise a timestamp to milliseconds, dividing by 1000 if it looks like µs."""
    return ts // 1000 if ts > _MAX_VALID_MS else ts


def _to_row(symbol: str, interval: str, fields: list[str]) -> dict:
    return {
        "symbol": symbol,
        "interval": interval,
        "open_time": _to_ms(int(fields[0])),
        "open": Decimal(fields[1]),
        "high": Decimal(fields[2]),
        "low": Decimal(fields[3]),
        "close": Decimal(fields[4]),
        "volume": Decimal(fields[5]),
        "close_time": _to_ms(int(fields[6])),
        "quote_volume": Decimal(fields[7]),
        "num_trades": int(fields[8]),
        "taker_buy_base_vol": Decimal(fields[9]),
        "taker_buy_quote_vol": Decimal(fields[10]),
    }


def parse_zip(
    zip_bytes: bytes, symbol: str, interval: str
) -> Iterator[dict]:
    """
    Unzip *zip_bytes* in memory and yield one dict per kline row.
    The zip is expected to contain exactly one CSV file.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV file found inside zip archive")

        with zf.open(csv_names[0]) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                if not row or not row[0].strip().isdigit():
                    # Skip any header-like lines (some older files have them)
                    continue
                yield _to_row(symbol, interval, row)
