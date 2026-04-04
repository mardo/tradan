"""
Download a single monthly kline zip from data.binance.vision.
Returns the raw zip bytes; does not write to disk.
"""
from __future__ import annotations

from urllib.error import HTTPError
from urllib.request import Request, urlopen

DATA_HOST = "https://data.binance.vision"
BASE_PREFIX = "data/futures/um/monthly"

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def build_url(symbol: str, interval: str, year: int, month: int) -> str:
    month_str = f"{year:04d}-{month:02d}"
    return (
        f"{DATA_HOST}/{BASE_PREFIX}/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{month_str}.zip"
    )


def fetch_zip(symbol: str, interval: str, year: int, month: int) -> bytes:
    """
    Download the monthly kline zip and return its raw bytes.

    Raises:
        FileNotFoundError: if the server returns 404 (month not yet published).
        RuntimeError: on any other HTTP error.
    """
    url = build_url(symbol, interval, year, month)
    req = Request(url, headers=_HEADERS)
    try:
        with urlopen(req, timeout=120) as resp:
            return resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(f"Not found on Binance: {url}") from exc
        raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
