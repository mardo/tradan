"""
Download a single monthly kline zip from data.binance.vision.
Returns the raw zip bytes; does not write to disk.
"""
from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .logutil import get_logger

DATA_HOST = "https://data.binance.vision"
# BASE_PREFIX = "data/futures/um/monthly"
BASE_PREFIX = "data/spot/monthly"

_HEADERS = {"User-Agent": "Mozilla/5.0"}

_log = get_logger("downloader")


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
        RuntimeError: on any other HTTP or network error.
    """
    url = build_url(symbol, interval, year, month)
    _log.debug("GET %s", url)
    req = Request(url, headers=_HEADERS)
    try:
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
            clen = resp.headers.get("Content-Length")
            _log.debug(
                "Response status=%s content_length_header=%s bytes_read=%s",
                getattr(resp, "status", "?"),
                clen,
                len(data),
            )
            return data
    except HTTPError as exc:
        body_preview = (exc.read() or b"")[:500]
        preview_txt = body_preview.decode("utf-8", errors="replace").replace("\n", " ")
        if exc.code == 404:
            _log.warning(
                "HTTP 404 (object missing) url=%s body_preview=%r",
                url,
                preview_txt,
            )
            raise FileNotFoundError(f"Not found on Binance: {url}") from exc
        _log.error(
            "HTTP %s url=%s reason=%r body_preview=%r",
            exc.code,
            url,
            exc.reason,
            preview_txt,
        )
        raise RuntimeError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
    except URLError as exc:
        _log.error("Network error url=%s: %s", url, exc.reason, exc_info=True)
        raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc
