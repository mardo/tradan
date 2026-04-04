"""
Endpoints:
  GET /api/symbols                            – distinct symbols and their intervals
  GET /api/klines?symbol=&interval=&limit=&from_time=&to_time=
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..db import get_conn

router = APIRouter()


@router.get("/symbols")
def get_symbols() -> list[dict]:
    """Return each distinct symbol with its available intervals."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol, interval FROM klines ORDER BY symbol, interval"
        ).fetchall()

    result: dict[str, list[str]] = {}
    for symbol, interval in rows:
        result.setdefault(symbol, []).append(interval)
    return [{"symbol": k, "intervals": v} for k, v in result.items()]


@router.get("/klines")
def get_klines(
    symbol: str = Query(..., description="e.g. BTCUSDT"),
    interval: str = Query(..., description="e.g. 1m, 1h, 1d"),
    limit: int = Query(1000, ge=1, le=5000),
    from_time: Optional[int] = Query(None, description="open_time >= this value (ms epoch)"),
    to_time: Optional[int] = Query(None, description="open_time <= this value (ms epoch)"),
) -> list[dict]:
    """
    Return candlestick rows sorted ascending by open_time.
    `time` field is open_time in milliseconds (convert to seconds for lightweight-charts).
    """
    conditions = ["symbol = %s", "interval = %s"]
    params: list = [symbol, interval]

    if from_time is not None:
        conditions.append("open_time >= %s")
        params.append(from_time)
    if to_time is not None:
        conditions.append("open_time <= %s")
        params.append(to_time)

    where = " AND ".join(conditions)

    with get_conn() as conn:
        # Fetch newest first to honour LIMIT, then reverse for ascending order
        rows = conn.execute(
            f"""
            SELECT open_time, open, high, low, close, volume, num_trades
            FROM   klines
            WHERE  {where}
            ORDER  BY open_time DESC
            LIMIT  %s
            """,
            params + [limit],
        ).fetchall()

    return [
        {
            "time": r[0],
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
            "num_trades": r[6],
        }
        for r in reversed(rows)
    ]
