"""
Endpoints:
  GET /api/jobs/summary          – status counts {pending: N, done: N, ...}
  GET /api/jobs?status=&symbol=&limit=50&offset=0
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..db import get_conn

router = APIRouter()


@router.get("/jobs/summary")
def get_jobs_summary() -> dict:
    """Return a count per status value."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT status, count(*) FROM ingest_jobs GROUP BY status ORDER BY status"
        ).fetchall()
    return {row[0]: row[1] for row in rows}


@router.get("/jobs")
def get_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """Return paginated job rows plus a total count."""
    conditions: list[str] = []
    params: list = []

    if status:
        conditions.append("status = %s")
        params.append(status)
    if symbol:
        conditions.append("symbol = %s")
        params.append(symbol)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_conn() as conn:
        total: int = conn.execute(
            f"SELECT count(*) FROM ingest_jobs {where}", params
        ).fetchone()[0]  # type: ignore[index]

        rows = conn.execute(
            f"""
            SELECT id, symbol, interval, year, month,
                   status, claimed_at, completed_at, error
            FROM   ingest_jobs
            {where}
            ORDER  BY id DESC
            LIMIT  %s OFFSET %s
            """,
            params + [limit, offset],
        ).fetchall()

    return {
        "total": total,
        "jobs": [
            {
                "id": r[0],
                "symbol": r[1],
                "interval": r[2],
                "year": r[3],
                "month": r[4],
                "status": r[5],
                "claimed_at": r[6].isoformat() if r[6] else None,
                "completed_at": r[7].isoformat() if r[7] else None,
                "error": r[8],
            }
            for r in rows
        ],
    }
