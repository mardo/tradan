"""
Worker process: claims one job at a time from ingest_jobs via
SELECT FOR UPDATE SKIP LOCKED, downloads + parses the zip, bulk-inserts
klines, then marks the job done or failed.
"""
from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone

import psycopg
import psycopg.rows

from .db import connect
from .downloader import fetch_zip
from .parser import parse_zip

_CLAIM_SQL = """
SELECT id, symbol, interval, year, month
FROM   ingest_jobs
WHERE  status = 'pending'
ORDER  BY year, month
FOR UPDATE SKIP LOCKED
LIMIT  1
"""

_INSERT_KLINE_SQL = """
INSERT INTO klines (
    symbol, interval, open_time, open, high, low, close, volume,
    close_time, quote_volume, num_trades,
    taker_buy_base_vol, taker_buy_quote_vol
) VALUES (
    %(symbol)s, %(interval)s, %(open_time)s, %(open)s, %(high)s,
    %(low)s, %(close)s, %(volume)s, %(close_time)s, %(quote_volume)s,
    %(num_trades)s, %(taker_buy_base_vol)s, %(taker_buy_quote_vol)s
)
ON CONFLICT (symbol, interval, open_time) DO NOTHING
"""

_BATCH_SIZE = 5_000


def _insert_rows(conn: psycopg.Connection, rows: list[dict]) -> int:
    """Bulk-insert a batch of kline rows. Returns number inserted."""
    with conn.transaction():
        conn.executemany(_INSERT_KLINE_SQL, rows)
    return len(rows)


def _process_job(
    conn: psycopg.Connection,
    job_id: int,
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> None:
    label = f"{symbol}/{interval} {year:04d}-{month:02d}"
    pid = os.getpid()
    print(f"[worker {pid}] Downloading {label}")

    zip_bytes = fetch_zip(symbol, interval, year, month)

    print(f"[worker {pid}] Parsing {label} ({len(zip_bytes):,} bytes)")
    batch: list[dict] = []
    total = 0
    for row in parse_zip(zip_bytes, symbol, interval):
        batch.append(row)
        if len(batch) >= _BATCH_SIZE:
            total += _insert_rows(conn, batch)
            batch.clear()
    if batch:
        total += _insert_rows(conn, batch)

    with conn.transaction():
        conn.execute(
            """
            UPDATE ingest_jobs
            SET status = 'done', completed_at = %s, error = NULL
            WHERE id = %s
            """,
            (datetime.now(timezone.utc), job_id),
        )

    print(f"[worker {pid}] Done {label} — {total:,} rows inserted")


def _mark_failed(conn: psycopg.Connection, job_id: int, error: str) -> None:
    try:
        conn.execute(
            """
            UPDATE ingest_jobs
            SET status = 'failed', completed_at = %s, error = %s
            WHERE id = %s
            """,
            (datetime.now(timezone.utc), error[:2000], job_id),
        )
        conn.commit()
    except Exception:
        pass


def run_worker() -> None:
    """
    Main loop for a single worker process.
    Keeps claiming and processing jobs until the queue is empty.
    """
    pid = os.getpid()
    conn = connect()
    conn.autocommit = False

    try:
        while True:
            # Claim one pending job atomically
            with conn.transaction():
                row = conn.execute(
                    _CLAIM_SQL,
                    row_factory=psycopg.rows.dict_row,  # type: ignore[call-arg]
                ).fetchone()

                if row is None:
                    print(f"[worker {pid}] No more pending jobs — exiting.")
                    return

                job_id = row["id"]
                conn.execute(
                    """
                    UPDATE ingest_jobs
                    SET status = 'running', claimed_at = %s
                    WHERE id = %s
                    """,
                    (datetime.now(timezone.utc), job_id),
                )

            try:
                _process_job(
                    conn,
                    job_id,
                    row["symbol"],
                    row["interval"],
                    row["year"],
                    row["month"],
                )
            except FileNotFoundError as exc:
                print(f"[worker {pid}] Missing file, marking failed: {exc}")
                _mark_failed(conn, job_id, str(exc))
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[worker {pid}] Error processing job {job_id}:\n{tb}")
                _mark_failed(conn, job_id, tb)
    finally:
        conn.close()
