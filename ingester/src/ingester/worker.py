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
from .downloader import build_url, fetch_zip
from .logutil import configure_logging, get_logger, job_extra
from .parser import parse_zip

_log = get_logger("worker")

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


def _insert_rows(
    conn: psycopg.Connection, rows: list[dict], job_id: int, label: str
) -> int:
    """Bulk-insert a batch of kline rows. Returns batch size (not exact insert count)."""
    extra = {"job_id": job_id, "job_label": label}
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(_INSERT_KLINE_SQL, rows)
    _log.debug("Inserted batch size=%s", len(rows), extra=extra)
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
    ex = job_extra(job_id, symbol, interval, year, month)
    url = build_url(symbol, interval, year, month)

    _log.info("Start job url=%s", url, extra=ex)

    zip_bytes = fetch_zip(symbol, interval, year, month)

    _log.info("Downloaded zip bytes=%s", f"{len(zip_bytes):,}", extra=ex)
    batch: list[dict] = []
    total_rows = 0
    batches = 0
    for row in parse_zip(zip_bytes, symbol, interval):
        batch.append(row)
        if len(batch) >= _BATCH_SIZE:
            _insert_rows(conn, batch, job_id, label)
            total_rows += len(batch)
            batches += 1
            batch.clear()
    if batch:
        _insert_rows(conn, batch, job_id, label)
        total_rows += len(batch)
        batches += 1

    with conn.transaction():
        conn.execute(
            """
            UPDATE ingest_jobs
            SET status = 'done', completed_at = %s, error = NULL
            WHERE id = %s
            """,
            (datetime.now(timezone.utc), job_id),
        )

    _log.info(
        "Job done rows_processed=%s insert_batches=%s (ON CONFLICT may skip duplicates)",
        f"{total_rows:,}",
        batches,
        extra=ex,
    )


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
        _log.exception("Could not persist job failure to DB job_id=%s", job_id)


def run_worker() -> None:
    """
    Main loop for a single worker process.
    Keeps claiming and processing jobs until the queue is empty.
    """
    configure_logging()
    pid = os.getpid()
    _log.info("Worker started pid=%s", pid)
    conn = connect()
    conn.autocommit = False

    try:
        while True:
            # Claim one pending job atomically
            with conn.transaction():
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute(_CLAIM_SQL)
                    row = cur.fetchone()

                    if row is None:
                        _log.info("No pending jobs left; worker exiting pid=%s", pid)
                        return

                    job_id = row["id"]
                    cur.execute(
                        """
                        UPDATE ingest_jobs
                        SET status = 'running', claimed_at = %s
                        WHERE id = %s
                        """,
                        (datetime.now(timezone.utc), job_id),
                    )

            sym, inv, yr, mo = (
                row["symbol"],
                row["interval"],
                row["year"],
                row["month"],
            )
            jl = f"{sym}/{inv} {yr:04d}-{mo:02d}"
            _log.info("Claimed job %s", jl, extra=job_extra(job_id, sym, inv, yr, mo))

            try:
                _process_job(conn, job_id, sym, inv, yr, mo)
            except FileNotFoundError as exc:
                _log.warning(
                    "Job failed (missing remote file): %s",
                    exc,
                    extra=job_extra(job_id, sym, inv, yr, mo),
                )
                _mark_failed(conn, job_id, str(exc))
            except Exception:
                _log.exception(
                    "Job failed",
                    extra=job_extra(job_id, sym, inv, yr, mo),
                )
                err = traceback.format_exc()
                _mark_failed(conn, job_id, err)
    finally:
        conn.close()
