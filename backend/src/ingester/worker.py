"""
Worker process: claims one job at a time from ingest_jobs via
SELECT FOR UPDATE SKIP LOCKED, downloads + parses the zip, bulk-inserts
klines, then marks the job done or failed.

Two worker modes are available:
  run_worker()       – standard mode: fetches Binance data.vision zips
  run_ccxt_worker()  – retry mode: fetches via ccxt (targets 'failed' jobs)
"""
from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from typing import Iterator

import psycopg
import psycopg.rows

from .ccxt_fetcher import fetch_month_klines
from .db import connect
from .downloader import build_url, fetch_zip
from .logutil import configure_logging, get_logger, job_extra
from .parser import parse_zip

_log = get_logger("worker")

_CLAIM_PENDING_SQL = """
SELECT id, symbol, interval, year, month
FROM   ingest_jobs
WHERE  status = 'pending'
ORDER  BY year, month
FOR UPDATE SKIP LOCKED
LIMIT  1
"""

_CLAIM_FAILED_SQL = """
SELECT id, symbol, interval, year, month
FROM   ingest_jobs
WHERE  status = 'failed'
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


def _run_job_pipeline(
    conn: psycopg.Connection,
    job_id: int,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    rows_iter: Iterator[dict],
) -> None:
    """
    Consume *rows_iter*, bulk-insert all klines in batches, then mark the job done.
    Shared by both the zip-based and CCXT-based worker paths.
    """
    label = f"{symbol}/{interval} {year:04d}-{month:02d}"
    ex = job_extra(job_id, symbol, interval, year, month)

    batch: list[dict] = []
    total_rows = 0
    batches = 0
    for row in rows_iter:
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


def _process_job(
    conn: psycopg.Connection,
    job_id: int,
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> None:
    """Download the Binance data.vision zip and insert all klines."""
    ex = job_extra(job_id, symbol, interval, year, month)
    url = build_url(symbol, interval, year, month)

    _log.info("Start job url=%s", url, extra=ex)
    zip_bytes = fetch_zip(symbol, interval, year, month)
    _log.info("Downloaded zip bytes=%s", f"{len(zip_bytes):,}", extra=ex)

    _run_job_pipeline(conn, job_id, symbol, interval, year, month,
                      parse_zip(zip_bytes, symbol, interval))


def _process_job_ccxt(
    conn: psycopg.Connection,
    job_id: int,
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> None:
    """Fetch klines via ccxt Binance USDT-M futures and insert them."""
    ex = job_extra(job_id, symbol, interval, year, month)
    _log.info(
        "Start CCXT job sym=%s interval=%s %04d-%02d",
        symbol, interval, year, month, extra=ex,
    )
    _run_job_pipeline(conn, job_id, symbol, interval, year, month,
                      fetch_month_klines(symbol, interval, year, month))


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


def _worker_loop(
    claim_sql: str,
    process_fn,
    mode_label: str,
) -> None:
    """
    Generic worker loop: claims jobs via *claim_sql*, processes each with
    *process_fn(conn, job_id, symbol, interval, year, month)*, and repeats
    until the queue is empty.
    """
    configure_logging()
    pid = os.getpid()
    _log.info("Worker started mode=%s pid=%s", mode_label, pid)
    conn = connect()
    conn.autocommit = False

    try:
        while True:
            with conn.transaction():
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute(claim_sql)
                    row = cur.fetchone()

                    if row is None:
                        _log.info(
                            "No jobs left for mode=%s; worker exiting pid=%s",
                            mode_label, pid,
                        )
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
            _log.info("Claimed job %s mode=%s", jl, mode_label,
                      extra=job_extra(job_id, sym, inv, yr, mo))

            try:
                process_fn(conn, job_id, sym, inv, yr, mo)
            except FileNotFoundError as exc:
                _log.warning(
                    "Job failed (missing remote file): %s",
                    exc,
                    extra=job_extra(job_id, sym, inv, yr, mo),
                )
                _mark_failed(conn, job_id, str(exc))
            except Exception:
                _log.exception(
                    "Job failed mode=%s", mode_label,
                    extra=job_extra(job_id, sym, inv, yr, mo),
                )
                _mark_failed(conn, job_id, traceback.format_exc())
    finally:
        conn.close()


def run_worker() -> None:
    """
    Standard worker: claims pending jobs, fetches Binance data.vision zips.
    Keeps running until the pending queue is empty.
    """
    _worker_loop(_CLAIM_PENDING_SQL, _process_job, "zip")


def run_ccxt_worker() -> None:
    """
    CCXT retry worker: claims failed jobs and re-fetches data via ccxt.
    Keeps running until no failed jobs remain.
    """
    _worker_loop(_CLAIM_FAILED_SQL, _process_job_ccxt, "ccxt")
