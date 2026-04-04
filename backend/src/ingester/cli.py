"""
CLI entry point.

Commands
--------
ingest migrate                          – apply pending DB migrations
ingest enqueue [options]                – add jobs to ingest_jobs
ingest run [--workers N]               – drain the queue with N parallel workers
ingest retry [--workers N]             – retry failed jobs via ccxt
ingest status                          – show job counts by status
ingest verify [--symbol S] [--interval I] – check for gaps in kline data
ingest fill-gaps [--symbol S] [--interval I] – mark months with gaps as failed for ccxt retry
ingest reset                           – truncate klines, reset jobs to pending
ingest reset --hard                    – truncate klines + delete all jobs
ingest reset --failed                  – re-queue only failed jobs
"""
from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing
import os
from datetime import datetime, timezone
from typing import Iterable

from .ccxt_fetcher import INTERVAL_MS
from .db import connect, migrate
from .logutil import configure_logging, get_logger
from .worker import run_ccxt_worker, run_worker

_cli_log = get_logger("cli")

ALL_INTERVALS = [
    "12h", "15m", "1d", "1h", "1m", "1mo", "1w",
    "2h", "30m", "3d", "3m", "4h", "5m", "6h", "8h",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_month(raw: str) -> dt.date:
    """Accept YYYY-MM or YYYY-MM-DD and return a date at the 1st of the month."""
    raw = raw.strip()
    if len(raw) == 7 and raw[4] == "-":
        return dt.date.fromisoformat(f"{raw}-01")
    return dt.date.fromisoformat(raw)


def _monthrange(start: dt.date, end: dt.date) -> Iterable[tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_migrate(_args: argparse.Namespace) -> None:
    migrate()


def cmd_enqueue(args: argparse.Namespace) -> None:
    start = _parse_month(args.start)
    end = _parse_month(args.end)
    intervals = args.interval or ALL_INTERVALS

    conn = connect()
    try:
        inserted = 0
        skipped = 0
        for interval in intervals:
            for year, month in _monthrange(start, end):
                result = conn.execute(
                    """
                    INSERT INTO ingest_jobs (symbol, interval, year, month)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (symbol, interval, year, month) DO NOTHING
                    """,
                    (args.symbol, interval, year, month),
                )
                if result.rowcount:
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
        print(f"Enqueued {inserted} job(s), skipped {skipped} already-existing.")
    finally:
        conn.close()


def cmd_run(args: argparse.Namespace) -> None:
    lvl = args.log_level.strip().upper()
    os.environ["INGEST_LOG_LEVEL"] = lvl
    configure_logging(lvl)

    n = max(1, args.workers)

    # Recover any stale 'running' jobs left by a previously killed process
    conn = connect()
    try:
        result = conn.execute(
            """
            UPDATE ingest_jobs
            SET status = 'pending', claimed_at = NULL,
                error  = 'recovered after crash'
            WHERE status = 'running'
            """
        )
        conn.commit()
        if result.rowcount:
            _cli_log.warning("Recovered stale running jobs count=%s", result.rowcount)
    finally:
        conn.close()

    _cli_log.info("Spawning workers count=%s log_level=%s", n, lvl)
    processes = [
        multiprocessing.Process(target=run_worker, daemon=False)
        for _ in range(n)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    _cli_log.info("All workers finished.")


def cmd_retry(args: argparse.Namespace) -> None:
    """
    Retry all failed jobs using ccxt as the data source.
    Jobs that succeed are marked 'done'; jobs that fail again remain 'failed'
    with the new error message.
    """
    lvl = args.log_level.strip().upper()
    os.environ["INGEST_LOG_LEVEL"] = lvl
    configure_logging(lvl)

    n = max(1, args.workers)

    conn = connect()
    try:
        rows = conn.execute(
            "SELECT count(*) FROM ingest_jobs WHERE status = 'failed'"
        ).fetchone()
        failed_count = rows[0] if rows else 0
    finally:
        conn.close()

    if failed_count == 0:
        _cli_log.info("No failed jobs to retry.")
        print("No failed jobs to retry.")
        return

    print(f"Retrying {failed_count} failed job(s) via ccxt with {n} worker(s).")
    _cli_log.info(
        "Spawning ccxt retry workers count=%s failed_jobs=%s log_level=%s",
        n, failed_count, lvl,
    )

    processes = [
        multiprocessing.Process(target=run_ccxt_worker, daemon=False)
        for _ in range(n)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    _cli_log.info("All ccxt retry workers finished.")


def cmd_status(_args: argparse.Namespace) -> None:
    conn = connect()
    try:
        rows = conn.execute(
            """
            SELECT status, count(*) AS n
            FROM   ingest_jobs
            GROUP  BY status
            ORDER  BY status
            """
        ).fetchall()
        if not rows:
            print("No jobs found in ingest_jobs.")
            return
        print(f"{'Status':<10}  {'Count':>8}")
        print("-" * 22)
        for status, n in rows:
            print(f"{status:<10}  {n:>8}")
    finally:
        conn.close()


def cmd_verify(args: argparse.Namespace) -> None:
    """
    Check for gaps (missing candles) in the klines table.

    For each (symbol, interval) combination — optionally filtered — uses a
    LAG window function to find consecutive open_times that differ by more
    than the expected interval duration.  Monthly candles ('1mo') are skipped
    because their duration varies.
    """
    conn = connect()
    try:
        conditions: list[str] = []
        params: list = []
        if args.symbol:
            conditions.append("symbol = %s")
            params.append(args.symbol)
        if args.interval:
            conditions.append("interval = %s")
            params.append(args.interval)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        pairs = conn.execute(
            f"SELECT DISTINCT symbol, interval FROM klines {where} ORDER BY symbol, interval",
            params,
        ).fetchall()

        if not pairs:
            print("No klines found matching the given filters.")
            return

        total_gaps = 0
        total_checked = 0

        for sym, inv in pairs:
            inv_ms = INTERVAL_MS.get(inv)
            if inv_ms is None:
                print(f"  [{sym}/{inv}] Skipping (variable-length interval)")
                continue

            gap_rows = conn.execute(
                """
                SELECT
                    prev_open_time,
                    open_time,
                    open_time - prev_open_time AS gap_ms
                FROM (
                    SELECT
                        open_time,
                        LAG(open_time) OVER (ORDER BY open_time) AS prev_open_time
                    FROM klines
                    WHERE symbol = %s AND interval = %s
                ) sub
                WHERE prev_open_time IS NOT NULL
                  AND open_time - prev_open_time > %s
                ORDER BY open_time
                """,
                (sym, inv, inv_ms),
            ).fetchall()

            total_checked += 1

            if gap_rows:
                print(f"[{sym}/{inv}] {len(gap_rows)} gap(s) found:")
                for prev_ms, curr_ms, gap_ms in gap_rows:
                    missing = gap_ms // inv_ms - 1
                    prev_dt = datetime.fromtimestamp(prev_ms / 1000, tz=timezone.utc)
                    curr_dt = datetime.fromtimestamp(curr_ms / 1000, tz=timezone.utc)
                    print(
                        f"  gap after {prev_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC "
                        f"→ {curr_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC "
                        f"(~{missing} missing candle(s), {gap_ms:,} ms)"
                    )
                total_gaps += len(gap_rows)
            else:
                print(f"[{sym}/{inv}] OK — no gaps")

        print()
        if total_gaps:
            print(f"Result: {total_gaps} gap(s) found across {total_checked} checked series.")
        else:
            print(f"Result: all {total_checked} checked series are contiguous.")
    finally:
        conn.close()


def cmd_fill_gaps(args: argparse.Namespace) -> None:
    """
    Detect gaps in the klines table and mark the affected months as 'failed'
    so that 'ingest retry' can re-fetch the missing candles via ccxt.

    For each gap between two consecutive open_times, this command derives all
    missing candle timestamps, maps them to their calendar months, and upserts
    those (symbol, interval, year, month) rows in ingest_jobs with status='failed'.
    Months that have no job row yet are inserted fresh.

    After running this command, run 'ingest retry' to fill in the holes.
    """
    conn = connect()
    try:
        conditions: list[str] = []
        params: list = []
        if args.symbol:
            conditions.append("symbol = %s")
            params.append(args.symbol)
        if args.interval:
            conditions.append("interval = %s")
            params.append(args.interval)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        pairs = conn.execute(
            f"SELECT DISTINCT symbol, interval FROM klines {where} ORDER BY symbol, interval",
            params,
        ).fetchall()

        if not pairs:
            print("No klines found matching the given filters.")
            return

        total_marked = 0

        for sym, inv in pairs:
            inv_ms = INTERVAL_MS.get(inv)
            if inv_ms is None:
                print(f"  [{sym}/{inv}] Skipping (variable-length interval)")
                continue

            gap_rows = conn.execute(
                """
                SELECT prev_open_time, open_time
                FROM (
                    SELECT
                        open_time,
                        LAG(open_time) OVER (ORDER BY open_time) AS prev_open_time
                    FROM klines
                    WHERE symbol = %s AND interval = %s
                ) sub
                WHERE prev_open_time IS NOT NULL
                  AND open_time - prev_open_time > %s
                ORDER BY open_time
                """,
                (sym, inv, inv_ms),
            ).fetchall()

            if not gap_rows:
                continue

            # Collect the unique (year, month) pairs that contain missing candles
            months_to_refetch: set[tuple[int, int]] = set()
            for prev_ms, curr_ms in gap_rows:
                missing_ts = prev_ms + inv_ms
                while missing_ts < curr_ms:
                    candle_dt = datetime.fromtimestamp(missing_ts / 1000, tz=timezone.utc)
                    months_to_refetch.add((candle_dt.year, candle_dt.month))
                    missing_ts += inv_ms

            for year, month in sorted(months_to_refetch):
                conn.execute(
                    """
                    INSERT INTO ingest_jobs (symbol, interval, year, month, status, error)
                    VALUES (%s, %s, %s, %s, 'failed', 'gap detected — queued for ccxt retry')
                    ON CONFLICT (symbol, interval, year, month) DO UPDATE
                        SET status       = 'failed',
                            claimed_at   = NULL,
                            completed_at = NULL,
                            error        = 'gap detected — queued for ccxt retry'
                    """,
                    (sym, inv, year, month),
                )
                print(f"  [{sym}/{inv}] Marked {year:04d}-{month:02d} for ccxt retry")
                total_marked += 1

        conn.commit()
        print()
        if total_marked:
            print(f"Marked {total_marked} job(s) as failed. Run 'ingest retry' to fetch missing candles via ccxt.")
        else:
            print("No gaps found — nothing to queue.")
    finally:
        conn.close()


def cmd_reset(args: argparse.Namespace) -> None:
    conn = connect()
    try:
        if args.hard:
            conn.execute("TRUNCATE klines")
            conn.execute("TRUNCATE ingest_jobs RESTART IDENTITY")
            conn.commit()
            print("Hard reset: klines and ingest_jobs truncated.")
        elif args.failed:
            result = conn.execute(
                """
                UPDATE ingest_jobs
                SET status = 'pending', claimed_at = NULL,
                    completed_at = NULL, error = NULL
                WHERE status = 'failed'
                """
            )
            conn.commit()
            print(f"Re-queued {result.rowcount} failed job(s).")
        else:
            conn.execute("TRUNCATE klines")
            result = conn.execute(
                """
                UPDATE ingest_jobs
                SET status = 'pending', claimed_at = NULL,
                    completed_at = NULL, error = NULL
                """
            )
            conn.commit()
            print(
                f"Reset: klines truncated, "
                f"{result.rowcount} job(s) reset to pending."
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Binance USDT-M futures kline ingester",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # migrate
    sub.add_parser("migrate", help="Apply pending DB migrations")

    # enqueue
    eq = sub.add_parser("enqueue", help="Add jobs to the queue")
    eq.add_argument("--symbol", default="BTCUSDT")
    eq.add_argument(
        "--interval", nargs="+",
        help="One or more intervals (default: all)",
    )
    eq.add_argument(
        "--start", required=True,
        help="Start month: YYYY-MM or YYYY-MM-DD",
    )
    eq.add_argument(
        "--end", required=True,
        help="End month: YYYY-MM or YYYY-MM-DD",
    )

    # run
    rn = sub.add_parser("run", help="Drain the queue with parallel workers")
    rn.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    rn.add_argument(
        "--log-level",
        default=os.environ.get("INGEST_LOG_LEVEL", "INFO"),
        metavar="LEVEL",
        help=(
            "Log level for worker processes: DEBUG, INFO, WARNING, ERROR "
            "(default: INFO, or INGEST_LOG_LEVEL env)"
        ),
    )

    # retry
    rt = sub.add_parser(
        "retry",
        help="Retry all failed jobs via ccxt (Binance USDT-M futures API)",
    )
    rt.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel ccxt worker processes (default: 1)",
    )
    rt.add_argument(
        "--log-level",
        default=os.environ.get("INGEST_LOG_LEVEL", "INFO"),
        metavar="LEVEL",
        help="Log level (default: INFO, or INGEST_LOG_LEVEL env)",
    )

    # status
    sub.add_parser("status", help="Show job counts by status")

    # verify
    vr = sub.add_parser(
        "verify",
        help="Check klines for missing candles (gaps between consecutive open_times)",
    )
    vr.add_argument(
        "--symbol",
        help="Limit check to a specific symbol (e.g. BTCUSDT)",
    )
    vr.add_argument(
        "--interval",
        help="Limit check to a specific interval (e.g. 1h)",
    )

    # fill-gaps
    fg = sub.add_parser(
        "fill-gaps",
        help=(
            "Find gaps in klines and mark affected months as failed "
            "so 'ingest retry' can re-fetch them via ccxt"
        ),
    )
    fg.add_argument(
        "--symbol",
        help="Limit gap search to a specific symbol (e.g. BTCUSDT)",
    )
    fg.add_argument(
        "--interval",
        help="Limit gap search to a specific interval (e.g. 1w)",
    )

    # reset
    rs = sub.add_parser("reset", help="Reset data and/or job queue")
    rs_group = rs.add_mutually_exclusive_group()
    rs_group.add_argument(
        "--hard", action="store_true",
        help="Truncate klines AND delete all jobs (full clean slate)",
    )
    rs_group.add_argument(
        "--failed", action="store_true",
        help="Re-queue only failed jobs (leaves done rows intact)",
    )

    return parser


_COMMANDS = {
    "migrate": cmd_migrate,
    "enqueue": cmd_enqueue,
    "run": cmd_run,
    "retry": cmd_retry,
    "status": cmd_status,
    "verify": cmd_verify,
    "fill-gaps": cmd_fill_gaps,
    "reset": cmd_reset,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command not in ("run", "retry"):
        configure_logging()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
