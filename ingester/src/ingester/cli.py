"""
CLI entry point.

Commands
--------
ingest migrate                          – apply pending DB migrations
ingest enqueue [options]                – add jobs to ingest_jobs
ingest run [--workers N]               – drain the queue with N parallel workers
ingest status                          – show job counts by status
ingest reset                           – truncate klines, reset jobs to pending
ingest reset --hard                    – truncate klines + delete all jobs
ingest reset --failed                  – re-queue only failed jobs
"""
from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing
import sys
from typing import Iterable

from .db import connect, migrate
from .worker import run_worker

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
            print(f"[run] Recovered {result.rowcount} stale running job(s).")
    finally:
        conn.close()

    print(f"[run] Spawning {n} worker(s)…")
    processes = [
        multiprocessing.Process(target=run_worker, daemon=False)
        for _ in range(n)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("[run] All workers finished.")


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

    # status
    sub.add_parser("status", help="Show job counts by status")

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
    "status": cmd_status,
    "reset": cmd_reset,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
