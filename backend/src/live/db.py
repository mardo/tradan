"""DB writes for live_runs, live_actions, live_orders, live_pnl_snapshots.

Functions take a psycopg connection so callers can decide transaction
boundaries (e.g., LiveRunner uses one connection for the whole run).
"""
from __future__ import annotations

from datetime import datetime

import psycopg
from psycopg.types.json import Json


def start_run(
    conn: psycopg.Connection,
    *,
    model_config_id: int,
    exchange: str,
    mode: str,
    symbol: str,
    interval: str,
    starting_equity: float,
    config_yaml: str,
    git_sha: str,
) -> int:
    row = conn.execute(
        """
        INSERT INTO live_runs (
            model_config_id, exchange, mode, symbol, interval,
            starting_equity, config_yaml, git_sha
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (model_config_id, exchange, mode, symbol, interval,
         starting_equity, config_yaml, git_sha),
    ).fetchone()
    return row[0]


def find_running_run(
    conn: psycopg.Connection,
    *,
    model_config_id: int,
    exchange: str,
) -> int | None:
    row = conn.execute(
        """
        SELECT id FROM live_runs
        WHERE model_config_id = %s AND exchange = %s AND status = 'running'
        """,
        (model_config_id, exchange),
    ).fetchone()
    return row[0] if row else None


def stop_run(
    conn: psycopg.Connection,
    run_id: int,
    *,
    reason: str,
) -> None:
    conn.execute(
        """
        UPDATE live_runs
        SET status = 'stopped', stopped_at = now(), stop_reason = %s
        WHERE id = %s
        """,
        (reason, run_id),
    )


def request_stop(conn: psycopg.Connection, run_id: int) -> None:
    conn.execute(
        "UPDATE live_runs SET kill_requested = TRUE WHERE id = %s",
        (run_id,),
    )


def is_kill_requested(conn: psycopg.Connection, run_id: int) -> bool:
    row = conn.execute(
        "SELECT kill_requested FROM live_runs WHERE id = %s", (run_id,),
    ).fetchone()
    return bool(row and row[0])


def log_action(
    conn: psycopg.Connection,
    *,
    live_run_id: int,
    event_type: str,
    candle_close: datetime | str | None = None,
    raw_action: list[float] | None = None,
    decoded_intent: dict | None = None,
    account_state: dict,
    inference_ms: int | None = None,
    notes: str | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO live_actions (
            live_run_id, event_type, candle_close, raw_action,
            decoded_intent, account_state, inference_ms, notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            live_run_id, event_type, candle_close,
            Json(raw_action) if raw_action is not None else None,
            Json(decoded_intent) if decoded_intent is not None else None,
            Json(account_state),
            inference_ms, notes,
        ),
    ).fetchone()
    return row[0]


def log_order(
    conn: psycopg.Connection,
    *,
    live_run_id: int,
    live_action_id: int | None,
    exchange_order_id: str,
    side: str,
    type: str,
    price: float | None,
    amount: float,
    status: str,
    fill_price: float | None = None,
    fill_amount: float | None = None,
    pnl: float | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO live_orders (
            live_run_id, live_action_id, exchange_order_id,
            side, type, price, amount, status,
            fill_price, fill_amount, pnl
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (live_run_id, live_action_id, exchange_order_id, side, type,
         price, amount, status, fill_price, fill_amount, pnl),
    ).fetchone()
    return row[0]


def log_pnl_snapshot(
    conn: psycopg.Connection,
    *,
    live_run_id: int,
    equity: float,
    realized_pnl: float,
    unrealized_pnl: float,
    open_positions: int,
    open_orders: int,
) -> int:
    row = conn.execute(
        """
        INSERT INTO live_pnl_snapshots (
            live_run_id, equity, realized_pnl, unrealized_pnl,
            open_positions, open_orders
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (live_run_id, equity, realized_pnl, unrealized_pnl,
         open_positions, open_orders),
    ).fetchone()
    return row[0]
