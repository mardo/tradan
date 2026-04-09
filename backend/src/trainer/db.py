from __future__ import annotations

import json

import psycopg

from ingester.db import connect
from trainer.config import ModelConfig


def save_model_config(config: ModelConfig) -> int:
    conn = connect()
    try:
        row = conn.execute(
            """
            INSERT INTO model_configs (name, config_json)
            VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET config_json = EXCLUDED.config_json
            RETURNING id
            """,
            (config.name, json.dumps(config.to_dict())),
        ).fetchone()
        conn.commit()
        return row[0]
    finally:
        conn.close()


def load_model_config(name: str) -> ModelConfig | None:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT config_json FROM model_configs WHERE name = %s", (name,)
        ).fetchone()
        if row is None:
            return None
        return ModelConfig.from_dict(row[0])
    finally:
        conn.close()


def get_model_config_id(name: str) -> int | None:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT id FROM model_configs WHERE name = %s", (name,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def list_model_configs() -> list[dict]:
    conn = connect()
    try:
        rows = conn.execute(
            """
            SELECT mc.name, mc.created_at,
                   count(tr.id) AS run_count,
                   max(tr.total_pnl) AS best_pnl
            FROM model_configs mc
            LEFT JOIN training_runs tr ON tr.model_config_id = mc.id
                AND tr.status = 'completed'
            GROUP BY mc.id
            ORDER BY mc.name
            """
        ).fetchall()
        return [
            {"name": r[0], "created_at": r[1], "run_count": r[2], "best_pnl": r[3]}
            for r in rows
        ]
    finally:
        conn.close()


def create_training_run(
    model_config_id: int, run_type: str, algorithm: str
) -> int:
    conn = connect()
    try:
        row = conn.execute(
            """
            INSERT INTO training_runs (model_config_id, run_type, algorithm)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (model_config_id, run_type, algorithm),
        ).fetchone()
        conn.commit()
        return row[0]
    finally:
        conn.close()


def complete_training_run(
    run_id: int,
    *,
    final_balance: float,
    final_equity: float,
    total_pnl: float,
    total_trades: int,
    win_rate: float,
    max_drawdown: float,
    sharpe_ratio: float,
    model_path: str,
) -> None:
    conn = connect()
    try:
        conn.execute(
            """
            UPDATE training_runs
            SET status = 'completed', completed_at = now(),
                final_balance = %s, final_equity = %s, total_pnl = %s,
                total_trades = %s, win_rate = %s, max_drawdown = %s,
                sharpe_ratio = %s, model_path = %s
            WHERE id = %s
            """,
            (final_balance, final_equity, total_pnl, total_trades,
             win_rate, max_drawdown, sharpe_ratio, model_path, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def fail_training_run(run_id: int, error: str) -> None:
    conn = connect()
    try:
        conn.execute(
            """
            UPDATE training_runs
            SET status = 'failed', completed_at = now(), error = %s
            WHERE id = %s
            """,
            (error, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_pnl_snapshots(
    conn: psycopg.Connection, snapshots: list[dict]
) -> None:
    if not snapshots:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO pnl_snapshots
                (training_run_id, step, candle_time, balance, equity,
                 unrealized_pnl, open_position_count, open_order_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (s["training_run_id"], s["step"], s["candle_time"],
                 s["balance"], s["equity"], s["unrealized_pnl"],
                 s["open_position_count"], s["open_order_count"])
                for s in snapshots
            ],
        )
    conn.commit()


def update_model_path(run_id: int, model_path: str) -> None:
    conn = connect()
    try:
        conn.execute(
            "UPDATE training_runs SET model_path = %s WHERE id = %s",
            (model_path, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def claim_pending_model(stale_minutes: int = 120) -> tuple[str, ModelConfig] | None:
    """Atomically claims one unclaimed, untrained model. Returns (name, config) or None.

    A model is claimable when:
    - claimed_at IS NULL or older than stale_minutes (worker crashed before starting a run)
    - AND no completed training run exists
    - AND no recently-started running run exists (also guarded by stale_minutes)
    """
    conn = connect()
    try:
        row = conn.execute(
            """
            WITH candidate AS (
                SELECT id FROM model_configs
                WHERE (claimed_at IS NULL OR claimed_at < now() - %s::interval)
                  AND NOT EXISTS (
                      SELECT 1 FROM training_runs tr
                      WHERE tr.model_config_id = model_configs.id
                        AND (tr.status = 'completed'
                             OR (tr.status = 'running'
                                 AND tr.started_at > now() - %s::interval))
                  )
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE model_configs SET claimed_at = now(), last_ping = now()
            FROM candidate
            WHERE model_configs.id = candidate.id
            RETURNING model_configs.name, model_configs.config_json
            """,
            (f"{stale_minutes} minutes", f"{stale_minutes} minutes"),
        ).fetchone()
        conn.commit()
        if row is None:
            return None
        return row[0], ModelConfig.from_dict(row[1])
    finally:
        conn.close()


def ping_model_claim(name: str) -> None:
    """Update last_ping to now() for the given model config (keepalive from worker)."""
    conn = connect()
    try:
        conn.execute(
            "UPDATE model_configs SET last_ping = now() WHERE name = %s",
            (name,),
        )
        conn.commit()
    finally:
        conn.close()


def list_stale_claims(older_than_seconds: int) -> list[dict]:
    """Return model_configs whose claim has gone silent for longer than older_than_seconds.

    Staleness is measured against last_ping when available, falling back to claimed_at
    for models claimed before the ping column was added.
    """
    conn = connect()
    try:
        rows = conn.execute(
            """
            SELECT mc.name,
                   mc.claimed_at,
                   mc.last_ping,
                   EXTRACT(EPOCH FROM (now() - COALESCE(mc.last_ping, mc.claimed_at)))::int
                       AS silent_seconds
            FROM model_configs mc
            WHERE mc.claimed_at IS NOT NULL
              AND COALESCE(mc.last_ping, mc.claimed_at) < now() - (%s || ' seconds')::interval
              AND NOT EXISTS (
                  SELECT 1 FROM training_runs tr
                  WHERE tr.model_config_id = mc.id
                    AND tr.status = 'running'
                    AND tr.started_at > now() - (%s || ' seconds')::interval
              )
            ORDER BY COALESCE(mc.last_ping, mc.claimed_at)
            """,
            (older_than_seconds, older_than_seconds),
        ).fetchall()
        return [
            {
                "name": r[0],
                "claimed_at": r[1],
                "last_ping": r[2],
                "silent_seconds": r[3],
            }
            for r in rows
        ]
    finally:
        conn.close()


def release_stale_claims(older_than_seconds: int) -> int:
    """Null out claimed_at and last_ping for stale claims. Returns rows updated."""
    conn = connect()
    try:
        cur = conn.execute(
            """
            UPDATE model_configs
            SET claimed_at = NULL, last_ping = NULL
            WHERE claimed_at IS NOT NULL
              AND COALESCE(last_ping, claimed_at) < now() - (%s || ' seconds')::interval
              AND NOT EXISTS (
                  SELECT 1 FROM training_runs tr
                  WHERE tr.model_config_id = model_configs.id
                    AND tr.status = 'running'
                    AND tr.started_at > now() - (%s || ' seconds')::interval
              )
            """,
            (older_than_seconds, older_than_seconds),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def get_training_run(run_id: int) -> dict | None:
    conn = connect()
    try:
        cur = conn.execute(
            """
            SELECT tr.*, mc.name AS model_name
            FROM training_runs tr
            JOIN model_configs mc ON mc.id = tr.model_config_id
            WHERE tr.id = %s
            """,
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [desc.name for desc in cur.description]
        return dict(zip(cols, row))
    finally:
        conn.close()
