"""
Database connection factory and migration runner.
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg

from .config import get_database_url

MIGRATIONS_DIR = Path(__file__).parent.parent.parent.parent / "migrations"


def connect() -> psycopg.Connection:
    """Open a new synchronous psycopg3 connection."""
    return psycopg.connect(get_database_url())


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

_ENSURE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS _migrations (
    name       TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def migrate(conn: psycopg.Connection | None = None) -> list[str]:
    """
    Apply all pending SQL migration files from the migrations/ directory.

    Returns the list of migration names that were applied in this call.
    """
    own_conn = conn is None
    if own_conn:
        conn = connect()

    try:
        with conn.transaction():
            conn.execute(_ENSURE_MIGRATIONS_TABLE)

        sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        applied: list[str] = []

        for path in sql_files:
            name = path.name
            row = conn.execute(
                "SELECT 1 FROM _migrations WHERE name = %s", (name,)
            ).fetchone()
            if row:
                continue

            sql = path.read_text()
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO _migrations (name) VALUES (%s)", (name,)
                )
            applied.append(name)
            print(f"[migrate] Applied: {name}")

        if not applied:
            print("[migrate] Nothing to apply — schema is up to date.")

        return applied
    finally:
        if own_conn:
            conn.close()
