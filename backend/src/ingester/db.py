"""
Database connection factory and migration runner.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg

from .config import get_database_url

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


def _exec_migration_sql(conn: psycopg.Connection, sql: str) -> None:
    """Run a migration file as one or more statements (split on ';')."""
    for part in sql.split(";"):
        lines = [
            ln
            for ln in part.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ]
        stmt = "\n".join(lines).strip()
        if stmt:
            conn.execute(stmt)


def connect() -> psycopg.Connection:
    """Open a new synchronous psycopg3 connection."""
    return psycopg.connect(get_database_url())


@contextmanager
def get_conn() -> Generator[psycopg.Connection, None, None]:
    """Yield an open connection and close it on exit (commit/rollback handled by caller)."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


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
                _exec_migration_sql(conn, sql)
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
