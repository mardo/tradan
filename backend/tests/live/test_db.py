"""Integration tests for live DB writes. Each test runs in a transaction
that is rolled back at teardown so the DB stays clean.

Requires DATABASE_URL pointing at a DB with migrations 001-006 applied.
A `model_configs` row will be created and torn down with the transaction.
"""
from __future__ import annotations

import os
import pytest


pytestmark = pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="needs DATABASE_URL with migrations 001-006 applied",
)


@pytest.fixture
def conn_tx():
    from ingester.db import connect
    conn = connect()
    conn.autocommit = False
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.fixture
def model_config_id(conn_tx):
    row = conn_tx.execute(
        """
        INSERT INTO model_configs (name, config_json)
        VALUES ('test_live', '{}'::jsonb)
        RETURNING id
        """
    ).fetchone()
    return row[0]


def test_start_run_inserts_row(conn_tx, model_config_id):
    from live.db import start_run

    run_id = start_run(
        conn_tx,
        model_config_id=model_config_id,
        exchange="bingx", mode="demo",
        symbol="BTC/USDT:USDT", interval="4h",
        starting_equity=10_000.0,
        config_yaml="exchange: {name: bingx}\n",
        git_sha="deadbeef",
    )
    row = conn_tx.execute(
        "SELECT status, mode, starting_equity FROM live_runs WHERE id = %s",
        (run_id,),
    ).fetchone()
    assert row[0] == "running"
    assert row[1] == "demo"
    assert float(row[2]) == 10_000.0


def test_start_run_blocks_second_running_for_same_model(conn_tx, model_config_id):
    from live.db import start_run

    start_run(
        conn_tx, model_config_id=model_config_id,
        exchange="bingx", mode="demo", symbol="BTC/USDT:USDT", interval="4h",
        starting_equity=10_000.0, config_yaml="x", git_sha="aaa",
    )
    with pytest.raises(Exception):
        start_run(
            conn_tx, model_config_id=model_config_id,
            exchange="bingx", mode="demo", symbol="BTC/USDT:USDT", interval="4h",
            starting_equity=10_000.0, config_yaml="y", git_sha="bbb",
        )


def test_log_action_writes_json_payload(conn_tx, model_config_id):
    from live.db import start_run, log_action

    run_id = start_run(
        conn_tx, model_config_id=model_config_id,
        exchange="bingx", mode="demo", symbol="BTC/USDT:USDT", interval="4h",
        starting_equity=10_000.0, config_yaml="x", git_sha="aaa",
    )
    action_id = log_action(
        conn_tx, live_run_id=run_id, event_type="inference",
        candle_close="2026-05-09T00:00:00Z",
        raw_action=[0.1, -0.2, 0.3],
        decoded_intent={"open": None, "cancels": [], "closes": []},
        account_state={"equity": 10_000.0},
        inference_ms=12,
    )
    row = conn_tx.execute(
        "SELECT raw_action, decoded_intent FROM live_actions WHERE id = %s",
        (action_id,),
    ).fetchone()
    assert row[0] == [0.1, -0.2, 0.3]
    assert row[1] == {"open": None, "cancels": [], "closes": []}


def test_request_stop_sets_kill_flag(conn_tx, model_config_id):
    from live.db import start_run, request_stop

    run_id = start_run(
        conn_tx, model_config_id=model_config_id,
        exchange="bingx", mode="demo", symbol="BTC/USDT:USDT", interval="4h",
        starting_equity=10_000.0, config_yaml="x", git_sha="aaa",
    )
    request_stop(conn_tx, run_id)
    row = conn_tx.execute(
        "SELECT kill_requested FROM live_runs WHERE id = %s", (run_id,),
    ).fetchone()
    assert row[0] is True
