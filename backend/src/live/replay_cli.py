"""Replay gate: run the live code path against historical klines and assert
terminal equity matches the trainer's stored eval result.

Usage:
  uv run live-replay \\
    --model btc_4h_a2c_lb500_3em4_p2_s1 \\
    --start 2024-12-01 --end 2026-04-30 \\
    --tolerance-pct 0.5

Status: depends on
  1. DATABASE_URL pointing at the tradan DB (model_configs, training_runs, klines).
  2. MODELS_DIR env (or --models-dir) pointing at the directory containing
     <model>/<run_id>/model.zip and the matching mean.npy / std.npy
     (per the trainer save convention).
  3. Phase A.4 part 3 (evaluator wired to load_stats) and A.10 (re-baselined
     training_runs.final_balance values). Until those run on the training
     server, the gate's "expected_terminal_equity" will read from training_runs
     rows that were produced under eval-time normalization. Treat early
     divergences with that lens until the baselines are refreshed.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ingester.db import connect
from live.exchange.replay import ReplayAdapter
from live.model_runner import ModelRunner
from live.runner import run_replay
from trainer.config import ModelConfig
from trainer.env.normalization import load_stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="live-replay")
    p.add_argument("--model", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--tolerance-pct", type=float, default=0.5,
                   help="abs(live - eval) / starting_equity * 100 must be ≤ this")
    p.add_argument("--models-dir", default=os.environ.get("MODELS_DIR"),
                   help="Directory containing <model>/<run_id>/model.zip and stats files. "
                        "Defaults to MODELS_DIR env.")
    return p.parse_args(argv)


def _load_model_config(conn, name: str) -> tuple[int, ModelConfig]:
    row = conn.execute(
        "SELECT id, config_json FROM model_configs WHERE name = %s", (name,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"model {name!r} not found in model_configs")
    cfg = ModelConfig.from_dict({"name": name, **row[1]})
    return row[0], cfg


def _expected_terminal_equity_and_path(conn, model_config_id: int) -> tuple[float, str]:
    """Return (expected_final_balance, model_path) from the most-recent eval run."""
    row = conn.execute(
        """
        SELECT final_balance, model_path
        FROM training_runs
        WHERE model_config_id = %s AND run_type = 'evaluate'
              AND status = 'completed'
        ORDER BY id DESC
        LIMIT 1
        """,
        (model_config_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(
            "no completed evaluate run for this model in training_runs"
        )
    return float(row[0]), str(row[1])


def _load_kline_window(
    conn, *, symbol: str, interval: str,
    start: datetime, end: datetime, columns: list[str],
):
    rows = conn.execute(
        f"""
        SELECT open_time, {", ".join(columns)} FROM klines
        WHERE symbol = %s AND interval = %s
              AND open_time >= %s AND open_time < %s
        ORDER BY open_time
        """,
        (symbol, interval, start, end),
    ).fetchall()
    if not rows:
        raise SystemExit("empty kline window — check --start/--end vs DB contents")
    df = pd.DataFrame(rows, columns=["open_time"] + columns)
    timestamps = df["open_time"].values.astype(np.int64)
    features = df[columns].values.astype(np.float32)
    price_columns = {c: i for i, c in enumerate(columns)}
    return timestamps, features, price_columns


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.models_dir:
        print("error: provide --models-dir or set MODELS_DIR env", file=sys.stderr)
        return 2
    models_dir = Path(args.models_dir)

    with connect() as conn:
        model_id, cfg = _load_model_config(conn, args.model)
        expected, model_path_str = _expected_terminal_equity_and_path(conn, model_id)

        symbol_db = cfg.symbols[0]   # e.g. "BTCUSDT"
        ts, feats, price_cols = _load_kline_window(
            conn,
            symbol=symbol_db,
            interval=cfg.intervals[0],
            start=datetime.fromisoformat(args.start),
            end=datetime.fromisoformat(args.end),
            columns=cfg.columns,
        )

    # The model path stored in training_runs is the absolute path to model.zip;
    # mean.npy / std.npy live alongside it.
    model_path = Path(model_path_str)
    stats_base = model_path.with_suffix("")   # strip .zip → /…/<run_id>/model
    stats = load_stats(stats_base)

    adapter = ReplayAdapter.from_arrays(
        timestamps=ts,
        features=feats,
        price_columns=price_cols,
        symbol="BTC/USDT:USDT",
        interval=cfg.intervals[0],
        starting_balance=cfg.initial_balance,
        exchange_config=cfg.exchange,
    )
    model_runner = ModelRunner(
        model_path=model_path,
        algorithm=cfg.algorithm,
    )

    result = run_replay(
        adapter=adapter,
        model_runner=model_runner,
        model_config=cfg,
        stats=stats,
    )

    diff = abs(result.final_equity - expected)
    diff_pct = (diff / cfg.initial_balance) * 100.0
    passed = diff_pct <= args.tolerance_pct

    print(f"model:           {args.model}")
    print(f"steps:           {result.total_steps}")
    print(f"expected equity: {expected:.4f}")
    print(f"live equity:     {result.final_equity:.4f}")
    print(f"abs diff:        {diff:.6f}  ({diff_pct:.4f}% of starting equity)")
    print(f"tolerance:       {args.tolerance_pct:.4f}%")
    print(f"result:          {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def replay_main() -> int:
    """Entry point for the `live-replay` console script."""
    return main()


if __name__ == "__main__":
    sys.exit(replay_main())
