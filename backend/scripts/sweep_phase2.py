#!/usr/bin/env python3
"""
Phase 2 — Hyperparameter expansion.

Reads top 5 Phase 1 winners from DB (by holdout Sharpe, after winner filters),
then generates variants across:
  4 lookback windows × 3 learning rates × 3 seeds = 36 runs per winner config
  5 winners × 36 = up to 180 runs total

Preserves interval + algorithm from each winner; varies lookback and learning rate.

Requires Phase 1 model names to contain `_p1_` (see sweep_phase1.py).

Run from repo: cd backend && uv run python scripts/sweep_phase2.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

from ingester.db import connect
from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

LOOKBACKS = [100, 250, 500, 1000]
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
SEEDS = [42, 123, 456]
TIMESTEPS = 1_000_000
PHASE = "p2"
TOP_N = 5


def get_phase1_winners(conn, top_n: int) -> list[dict]:
    """Return top N Phase 1 configs by holdout Sharpe after applying winner filters."""
    rows = conn.execute(
        """
        SELECT
            mc.name,
            mc.config_json,
            tr_eval.sharpe_ratio
        FROM model_configs mc
        JOIN training_runs tr_train ON tr_train.model_config_id = mc.id
            AND tr_train.run_type = 'train' AND tr_train.status = 'completed'
        JOIN training_runs tr_eval ON tr_eval.model_config_id = mc.id
            AND tr_eval.run_type = 'evaluate' AND tr_eval.status = 'completed'
        WHERE mc.name LIKE '%_p1_%'
          AND tr_eval.total_trades > 10
          AND tr_eval.total_pnl > 0
          AND tr_eval.max_drawdown < 0.25
          AND tr_eval.total_pnl / NULLIF(tr_train.total_pnl, 0) > 0.5
        ORDER BY tr_eval.sharpe_ratio DESC
        LIMIT %s
        """,
        (top_n,),
    ).fetchall()
    return [{"name": r[0], "config": r[1], "sharpe": r[2]} for r in rows]


def lr_slug(lr: float) -> str:
    """Convert a learning rate float to a compact string: 3e-4 -> 3em4, 1e-3 -> 1em3."""
    s = f"{lr:.0e}"  # "3e-04"
    return s.replace("e-0", "em").replace("e-", "em").replace(".", "p")


def main() -> None:
    conn = connect()
    try:
        winners = get_phase1_winners(conn, TOP_N)
    finally:
        conn.close()

    if not winners:
        print("No Phase 1 winners found. Run evaluate_winners.sh after Phase 1 first.")
        sys.exit(1)

    print(f"Found {len(winners)} Phase 1 winners. Generating Phase 2 configs...")
    count = 0

    for winner in winners:
        base_cfg = ModelConfig.from_dict(winner["config"])
        interval = base_cfg.intervals[0]
        algo = base_cfg.algorithm

        for lookback, lr, (seed_idx, _seed) in product(LOOKBACKS, LEARNING_RATES, enumerate(SEEDS)):
            name = f"btc_{interval}_{algo.lower()}_lb{lookback}_{lr_slug(lr)}_{PHASE}_s{seed_idx}"
            config = ModelConfig(
                name=name,
                symbols=["BTCUSDT"],
                intervals=[interval],
                columns=list(ALL_KLINE_COLUMNS),
                exchange=ExchangeConfig(),
                lookback_window=lookback,
                algorithm=algo,
                learning_rate=lr,
                total_timesteps=TIMESTEPS,
            )
            save_model_config(config)
            print(f"  Registered: {name}")
            count += 1

    print(f"\nDone. {count} Phase 2 configs registered.")
    print("Train: uv run train worker  |  Or: bash ../infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
