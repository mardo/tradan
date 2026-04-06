#!/usr/bin/env python3
"""
Phase 3 — Long training.

Reads top 5 Phase 2 winners from DB, retrains them with 5M timesteps each.
3 seeds per config = up to 15 runs total.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND / "src"))

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")

from ingester.db import connect
from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

SEEDS = [42, 123, 456]
TIMESTEPS = 5_000_000
PHASE = "p3"
TOP_N = 5


def get_phase2_winners(conn, top_n: int) -> list[dict]:
    """Return top N Phase 2 configs by holdout Sharpe after applying winner filters."""
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
        WHERE mc.name LIKE '%_p2_%'
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


def main() -> None:
    conn = connect()
    try:
        winners = get_phase2_winners(conn, TOP_N)
    finally:
        conn.close()

    if not winners:
        print("No Phase 2 winners found. Run evaluate_winners.sh after Phase 2 first.")
        sys.exit(1)

    print(f"Found {len(winners)} Phase 2 winners. Generating Phase 3 long-training configs...")
    count = 0

    for winner in winners:
        base_cfg = ModelConfig.from_dict(winner["config"])

        for seed_idx in range(len(SEEDS)):
            # Build a clean Phase 3 name from the winner's config attributes
            interval = base_cfg.intervals[0]
            algo = base_cfg.algorithm.lower()
            lb = base_cfg.lookback_window
            lr = base_cfg.learning_rate
            # Reuse lr_slug logic inline
            lr_str = f"{base_cfg.learning_rate:.0e}".replace("e-0", "em").replace("e-", "em")
            name = f"btc_{interval}_{algo}_lb{lb}_{lr_str}_{PHASE}_s{seed_idx}"
            config = ModelConfig(
                name=name,
                symbols=base_cfg.symbols,
                intervals=base_cfg.intervals,
                columns=base_cfg.columns,
                exchange=base_cfg.exchange,
                lookback_window=base_cfg.lookback_window,
                algorithm=base_cfg.algorithm,
                learning_rate=base_cfg.learning_rate,
                total_timesteps=TIMESTEPS,
            )
            save_model_config(config)
            print(f"  Registered: {name}")
            count += 1

    print(f"\nDone. {count} Phase 3 configs registered (5M timesteps each).")
    print("Run: bash /opt/tradan/infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
