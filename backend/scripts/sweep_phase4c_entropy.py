#!/usr/bin/env python3
"""
Phase 4C — A2C entropy regularization sweep.

Registers 5 ModelConfigs (lb500 × 5 seeds, A2C, lr=3e-4, ent_coef=0.01, 2M steps)
to test whether entropy regularization rescues lb500 from the holdout failures
observed in Phase 4A.

Seeds are paired with the 4A run (same seed values 1001..5005 used for the same
lookback) so the per-seed delta isolates the effect of entropy regularization
plus extended training.

Pass criterion (in phase4c_entropy_summary.py): median holdout PnL > 0 AND
≥3 of 5 seeds positive — same as 4A.

Run from repo: cd backend && uv run python scripts/sweep_phase4c_entropy.py
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

LOOKBACK = 500
SEEDS = [1001, 2002, 3003, 4004, 5005]
LEARNING_RATE = 3e-4
ENT_COEF = 0.01
TIMESTEPS = 2_000_000
INTERVAL = "4h"
ALGO = "A2C"
PHASE = "p4c"
ENT_SLUG = "ent01"


def build_phase4c_entropy_configs() -> list[ModelConfig]:
    """Return the 5 ModelConfigs for the Phase 4C entropy sweep.

    Pure (no DB I/O); separated so tests can validate the spec without a DB.
    """
    configs: list[ModelConfig] = []
    for seed_idx, seed in enumerate(SEEDS):
        name = (
            f"btc_{INTERVAL}_{ALGO.lower()}_lb{LOOKBACK}_3em4_{ENT_SLUG}"
            f"_{PHASE}_s{seed_idx}"
        )
        configs.append(
            ModelConfig(
                name=name,
                symbols=["BTCUSDT"],
                intervals=[INTERVAL],
                columns=list(ALL_KLINE_COLUMNS),
                exchange=ExchangeConfig(),
                lookback_window=LOOKBACK,
                algorithm=ALGO,
                learning_rate=LEARNING_RATE,
                total_timesteps=TIMESTEPS,
                seed=seed,
                ent_coef=ENT_COEF,
            )
        )
    return configs


def main() -> None:
    configs = build_phase4c_entropy_configs()
    print(f"Registering {len(configs)} Phase 4C entropy configs...")
    for cfg in configs:
        save_model_config(cfg)
        print(f"  Registered: {cfg.name}  seed={cfg.seed}  ent_coef={cfg.ent_coef}")
    print(f"\nDone. {len(configs)} Phase 4C configs registered.")
    print("Train: uv run train worker  |  Or: bash ../infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
