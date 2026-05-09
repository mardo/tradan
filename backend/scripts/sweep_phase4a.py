#!/usr/bin/env python3
"""
Phase 4A — Seed robustness sweep.

Registers 15 ModelConfigs (3 architectures × 5 explicit seeds) for the top P2
architectures (4h, A2C, lr=3e-4, lookback ∈ {100, 250, 500}, 1M timesteps).

Goal: characterize seed variance so we can reject architectures whose median
seed is unprofitable. Pass criterion (per architecture, applied in
phase4a_summary.py): median holdout PnL > 0 AND ≥3 of 5 seeds positive.

Run from repo: cd backend && uv run python scripts/sweep_phase4a.py
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

LOOKBACKS = [100, 250, 500]
SEEDS = [1001, 2002, 3003, 4004, 5005]
LEARNING_RATE = 3e-4
TIMESTEPS = 1_000_000
INTERVAL = "4h"
ALGO = "A2C"
PHASE = "p4a"


def build_phase4a_configs() -> list[ModelConfig]:
    """Return the 15 ModelConfigs that make up the Phase 4A sweep.

    Pure (no DB I/O); separated so tests can validate the spec without a database.
    """
    configs: list[ModelConfig] = []
    for lookback, (seed_idx, seed) in product(LOOKBACKS, enumerate(SEEDS)):
        name = f"btc_{INTERVAL}_{ALGO.lower()}_lb{lookback}_3em4_{PHASE}_s{seed_idx}"
        configs.append(
            ModelConfig(
                name=name,
                symbols=["BTCUSDT"],
                intervals=[INTERVAL],
                columns=list(ALL_KLINE_COLUMNS),
                exchange=ExchangeConfig(),
                lookback_window=lookback,
                algorithm=ALGO,
                learning_rate=LEARNING_RATE,
                total_timesteps=TIMESTEPS,
                seed=seed,
            )
        )
    return configs


def main() -> None:
    configs = build_phase4a_configs()
    print(f"Registering {len(configs)} Phase 4A configs...")
    for cfg in configs:
        save_model_config(cfg)
        print(f"  Registered: {cfg.name}  seed={cfg.seed}")
    print(f"\nDone. {len(configs)} Phase 4A configs registered.")
    print("Train: uv run train worker  |  Or: bash ../infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
