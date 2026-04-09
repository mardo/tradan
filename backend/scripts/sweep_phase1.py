#!/usr/bin/env python3
"""
Phase 1 — BTCUSDT baseline sweep.

Generates and registers 63 model configs:
  7 intervals × 3 algorithms × 3 seeds = 63 runs

All other parameters fixed. Purpose: find which interval+algorithm combos
work at all before varying hyperparameters.

Run from repo: cd backend && uv run python scripts/sweep_phase1.py
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
ALGORITHMS = ["PPO", "SAC", "A2C"]
SEEDS = [42, 123, 456]

PHASE = "p1"
TARGET = "BTCUSDT"
LOOKBACK = 500
LEARNING_RATE = 3e-4
TIMESTEPS = 1_000_000


def main() -> None:
    combos = list(product(INTERVALS, ALGORITHMS, enumerate(SEEDS)))
    print(f"Registering {len(combos)} Phase 1 configs for {TARGET}...")

    for interval, algo, (seed_idx, _seed) in combos:
        name = f"btc_{interval}_{algo.lower()}_{PHASE}_s{seed_idx}"
        config = ModelConfig(
            name=name,
            symbols=[TARGET],
            intervals=[interval],
            columns=list(ALL_KLINE_COLUMNS),
            exchange=ExchangeConfig(),
            lookback_window=LOOKBACK,
            algorithm=algo,
            learning_rate=LEARNING_RATE,
            total_timesteps=TIMESTEPS,
        )
        save_model_config(config)
        print(f"  Registered: {name}")

    print(f"\nDone. {len(combos)} configs registered.")
    print("Train (one process, DB queue): uv run train worker")
    print("Or fan-out workers: bash ../infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
