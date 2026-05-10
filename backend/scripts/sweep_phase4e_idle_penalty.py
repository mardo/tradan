#!/usr/bin/env python3
"""
Phase 4E — Idle-step penalty sweep.

Mirrors the Phase 4D sweep shape (3 architectures × 5 paired seeds × 1M
timesteps, A2C, lr=3e-4, env audit defaults) but adds an additive
per-step penalty on idle steps:

  reward = Δequity − idle_step_penalty_usd  (when no positions and no orders)

Two penalty magnitudes are tested:

  0.05 USD/step  → ~5 bps/step at $10K initial balance, light pressure
  0.5  USD/step  → ~50 bps/step, strong pressure

The hypothesis (see docs/plans/2026-05-09-phase4-training-plan.md, F4
deferred finding) is that reward = Δ equity alone fails to differentiate
"do nothing" from "small loss", which under entropy decay collapses lb100
policies to zero trades. A small idle penalty creates a clear gradient
toward "trade something" without forcing aggressive sizing.

Seeds are paired with 4A/4C/4D (1001..5005) so each (arch × penalty)
cell has a clean per-seed delta vs the 4D baseline.

Total: 3 archs × 2 penalty values × 5 seeds = 30 runs at 1M timesteps.

Run from repo: cd backend && uv run python scripts/sweep_phase4e_idle_penalty.py
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
# (slug, idle_step_penalty_usd) pairs. Slugs go into the model name so the
# DB-side LIKE filters can pick out a single penalty cell:
#   0.05 -> "idle05"  (light)
#   0.5  -> "idle5"   (strong)
PENALTY_VALUES: list[tuple[str, float]] = [("idle05", 0.05), ("idle5", 0.5)]
SEEDS = [1001, 2002, 3003, 4004, 5005]
LEARNING_RATE = 3e-4
TIMESTEPS = 1_000_000
INTERVAL = "4h"
ALGO = "A2C"
PHASE = "p4e"


def build_phase4e_configs() -> list[ModelConfig]:
    """Return the 30 ModelConfigs that make up the Phase 4E sweep.

    Each config inherits the post-audit ExchangeConfig defaults
    (max_leverage=10, max_position_size_pct=0.25, max_drawdown_pct=0.5)
    and additionally sets idle_step_penalty_usd to one of the test values.

    Pure (no DB I/O); separated so tests can validate the spec without a DB.
    """
    configs: list[ModelConfig] = []
    for lookback, (slug, penalty), (seed_idx, seed) in product(
        LOOKBACKS, PENALTY_VALUES, enumerate(SEEDS)
    ):
        name = (
            f"btc_{INTERVAL}_{ALGO.lower()}_lb{lookback}_3em4_{slug}_{PHASE}_s{seed_idx}"
        )
        configs.append(
            ModelConfig(
                name=name,
                symbols=["BTCUSDT"],
                intervals=[INTERVAL],
                columns=list(ALL_KLINE_COLUMNS),
                exchange=ExchangeConfig(idle_step_penalty_usd=penalty),
                lookback_window=lookback,
                algorithm=ALGO,
                learning_rate=LEARNING_RATE,
                total_timesteps=TIMESTEPS,
                seed=seed,
            )
        )
    return configs


def main() -> None:
    configs = build_phase4e_configs()
    print(f"Registering {len(configs)} Phase 4E configs (idle-step penalty)...")
    for cfg in configs:
        save_model_config(cfg)
        print(
            f"  Registered: {cfg.name}  seed={cfg.seed}  "
            f"idle_pen={cfg.exchange.idle_step_penalty_usd}"
        )
    print(f"\nDone. {len(configs)} Phase 4E configs registered.")
    print("Train: uv run train worker  |  Or: bash ../infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
