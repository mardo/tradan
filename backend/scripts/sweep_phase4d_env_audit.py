#!/usr/bin/env python3
"""
Phase 4D — Re-run seed-robustness sweep under the new env risk caps.

Mirrors the Phase 4A sweep exactly (3 architectures × 5 paired seeds × 1M
timesteps, A2C, lr=3e-4) but with each config registered AFTER the env
audit, so they carry the new ExchangeConfig defaults:

  max_leverage = 10        (was 125)
  max_position_size_pct = 0.25  (new — caps single-trade margin)
  max_drawdown_pct = 0.5   (new — early-terminates episode at 50% drawdown)

Goal: clean A/B vs Phase 4A. Same architectures, same seeds, same algo,
same timesteps — only the env risk controls differ. Decision criterion
matches 4A: median holdout PnL > 0 AND ≥3 of 5 seeds positive per arch.

See docs/plans/2026-05-10-phase4-env-audit-design.md for the audit
rationale.

Run from repo: cd backend && uv run python scripts/sweep_phase4d_env_audit.py
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
PHASE = "p4d"


def build_phase4d_configs() -> list[ModelConfig]:
    """Return the 15 ModelConfigs that make up the Phase 4D sweep.

    Each ModelConfig instantiates a fresh ExchangeConfig(), so the new
    audit defaults (max_leverage=10, max_position_size_pct=0.25,
    max_drawdown_pct=0.5) are baked in.

    Pure (no DB I/O); separated so tests can validate the spec without a DB.
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
    configs = build_phase4d_configs()
    print(f"Registering {len(configs)} Phase 4D configs (under new env caps)...")
    for cfg in configs:
        save_model_config(cfg)
        print(
            f"  Registered: {cfg.name}  seed={cfg.seed}  "
            f"max_lev={cfg.exchange.max_leverage}  "
            f"max_pos_pct={cfg.exchange.max_position_size_pct}  "
            f"max_dd={cfg.exchange.max_drawdown_pct}"
        )
    print(f"\nDone. {len(configs)} Phase 4D configs registered.")
    print("Train: uv run train worker  |  Or: bash ../infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
