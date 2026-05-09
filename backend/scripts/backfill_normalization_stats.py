"""Backfill mean.npy/std.npy for already-trained models.

Reconstructs the same train slice that trainer.py used (first 80% of total_steps,
extended by lookback_window candles) and saves stats next to the model file.

Idempotent — skips models that already have stats. Designed to run on a host with
DATABASE_URL set and access to the models directory (typically the training server).

Usage:
    cd backend
    uv run python scripts/backfill_normalization_stats.py \\
        btc_4h_a2c_lb500_3em4_p2_s1 \\
        btc_4h_a2c_lb100_3em4_p2_s0 \\
        btc_4h_a2c_lb500_3em4_p2_s0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingester.db import get_conn
from trainer.config import ModelConfig
from trainer.env.data_feed import load_data_feed
from trainer.env.normalization import fit_stats, save_stats


def _load_completed_run(conn, name: str) -> tuple[ModelConfig, Path]:
    """Return (config, model_path_no_ext) for the most-recent completed run.

    model_path_no_ext is the .zip path with the suffix stripped so that
    save_stats produces <base>.mean.npy / <base>.std.npy next to the .zip file.
    """
    row = conn.execute(
        """
        SELECT mc.config_json, tr.model_path
        FROM model_configs mc
        JOIN training_runs tr ON tr.model_config_id = mc.id
        WHERE mc.name = %s AND tr.status = 'completed'
        ORDER BY tr.completed_at DESC
        LIMIT 1
        """,
        (name,),
    ).fetchone()
    if row is None:
        raise SystemExit(
            f"No completed training run found for model {name!r}. "
            "Ensure the model name is correct and the run has status='completed'."
        )
    cfg_json, model_path_str = row
    cfg = ModelConfig.from_dict(cfg_json)
    # model_path_str ends in .zip (e.g. /…/model.zip); strip extension so
    # save_stats writes model.mean.npy / model.std.npy alongside the .zip.
    model_path_no_ext = Path(model_path_str).with_suffix("")
    return cfg, model_path_no_ext


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Backfill normalization stats (mean.npy / std.npy) for trained models."
    )
    ap.add_argument("model_names", nargs="+", help="One or more model names to backfill.")
    args = ap.parse_args()

    with get_conn() as conn:
        for name in args.model_names:
            cfg, base = _load_completed_run(conn, name)

            mean_path = base.with_suffix(".mean.npy")
            if mean_path.exists():
                print(f"[skip] {name} — stats already at {mean_path}")
                continue

            zip_path = base.with_suffix(".zip")
            if not zip_path.exists():
                print(
                    f"[warn] {name} — model file not found at {zip_path}, skipping",
                    file=sys.stderr,
                )
                continue

            # Reconstruct full-history DataFeed (without applying saved stats).
            full_feed = load_data_feed(cfg, conn)

            # Reproduce the exact train slice trainer.py used:
            #   first 80% of steps, extended by lookback_window for context.
            split_idx = int(full_feed.total_steps * 0.8)
            train_slice = full_feed.raw_features[: split_idx + cfg.lookback_window]

            stats = fit_stats(train_slice)
            save_stats(stats, base)

            print(f"[ok] {name} -> {mean_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
