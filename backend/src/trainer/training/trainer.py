from __future__ import annotations

import os
from pathlib import Path

import torch
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)

from ingester.db import connect
from trainer.config import ModelConfig
from trainer.db import (
    complete_training_run,
    create_training_run,
    fail_training_run,
    get_model_config_id,
    save_pnl_snapshots,
)
from trainer.env.data_feed import DataFeed, load_data_feed
from trainer.env.trading_env import TradingEnv

ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
}

MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "trained_models"


class PnlSnapshotCallback(BaseCallback):
    def __init__(
        self, env: TradingEnv, run_id: int, interval: int = 100, verbose: int = 0
    ) -> None:
        super().__init__(verbose)
        self.env = env
        self.run_id = run_id
        self.interval = interval
        self._buffer: list[dict] = []
        self._last_flushed = 0

    def _on_step(self) -> bool:
        if self.env.pnl_history and len(self.env.pnl_history) % self.interval == 0:
            new_entries = self.env.pnl_history[self._last_flushed:]
            for entry in new_entries:
                self._buffer.append({**entry, "training_run_id": self.run_id})
            self._last_flushed = len(self.env.pnl_history)

            if len(self._buffer) >= 500:
                self._flush()
        return True

    def _flush(self) -> None:
        if not self._buffer:
            return
        conn = connect()
        try:
            save_pnl_snapshots(conn, self._buffer)
            self._buffer.clear()
        finally:
            conn.close()

    def _on_training_end(self) -> None:
        new_entries = self.env.pnl_history[self._last_flushed:]
        for entry in new_entries:
            self._buffer.append({**entry, "training_run_id": self.run_id})
        self._flush()


def compute_metrics(env: TradingEnv) -> dict:
    if not env.pnl_history:
        return {
            "final_balance": env.account.balance,
            "final_equity": env.account.balance,
            "total_pnl": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    equities = [h["equity"] for h in env.pnl_history]
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    import numpy as np
    returns = np.diff(equities) / (np.array(equities[:-1]) + 1e-9)
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 1e-9:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24))

    return {
        "final_balance": env.account.balance,
        "final_equity": equities[-1] if equities else env.account.balance,
        "total_pnl": equities[-1] - env.config.initial_balance if equities else 0.0,
        "total_trades": env.exchange.total_trades,
        "win_rate": env.exchange.win_rate,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
    }


def train_model(
    config: ModelConfig,
    *,
    algo_override: str | None = None,
    timesteps_override: int | None = None,
) -> int:
    # Cap PyTorch's intraop thread pool so parallel workers don't saturate all cores.
    # OMP_NUM_THREADS is set by run_sweep.sh to (nproc / worker_count).
    # When TRADAN_FULL_THREADS=1 (sequential mode), skip capping so PyTorch uses all cores.
    if not os.environ.get("TRADAN_FULL_THREADS"):
        _cpu_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
        torch.set_num_threads(_cpu_threads)
        torch.set_num_interop_threads(1)

    algorithm = algo_override or config.algorithm
    total_timesteps = timesteps_override or config.total_timesteps

    algo_cls = ALGO_MAP.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use one of: {list(ALGO_MAP)}")

    config_id = get_model_config_id(config.name)
    if config_id is None:
        raise ValueError(f"Model '{config.name}' not found in DB. Run create-model first.")

    run_id = create_training_run(config_id, "train", algorithm)
    print(f"Training run #{run_id} started: model={config.name} algo={algorithm} steps={total_timesteps}")

    try:
        conn = connect()
        try:
            data_feed = load_data_feed(config, conn)
        finally:
            conn.close()

        split_idx = int(data_feed.total_steps * 0.8)
        train_timestamps = data_feed.timestamps[: split_idx + config.lookback_window]
        train_features = data_feed.raw_features[: split_idx + config.lookback_window]

        train_feed = DataFeed(
            timestamps=train_timestamps,
            features=train_features,
            lookback=config.lookback_window,
            price_columns=data_feed.price_columns,
        )

        env = TradingEnv(config=config, data_feed=train_feed)

        model_dir = MODELS_DIR / config.name / str(run_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = CheckpointCallback(
            save_freq=100_000,
            save_path=str(model_dir),
            name_prefix="checkpoint",
        )
        pnl_cb = PnlSnapshotCallback(
            env=env, run_id=run_id, interval=config.snapshot_interval
        )

        model = algo_cls(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            verbose=1,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, pnl_cb],
        )

        model_path = str(model_dir / "model.zip")
        model.save(model_path)

        metrics = compute_metrics(env)
        complete_training_run(run_id, model_path=model_path, **metrics)

        print(f"Training run #{run_id} completed.")
        print(f"  Final equity: ${metrics['final_equity']:.2f}")
        print(f"  Total PnL:    ${metrics['total_pnl']:.2f}")
        print(f"  Win rate:     {metrics['win_rate']*100:.1f}%")
        print(f"  Max drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Model saved:  {model_path}")
        return run_id

    except Exception as e:
        fail_training_run(run_id, str(e))
        print(f"Training run #{run_id} FAILED: {e}")
        raise
