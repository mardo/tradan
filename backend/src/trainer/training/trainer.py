from __future__ import annotations

import os
import queue
import threading
import time
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
    ping_model_claim,
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
    """Periodically persists PnL snapshots to the database.

    DB writes happen on a dedicated background thread so the training loop is
    never blocked waiting for network I/O to the base droplet.
    """

    def __init__(
        self, env: TradingEnv, run_id: int, interval: int = 100, verbose: int = 0
    ) -> None:
        super().__init__(verbose)
        self.env = env
        self.run_id = run_id
        self.interval = interval
        self._buffer: list[dict] = []
        self._last_flushed = 0
        self._write_queue: queue.Queue[list[dict] | None] = queue.Queue()
        self._db_thread = threading.Thread(
            target=self._db_writer, daemon=True, name=f"pnl-writer-{run_id}"
        )
        self._db_thread.start()

    def _db_writer(self) -> None:
        """Drain the write queue and persist batches; runs on a background thread."""
        conn = connect()
        try:
            while True:
                batch = self._write_queue.get()
                try:
                    if batch is None:  # sentinel — training finished
                        return
                    try:
                        save_pnl_snapshots(conn, batch)
                    except Exception as exc:
                        print(f"  [pnl-writer] DB write failed: {exc}")
                finally:
                    self._write_queue.task_done()
        finally:
            conn.close()

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
        # Hand the batch off to the background writer; don't block the training thread.
        self._write_queue.put(list(self._buffer))
        self._buffer.clear()

    def _on_training_end(self) -> None:
        new_entries = self.env.pnl_history[self._last_flushed:]
        for entry in new_entries:
            self._buffer.append({**entry, "training_run_id": self.run_id})
        self._flush()
        # Send sentinel and wait for all queued writes to land before returning.
        self._write_queue.put(None)
        self._db_thread.join()


class TrainingProgressCallback(BaseCallback):
    """Renders an in-place progress bar with ETA.

    Example:
      [================------------------------]  40.0%  400,000/1,000,000 steps  eta 12m 34s
    """

    _BAR_WIDTH = 40
    _REFRESH_SECS = 0.5  # max two redraws per second

    def __init__(self, total_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._total = total_timesteps
        self._last_print = 0.0
        self._start: float | None = None

    def _on_step(self) -> bool:
        if self._start is None:
            self._start = time.monotonic()
        now = time.monotonic()
        if now - self._last_print >= self._REFRESH_SECS:
            self._render(now)
            self._last_print = now
        return True

    def _on_training_end(self) -> None:
        self._render(time.monotonic(), final=True)
        print()  # newline so subsequent output starts on a fresh line

    @staticmethod
    def _fmt_seconds(secs: float) -> str:
        secs = int(secs)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    def _render(self, now: float, final: bool = False) -> None:
        pct = min(self.num_timesteps / self._total, 1.0)
        filled = int(self._BAR_WIDTH * pct)
        bar = "=" * filled + "-" * (self._BAR_WIDTH - filled)

        eta_str = ""
        if final:
            elapsed = now - (self._start or now)
            eta_str = f"  elapsed {self._fmt_seconds(elapsed)}"
        elif self._start is not None and pct > 0:
            elapsed = now - self._start
            remaining = elapsed / pct * (1.0 - pct)
            eta_str = f"  eta {self._fmt_seconds(remaining)}"

        print(
            f"\r  [{bar}] {pct * 100:5.1f}%  "
            f"{self.num_timesteps:,}/{self._total:,} steps{eta_str}",
            end="",
            flush=True,
        )


class ModelPingThread:
    """Background thread that pings last_ping every `interval` seconds while training.

    Keeps the model claim alive so release-claims doesn't reclaim it mid-run.
    Stopped via stop(), which blocks until the thread exits.
    """

    def __init__(self, model_name: str, interval: int = 60) -> None:
        self._model_name = model_name
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"ping-{model_name}"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self._interval + 5)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                ping_model_claim(self._model_name)
            except Exception as exc:
                print(f"  [ping] DB write failed: {exc}")


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
    # Only cap PyTorch's thread pool when OMP_NUM_THREADS is explicitly set
    # (e.g. by run_sweep.sh for multi-worker parallel mode). By default, let
    # PyTorch use all available cores — correct for single-worker/distributed mode.
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp:
        torch.set_num_threads(int(omp))
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

    ping_thread = ModelPingThread(config.name)
    ping_thread.start()

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
        progress_cb = TrainingProgressCallback(total_timesteps=total_timesteps)

        model = algo_cls(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            verbose=1,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, pnl_cb, progress_cb],
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

    finally:
        ping_thread.stop()
