from __future__ import annotations

from pathlib import Path

from stable_baselines3 import A2C, PPO, SAC

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
from trainer.training.trainer import ALGO_MAP, MODELS_DIR, compute_metrics


def evaluate_model(
    config: ModelConfig,
    model_path: str,
    algorithm: str | None = None,
) -> int:
    algorithm = algorithm or config.algorithm

    algo_cls = ALGO_MAP.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    config_id = get_model_config_id(config.name)
    if config_id is None:
        raise ValueError(f"Model '{config.name}' not found in DB.")

    run_id = create_training_run(config_id, "evaluate", algorithm)
    print(f"Evaluation run #{run_id}: model={config.name} path={model_path}")

    try:
        conn = connect()
        try:
            data_feed = load_data_feed(config, conn)
        finally:
            conn.close()

        split_idx = int(data_feed.total_steps * 0.8)
        holdout_start = split_idx
        holdout_timestamps = data_feed.timestamps[holdout_start:]
        holdout_features = data_feed.raw_features[holdout_start:]

        holdout_feed = DataFeed(
            timestamps=holdout_timestamps,
            features=holdout_features,
            lookback=config.lookback_window,
            price_columns=data_feed.price_columns,
        )

        env = TradingEnv(config=config, data_feed=holdout_feed)
        model = algo_cls.load(model_path, env=env)

        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

        snapshots = []
        for entry in env.pnl_history[::config.snapshot_interval]:
            snapshots.append({**entry, "training_run_id": run_id})
        if snapshots:
            conn = connect()
            try:
                save_pnl_snapshots(conn, snapshots)
            finally:
                conn.close()

        metrics = compute_metrics(env)
        eval_model_path = str(MODELS_DIR / config.name / str(run_id) / "eval_reference.txt")
        Path(eval_model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(eval_model_path).write_text(f"Evaluated from: {model_path}\n")

        complete_training_run(run_id, model_path=model_path, **metrics)

        print(f"Evaluation run #{run_id} completed.")
        print(f"  Final equity: ${metrics['final_equity']:.2f}")
        print(f"  Total PnL:    ${metrics['total_pnl']:.2f}")
        print(f"  Win rate:     {metrics['win_rate']*100:.1f}%")
        print(f"  Max drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        return run_id

    except Exception as e:
        fail_training_run(run_id, str(e))
        print(f"Evaluation run #{run_id} FAILED: {e}")
        raise
