from __future__ import annotations

import argparse
import sys

from trainer.config import ModelConfig
from trainer.db import (
    get_training_run,
    list_model_configs,
    load_model_config,
    save_model_config,
)
from trainer.models.btc_config import make_btc_config
from trainer.models.sol_config import make_sol_config

BUILTIN_CONFIGS = {
    "btc": make_btc_config,
    "sol": make_sol_config,
}


def cmd_create_model(args: argparse.Namespace) -> None:
    name = args.config
    factory = BUILTIN_CONFIGS.get(name)
    if factory is None:
        print(f"Unknown config: {name}. Available: {list(BUILTIN_CONFIGS)}")
        sys.exit(1)
    config = factory()
    config_id = save_model_config(config)
    print(f"Model '{config.name}' registered (id={config_id}).")


def cmd_start(args: argparse.Namespace) -> None:
    from trainer.training.trainer import train_model

    config = load_model_config(args.model)
    if config is None:
        print(f"Model '{args.model}' not found. Run create-model first.")
        sys.exit(1)

    train_model(
        config,
        algo_override=args.algo,
        timesteps_override=args.timesteps,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    from trainer.training.evaluator import evaluate_model

    config = load_model_config(args.model)
    if config is None:
        print(f"Model '{args.model}' not found.")
        sys.exit(1)

    run = get_training_run(args.run)
    if run is None:
        print(f"Training run #{args.run} not found.")
        sys.exit(1)

    model_path = run.get("model_path")
    if not model_path:
        print(f"Run #{args.run} has no saved model.")
        sys.exit(1)

    evaluate_model(config, model_path, algorithm=run.get("algorithm"))


def cmd_list(_args: argparse.Namespace) -> None:
    models = list_model_configs()
    if not models:
        print("No models registered.")
        return
    print(f"{'Name':<20} {'Runs':>6} {'Best PnL':>12} {'Created'}")
    print("-" * 60)
    for m in models:
        pnl = f"${m['best_pnl']:.2f}" if m["best_pnl"] is not None else "—"
        created = m["created_at"].strftime("%Y-%m-%d %H:%M") if m["created_at"] else "—"
        print(f"{m['name']:<20} {m['run_count']:>6} {pnl:>12} {created}")


def cmd_status(args: argparse.Namespace) -> None:
    run = get_training_run(args.run)
    if run is None:
        print(f"Training run #{args.run} not found.")
        sys.exit(1)
    print(f"Run #{run['id']}:")
    print(f"  Model:        {run.get('model_name', '?')}")
    print(f"  Type:         {run['run_type']}")
    print(f"  Algorithm:    {run['algorithm']}")
    print(f"  Status:       {run['status']}")
    print(f"  Started:      {run['started_at']}")
    if run.get("completed_at"):
        print(f"  Completed:    {run['completed_at']}")
    if run.get("final_equity") is not None:
        print(f"  Final equity: ${float(run['final_equity']):.2f}")
        print(f"  Total PnL:    ${float(run['total_pnl']):.2f}")
        print(f"  Win rate:     {float(run['win_rate'])*100:.1f}%")
        print(f"  Max drawdown: {float(run['max_drawdown'])*100:.1f}%")
        print(f"  Sharpe ratio: {float(run['sharpe_ratio']):.2f}")
        print(f"  Total trades: {run['total_trades']}")
    if run.get("model_path"):
        print(f"  Model:        {run['model_path']}")
    if run.get("error"):
        print(f"  Error:        {run['error']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Trading model trainer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cm = sub.add_parser("create-model", help="Register a model config")
    cm.add_argument(
        "--config", required=True,
        help=f"Built-in config name: {list(BUILTIN_CONFIGS)}",
    )

    st = sub.add_parser("start", help="Start a training run")
    st.add_argument("--model", required=True, help="Model name (registered)")
    st.add_argument("--algo", default=None, help="Algorithm override (PPO, SAC, A2C)")
    st.add_argument("--timesteps", type=int, default=None, help="Total timesteps override")

    ev = sub.add_parser("evaluate", help="Evaluate a trained model on holdout data")
    ev.add_argument("--model", required=True, help="Model name")
    ev.add_argument("--run", type=int, required=True, help="Training run ID to evaluate")

    sub.add_parser("list", help="List all registered models")

    sr = sub.add_parser("status", help="Show details for a training run")
    sr.add_argument("--run", type=int, required=True, help="Run ID")

    return parser


_COMMANDS = {
    "create-model": cmd_create_model,
    "start": cmd_start,
    "evaluate": cmd_evaluate,
    "list": cmd_list,
    "status": cmd_status,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
