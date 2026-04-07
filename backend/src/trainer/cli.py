from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingester.db import connect
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


def cmd_list(args: argparse.Namespace) -> None:
    models = list_model_configs()

    names_only = getattr(args, "names_only", False)
    status_filter = getattr(args, "status", None)

    # "pending"   = models with zero completed runs (not yet trained)
    # "completed" = models with at least one completed run
    if status_filter == "pending":
        models = [m for m in models if m["run_count"] == 0]
    elif status_filter == "completed":
        models = [m for m in models if m["run_count"] > 0]

    if names_only:
        for m in models:
            print(m["name"])
        return

    if not models:
        print("No models registered.")
        return

    print(f"{'Name':<20} {'Runs':>6} {'Best PnL':>12} {'Created'}")
    print("-" * 60)
    for m in models:
        pnl = f"${m['best_pnl']:.2f}" if m["best_pnl"] is not None else "—"
        created = m["created_at"].strftime("%Y-%m-%d %H:%M") if m["created_at"] else "—"
        print(f"{m['name']:<20} {m['run_count']:>6} {pnl:>12} {created}")


def _repo_root() -> Path:
    # backend/src/trainer/cli.py -> tradan/
    return Path(__file__).resolve().parents[3]


def _load_sql_query(relative_under_infra: str) -> str:
    path = _repo_root() / "infra" / "scripts" / relative_under_infra
    if not path.is_file():
        raise FileNotFoundError(f"SQL file not found: {path}")
    lines: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines).strip().rstrip(";")


def _print_query_result(sql: str) -> None:
    conn = connect()
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        cols = [d.name for d in (cur.description or [])]
        if not cols:
            print("(no columns)")
            return
        if not rows:
            print("(no rows)")
            return
        widths = [len(c) for c in cols]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell) if cell is not None else ""))
        fmt = "  ".join(f"{{:{w}}}" for w in widths)
        print(fmt.format(*cols))
        print("  ".join("-" * w for w in widths))
        for row in rows:
            print(fmt.format(*(str(c) if c is not None else "" for c in row)))
    finally:
        conn.close()


def cmd_winners(_args: argparse.Namespace) -> None:
    """Ranked configs passing eval filters (infra/scripts/winners.sql)."""
    _print_query_result(_load_sql_query("winners.sql"))


def cmd_winners_no_eval(_args: argparse.Namespace) -> None:
    """Top completed train runs with no evaluate run yet (winners_no_eval.sql)."""
    _print_query_result(_load_sql_query("winners_no_eval.sql"))


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

    ls = sub.add_parser("list", help="List all registered models")
    ls.add_argument(
        "--names-only",
        action="store_true",
        dest="names_only",
        help="Print only model names, one per line (for use with GNU parallel)",
    )
    ls.add_argument(
        "--status",
        choices=["pending", "completed"],
        default=None,
        help="Filter: pending = no completed runs, completed = has completed runs",
    )

    sr = sub.add_parser("status", help="Show details for a training run")
    sr.add_argument("--run", type=int, required=True, help="Run ID")

    sub.add_parser(
        "winners",
        help="Print ranked winners (holdout Sharpe) — infra/scripts/winners.sql",
    )
    sub.add_parser(
        "winners-no-eval",
        help="Top train runs without an eval run — infra/scripts/winners_no_eval.sql",
    )

    return parser


_COMMANDS = {
    "create-model": cmd_create_model,
    "start": cmd_start,
    "evaluate": cmd_evaluate,
    "list": cmd_list,
    "status": cmd_status,
    "winners": cmd_winners,
    "winners-no-eval": cmd_winners_no_eval,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
