from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingester.db import connect
from trainer.db import (
    claim_pending_model,
    get_training_run,
    list_model_configs,
    list_stale_claims,
    load_model_config,
    release_stale_claims,
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


def _rsync_model(run_id: int, model_dir: Path, rsync_target: str) -> bool:
    """Rsync model dir to target, update DB path, delete local copy on success.

    rsync_target format: user@host:/remote/base/dir
      e.g. root@BASE_IP:/mnt/models  (when train_enabled=false, models volume on base)
           root@BASE_IP:/opt/tradan/trained_models  (symlinks to same place)
    The run dir is uploaded as: <rsync_target>/<model_name>/<run_id>/

    SSH key is taken from MODELS_SSH_KEY env var when set; otherwise the SSH agent
    / default key is used. Password authentication is always disabled.
    Returns True if rsync succeeded and local files were cleaned up.
    """
    import os
    import shlex
    import shutil
    import subprocess

    from trainer.db import update_model_path

    ssh_key = os.environ.get("MODELS_SSH_KEY")
    ssh_cmd = "ssh -o StrictHostKeyChecking=accept-new -o PasswordAuthentication=no"
    if ssh_key:
        ssh_cmd += f" -i {shlex.quote(os.path.expanduser(ssh_key))}"

    remote_run_dir = f"{rsync_target}/{model_dir.parent.name}/{model_dir.name}/"
    cmd = [
        "rsync", "-av", "--mkpath",
        "-e", ssh_cmd,
        str(model_dir) + "/",
        remote_run_dir,
    ]
    print(f"  Uploading models: {' '.join(shlex.quote(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: rsync failed (exit {result.returncode}). Models kept locally at {model_dir}")
        return False

    remote_model_path = f"{rsync_target}/{model_dir.parent.name}/{model_dir.name}/model.zip"
    update_model_path(run_id, remote_model_path)
    print(f"  DB model_path updated to: {remote_model_path}")

    shutil.rmtree(model_dir, ignore_errors=True)
    print(f"  Local model dir removed: {model_dir}")
    return True


def cmd_worker(args: argparse.Namespace) -> None:
    import os
    import threading
    import time

    from trainer.training.trainer import MODELS_DIR, train_model

    poll_seconds = args.poll_seconds
    rsync_target = os.environ.get("MODELS_RSYNC_TARGET")

    # Only cap OMP/MKL thread counts when explicitly throttling below 100%.
    # At 100% (default for single-worker mode) we leave the env vars unset so
    # PyTorch manages its own thread pools without interference.
    cpus = os.cpu_count() or 1
    if args.cpu_usage < 100:
        threads = max(1, cpus * args.cpu_usage // 100)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

    # Track background upload threads so we can wait for them before exiting.
    upload_threads: list[threading.Thread] = []

    while True:
        # Reap finished upload threads to avoid unbounded growth.
        upload_threads = [t for t in upload_threads if t.is_alive()]

        result = claim_pending_model()
        if result is None:
            if poll_seconds == 0:
                break
            time.sleep(poll_seconds)
            continue

        name, config = result
        print(f"claimed: {name}")
        run_id = None
        try:
            run_id = train_model(config)
        except Exception as e:
            print(f"FAILED: {name}: {e}")
            # fail_training_run already called inside train_model; continue to next

        # Upload happens once per completed training run (not per iteration).
        # Run it on a background thread so the worker can claim the next model
        # immediately while the rsync transfer proceeds in parallel.
        if run_id is not None and rsync_target:
            model_dir = MODELS_DIR / name / str(run_id)
            if model_dir.exists():
                t = threading.Thread(
                    target=_rsync_model,
                    args=(run_id, model_dir, rsync_target),
                    daemon=True,
                    name=f"upload-{run_id}",
                )
                t.start()
                upload_threads.append(t)
            else:
                print(f"WARNING: model dir not found: {model_dir}")

    # Drain any in-flight uploads before the process exits.
    for t in [t for t in upload_threads if t.is_alive()]:
        t.join()


def cmd_release_claims(args: argparse.Namespace) -> None:
    older_than = args.older_than_seconds
    stale = list_stale_claims(older_than)

    if not stale:
        print(f"No stale claims found (older than {older_than}s).")
        return

    print(f"Found {len(stale)} model(s) with claims older than {older_than} seconds:\n")
    print(f"  {'Name':<30} {'Last Ping':<25} {'Silent For'}")
    print("  " + "-" * 72)
    for m in stale:
        age = m["silent_seconds"]
        if age >= 3600:
            age_str = f"{age // 3600}h {(age % 3600) // 60}m"
        elif age >= 60:
            age_str = f"{age // 60}m {age % 60}s"
        else:
            age_str = f"{age}s"
        ping_ts = m["last_ping"] or m["claimed_at"]
        ping = ping_ts.strftime("%Y-%m-%d %H:%M:%S") if ping_ts else "—"
        print(f"  {m['name']:<30} {ping:<25} {age_str}")

    print()
    try:
        answer = input(f"Release these {len(stale)} claim(s)? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return

    if answer != "y":
        print("Aborted.")
        return

    released = release_stale_claims(older_than)
    print(f"Released {released} claim(s).")


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

    rc = sub.add_parser(
        "release-claims",
        help="Release model claims not updated within a given timeout",
        description=(
            "List model configs whose claim is older than --older-than-seconds "
            "with no active running training run, then optionally null out the claim "
            "so another worker can pick them up."
        ),
    )
    rc.add_argument(
        "--older-than-seconds",
        type=int,
        default=3600,
        dest="older_than_seconds",
        metavar="SECONDS",
        help="Release claims not updated for this many seconds (default: 3600). Can be as low as 1.",
    )

    wk = sub.add_parser(
        "worker",
        help="Claim and train pending models one at a time until none remain",
        description=(
            "Claim and train models until none remain. "
            "Set MODELS_RSYNC_TARGET=user@host:/path to upload each run's model dir "
            "to a remote host after training and delete the local copy."
        ),
    )
    wk.add_argument(
        "--poll-seconds",
        type=int,
        default=0,
        dest="poll_seconds",
        help="Seconds to wait between polls when no pending models exist. 0 = exit immediately (default).",
    )
    wk.add_argument(
        "--cpu-usage",
        type=int,
        default=100,
        dest="cpu_usage",
        metavar="PCT",
        help="Target CPU usage percentage (1-100). At 100 (default) OMP/MKL are left unset so PyTorch manages its own thread pools. Below 100, sets OMP/MKL thread count to floor(nproc * PCT / 100).",
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
    "release-claims": cmd_release_claims,
    "worker": cmd_worker,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
