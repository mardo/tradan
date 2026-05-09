"""Live runner CLI: live-test."""
from __future__ import annotations

import argparse
import sys

from ingester.db import connect


def _cmd_run(args: argparse.Namespace) -> int:
    from live.runner import run_live
    return run_live(config_path=args.config, dry_run=args.dry_run)


def _cmd_status(args: argparse.Namespace) -> int:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT lr.id, mc.name, lr.exchange, lr.mode, lr.started_at,
                   lr.starting_equity, lr.kill_requested,
                   COALESCE(
                       (SELECT equity FROM live_pnl_snapshots
                        WHERE live_run_id = lr.id
                        ORDER BY taken_at DESC LIMIT 1),
                       lr.starting_equity
                   ) AS current_equity
            FROM live_runs lr
            JOIN model_configs mc ON mc.id = lr.model_config_id
            WHERE lr.status = 'running'
            ORDER BY lr.started_at
            """,
        ).fetchall()
    if not rows:
        print("No active runs.")
        return 0
    print(f"{'id':>4}  {'model':<40}  {'exch':<8}  {'mode':<6}  "
          f"{'start_equity':>14}  {'cur_equity':>14}  {'kill?':>5}")
    for r in rows:
        run_id, name, exc, mode, _started, start_eq, kill, cur_eq = r
        print(f"{run_id:>4}  {name:<40}  {exc:<8}  {mode:<6}  "
              f"{float(start_eq):>14.2f}  {float(cur_eq):>14.2f}  {str(kill):>5}")
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    from live.db import request_stop
    with connect() as conn:
        conn.autocommit = True
        request_stop(conn, args.run_id)
    print(f"Stop requested for run {args.run_id}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="live-test")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Start (or resume) a live run.")
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=_cmd_run)

    sub.add_parser("status", help="List active runs.").set_defaults(func=_cmd_status)

    stop = sub.add_parser("stop", help="Request graceful stop of a run.")
    stop.add_argument("run_id", type=int)
    stop.set_defaults(func=_cmd_stop)

    args = p.parse_args(argv)
    if args.cmd is None:
        p.print_help()
        return 1
    return args.func(args)


def replay_main() -> int:
    from live.replay_cli import replay_main as _replay
    return _replay()


if __name__ == "__main__":
    sys.exit(main())
