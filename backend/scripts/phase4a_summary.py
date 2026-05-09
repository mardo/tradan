#!/usr/bin/env python3
"""
Phase 4A — Per-architecture decision matrix.

For each of the three Phase 4A architectures (lb=100/250/500), pulls the
holdout-eval PnL for every `_p4a_` seed and prints:

  - count of seeds that completed eval
  - count of seeds with positive PnL
  - median PnL across the 5 _p4a_ seeds (the decision metric)
  - PASS / FAIL  (PASS = median > 0 AND ≥3 of 5 seeds positive)
  - For context: the original 3 _p2_ seed PnLs and their median (informational only;
    they don't gate the decision because P2 runs were unseeded).

Run from repo: cd backend && uv run python scripts/phase4a_summary.py
"""
from __future__ import annotations

import statistics
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

from ingester.db import connect

ARCHITECTURES = [100, 250, 500]
EXPECTED_P4A_SEEDS = 5


def fetch_holdout_pnls(conn, name_pattern: str) -> list[float]:
    # DISTINCT ON dedupes when a model has been re-evaluated (multiple completed
    # eval runs); keep the latest by training_runs.id so the decision matrix isn't
    # skewed by stale evals from before a bug fix or re-run.
    rows = conn.execute(
        """
        SELECT DISTINCT ON (mc.name) tr_eval.total_pnl
        FROM model_configs mc
        JOIN training_runs tr_eval ON tr_eval.model_config_id = mc.id
            AND tr_eval.run_type = 'evaluate' AND tr_eval.status = 'completed'
        WHERE mc.name LIKE %s
        ORDER BY mc.name, tr_eval.id DESC
        """,
        (name_pattern,),
    ).fetchall()
    return [float(r[0]) for r in rows]


def evaluate_arch(
    *, arch: str, p4a_pnls: list[float], p2_pnls: list[float]
) -> dict:
    """Compute pass/fail decision for an architecture.

    Decision uses _p4a_ seeds only (5 expected). _p2_ seeds are reported for
    context — they were unseeded and pre-date the seed-plumbing fix.

    Returns dict with keys:
      arch, p4a_count, count_positive, median_pnl, pass (bool|None),
      p2_count, p2_median_pnl, incomplete_reason (str|None)
    """
    p4a_count = len(p4a_pnls)
    if p4a_count < EXPECTED_P4A_SEEDS:
        return {
            "arch": arch,
            "p4a_count": p4a_count,
            "count_positive": sum(1 for p in p4a_pnls if p > 0),
            "median_pnl": statistics.median(p4a_pnls) if p4a_pnls else None,
            "pass": None,
            "incomplete_reason": (
                f"only {p4a_count} of {EXPECTED_P4A_SEEDS} p4a seeds have eval results"
            ),
            "p2_count": len(p2_pnls),
            "p2_median_pnl": statistics.median(p2_pnls) if p2_pnls else None,
        }

    count_positive = sum(1 for p in p4a_pnls if p > 0)
    median_pnl = statistics.median(p4a_pnls)
    passed = median_pnl > 0 and count_positive >= 3

    return {
        "arch": arch,
        "p4a_count": p4a_count,
        "count_positive": count_positive,
        "median_pnl": median_pnl,
        "pass": passed,
        "incomplete_reason": None,
        "p2_count": len(p2_pnls),
        "p2_median_pnl": statistics.median(p2_pnls) if p2_pnls else None,
    }


def main() -> None:
    conn = connect()
    try:
        results = []
        for lb in ARCHITECTURES:
            p4a = fetch_holdout_pnls(conn, f"%_lb{lb}_3em4_p4a_s%")
            p2 = fetch_holdout_pnls(conn, f"%_lb{lb}_3em4_p2_s%")
            results.append(evaluate_arch(arch=f"lb{lb}", p4a_pnls=p4a, p2_pnls=p2))
    finally:
        conn.close()

    print(f"{'arch':<8} {'seeds':>6} {'pos':>4} {'median PnL':>14} {'P2 median':>14}  decision")
    print("-" * 70)
    for r in results:
        median_str = f"${r['median_pnl']:+,.0f}" if r["median_pnl"] is not None else "—"
        p2_median_str = f"${r['p2_median_pnl']:+,.0f}" if r["p2_median_pnl"] is not None else "—"
        if r["pass"] is None:
            decision = f"INCOMPLETE ({r['incomplete_reason']})"
        else:
            decision = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['arch']:<8} {r['p4a_count']:>6} {r['count_positive']:>4} "
            f"{median_str:>14} {p2_median_str:>14}  {decision}"
        )


if __name__ == "__main__":
    main()
