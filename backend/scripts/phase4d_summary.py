#!/usr/bin/env python3
"""
Phase 4D — Decision matrix and per-architecture paired diff vs Phase 4A.

For each of the three Phase 4D architectures (lb=100/250/500 under the new
env risk caps):
  - Apply the same 4A pass criterion (median > 0 AND ≥3/5 positive).
  - Print a side-by-side per-seed comparison: 4A holdout PnL vs 4D holdout
    PnL plus the delta. If the env caps help, deltas should be systematically
    positive across seeds and architectures.

Reuses `evaluate_arch` and `fetch_holdout_pnls` from `phase4a_summary` so the
pass/fail rule stays in one place. Loaded via importlib.util because
`backend/scripts/` is not an importable package (same pattern as 4C summary).

Run from repo: cd backend && uv run python scripts/phase4d_summary.py
"""
from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

from ingester.db import connect

_SPEC = importlib.util.spec_from_file_location(
    "phase4a_summary", Path(__file__).resolve().parent / "phase4a_summary.py"
)
_P4A = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_P4A)
evaluate_arch = _P4A.evaluate_arch
fetch_holdout_pnls = _P4A.fetch_holdout_pnls

ARCHITECTURES = [100, 250, 500]


def fetch_holdout_pnls_by_seed(conn, name_pattern: str) -> dict[int, float]:
    """Return {seed: latest_holdout_pnl} for every model_config matching the
    pattern that has a completed eval. DISTINCT ON dedupes re-evals."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (mc.name)
            (mc.config_json->>'seed')::int AS seed,
            tr_eval.total_pnl
        FROM model_configs mc
        JOIN training_runs tr_eval ON tr_eval.model_config_id = mc.id
            AND tr_eval.run_type = 'evaluate' AND tr_eval.status = 'completed'
        WHERE mc.name LIKE %s
        ORDER BY mc.name, tr_eval.id DESC
        """,
        (name_pattern,),
    ).fetchall()
    return {int(seed): float(pnl) for seed, pnl in rows if seed is not None}


def paired_diff_per_arch(
    p4a_by_arch: dict[str, dict[int, float]],
    p4d_by_arch: dict[str, dict[int, float]],
) -> list[dict]:
    """Align per-architecture seed dicts and emit one row per (arch, seed).

    Each row: {arch, seed, p4a_pnl, p4d_pnl, delta}. delta is p4d - p4a,
    or None if either side is missing for that (arch, seed).
    """
    archs = sorted(set(p4a_by_arch) | set(p4d_by_arch))
    rows: list[dict] = []
    for arch in archs:
        a_seeds = p4a_by_arch.get(arch, {})
        d_seeds = p4d_by_arch.get(arch, {})
        seeds = sorted(set(a_seeds) | set(d_seeds))
        for s in seeds:
            a = a_seeds.get(s)
            d = d_seeds.get(s)
            delta = d - a if (a is not None and d is not None) else None
            rows.append(
                {"arch": arch, "seed": s, "p4a_pnl": a, "p4d_pnl": d, "delta": delta}
            )
    return rows


def _fmt_pnl(v: float | None) -> str:
    return f"${v:+,.0f}" if v is not None else "—"


def main() -> None:
    conn = connect()
    try:
        decisions: list[dict] = []
        p4a_by_arch: dict[str, dict[int, float]] = {}
        p4d_by_arch: dict[str, dict[int, float]] = {}

        for lb in ARCHITECTURES:
            arch = f"lb{lb}"
            p4d_pattern = f"%_lb{lb}_3em4_p4d_s%"
            p4a_pattern = f"%_lb{lb}_3em4_p4a_s%"

            p4d_pnls = fetch_holdout_pnls(conn, p4d_pattern)
            p4a_pnls = fetch_holdout_pnls(conn, p4a_pattern)
            decisions.append(
                evaluate_arch(arch=arch, p4a_pnls=p4d_pnls, p2_pnls=p4a_pnls)
            )
            p4a_by_arch[arch] = fetch_holdout_pnls_by_seed(conn, p4a_pattern)
            p4d_by_arch[arch] = fetch_holdout_pnls_by_seed(conn, p4d_pattern)
    finally:
        conn.close()

    # Headline decision per architecture.
    print(
        f"{'arch':<8} {'seeds':>6} {'pos':>4} {'median PnL':>14} "
        f"{'4A median':>14}  decision"
    )
    print("-" * 72)
    for r in decisions:
        if r["pass"] is None:
            verdict = f"INCOMPLETE ({r['incomplete_reason']})"
        else:
            verdict = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['arch']:<8} {r['p4a_count']:>6} {r['count_positive']:>4} "
            f"{_fmt_pnl(r['median_pnl']):>14} {_fmt_pnl(r['p2_median_pnl']):>14}  {verdict}"
        )

    # Paired per-seed diff per architecture.
    print()
    print("Paired per-seed delta (4D = new env caps; 4A = original env):")
    print(
        f"{'arch':<8} {'seed':>6}  {'4A holdout':>14}  {'4D holdout':>14}  {'delta':>14}"
    )
    print("-" * 70)
    rows = paired_diff_per_arch(p4a_by_arch, p4d_by_arch)
    for r in rows:
        print(
            f"{r['arch']:<8} {r['seed']:>6}  "
            f"{_fmt_pnl(r['p4a_pnl']):>14}  "
            f"{_fmt_pnl(r['p4d_pnl']):>14}  "
            f"{_fmt_pnl(r['delta']):>14}"
        )

    # Aggregate paired stats.
    valid = [r["delta"] for r in rows if r["delta"] is not None]
    if valid:
        positive = sum(1 for d in valid if d > 0)
        median_delta = statistics.median(valid)
        print()
        print(
            f"4D beat 4A on {positive} of {len(valid)} paired (arch, seed) cells; "
            f"median delta = {_fmt_pnl(median_delta)}."
        )


if __name__ == "__main__":
    main()
