#!/usr/bin/env python3
"""
Phase 4C — Per-architecture decision matrix and paired diff vs Phase 4A.

For lb500 with ent_coef=0.01 (5 seeds, paired with 4A's seeds 1001..5005):
  - Apply the same 4A pass criterion (median > 0 AND ≥3/5 positive).
  - Print a side-by-side per-seed table comparing 4A holdout PnL vs 4C
    holdout PnL plus the delta. If entropy reg helps, deltas should be
    systematically positive across seeds.

Reuses `evaluate_arch` from `phase4a_summary` so the pass/fail rule stays
in one place. The two scripts live as siblings under `backend/scripts/`,
which is not an importable package — `evaluate_arch` is loaded via
`importlib.util` (the same pattern the unit tests already use).

Run from repo: cd backend && uv run python scripts/phase4c_entropy_summary.py
"""
from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

from ingester.db import connect

# Load evaluate_arch from the sibling 4A summary script. backend/scripts/ is
# not a package, so importlib.util is the established pattern for loading
# these scripts (see tests/trainer/test_*.py for the same approach).
_SPEC = importlib.util.spec_from_file_location(
    "phase4a_summary", Path(__file__).resolve().parent / "phase4a_summary.py"
)
_P4A = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_P4A)
evaluate_arch = _P4A.evaluate_arch
fetch_holdout_pnls = _P4A.fetch_holdout_pnls

EXPECTED_SEEDS = [1001, 2002, 3003, 4004, 5005]


def fetch_holdout_pnls_by_seed(conn, name_pattern: str) -> dict[int, float]:
    """Return {seed: latest_holdout_pnl} for every model_config matching the
    pattern that has a completed eval. Uses DISTINCT ON to dedupe re-evals,
    same as fetch_holdout_pnls in phase4a_summary."""
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


def paired_diff(
    p4a_by_seed: dict[int, float], p4c_by_seed: dict[int, float]
) -> list[dict]:
    """Align the two PnL series by seed and compute deltas.

    Returns one row per seed (union of seeds across the two series). For
    each row: {seed, p4a_pnl, p4c_pnl, delta}. delta = p4c - p4a, or None if
    either side is missing.
    """
    seeds = sorted(set(p4a_by_seed) | set(p4c_by_seed))
    rows: list[dict] = []
    for s in seeds:
        a = p4a_by_seed.get(s)
        c = p4c_by_seed.get(s)
        delta = c - a if (a is not None and c is not None) else None
        rows.append({"seed": s, "p4a_pnl": a, "p4c_pnl": c, "delta": delta})
    return rows


def _fmt_pnl(v: float | None) -> str:
    return f"${v:+,.0f}" if v is not None else "—"


def main() -> None:
    conn = connect()
    try:
        p4c_pnls = fetch_holdout_pnls(conn, "%_lb500_3em4_ent01_p4c_s%")
        p4a_pnls = fetch_holdout_pnls(conn, "%_lb500_3em4_p4a_s%")
        decision = evaluate_arch(arch="lb500_ent01", p4a_pnls=p4c_pnls, p2_pnls=p4a_pnls)
        # Note: we deliberately pass 4A as the "p2_pnls" argument because
        # evaluate_arch's "p2" slot is used for the informational baseline,
        # which here is 4A (the version of lb500 without entropy reg).

        p4c_by_seed = fetch_holdout_pnls_by_seed(conn, "%_lb500_3em4_ent01_p4c_s%")
        p4a_by_seed = fetch_holdout_pnls_by_seed(conn, "%_lb500_3em4_p4a_s%")
    finally:
        conn.close()

    # Headline decision
    print(
        f"{'arch':<14} {'seeds':>6} {'pos':>4} {'median PnL':>14} "
        f"{'4A median':>14}  decision"
    )
    print("-" * 72)
    if decision["pass"] is None:
        verdict = f"INCOMPLETE ({decision['incomplete_reason']})"
    else:
        verdict = "PASS" if decision["pass"] else "FAIL"
    print(
        f"{decision['arch']:<14} {decision['p4a_count']:>6} "
        f"{decision['count_positive']:>4} "
        f"{_fmt_pnl(decision['median_pnl']):>14} "
        f"{_fmt_pnl(decision['p2_median_pnl']):>14}  {verdict}"
    )

    # Paired per-seed diff
    print()
    print("Paired per-seed delta (entropy reg vs 4A baseline):")
    print(f"{'seed':>6}  {'4A holdout PnL':>16}  {'4C holdout PnL':>16}  {'delta':>14}")
    print("-" * 60)
    rows = paired_diff(p4a_by_seed, p4c_by_seed)
    for r in rows:
        print(
            f"{r['seed']:>6}  {_fmt_pnl(r['p4a_pnl']):>16}  "
            f"{_fmt_pnl(r['p4c_pnl']):>16}  {_fmt_pnl(r['delta']):>14}"
        )

    # Summary stat: how often did entropy reg beat 4A?
    valid = [r["delta"] for r in rows if r["delta"] is not None]
    if valid:
        positive = sum(1 for d in valid if d > 0)
        median_delta = statistics.median(valid)
        print()
        print(
            f"Entropy reg beat 4A on {positive} of {len(valid)} paired seeds; "
            f"median delta = {_fmt_pnl(median_delta)}."
        )


if __name__ == "__main__":
    main()
