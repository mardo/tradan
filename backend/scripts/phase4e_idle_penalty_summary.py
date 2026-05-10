#!/usr/bin/env python3
"""
Phase 4E — Per-cell decision matrix and paired diff vs Phase 4D.

For each (architecture × penalty value) cell:
  - Apply the same 4A pass criterion (median > 0 AND ≥3/5 positive).
  - Print a side-by-side per-seed table comparing 4D holdout PnL vs 4E
    cell holdout PnL plus the delta. If the idle penalty helps, deltas
    should be systematically positive across seeds within a cell.

Architectures: lb100, lb250, lb500. Penalty slugs: idle05 (0.05 USD/step),
idle5 (0.5 USD/step). 5 paired seeds per cell (matching 4A/4D's
1001..5005). 6 cells × 5 seeds = 30 paired comparisons.

Reuses `evaluate_arch` and `fetch_holdout_pnls` from `phase4a_summary` so
the pass/fail rule stays in one place. Loaded via importlib.util because
`backend/scripts/` is not an importable package — same pattern as the
4C/4D summary scripts.

Run from repo: cd backend && uv run python scripts/phase4e_idle_penalty_summary.py
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
# (slug, label) — slug is what's in the model_config name; label is the
# human-readable penalty value used in printed headers.
PENALTY_SLUGS: list[tuple[str, str]] = [("idle05", "0.05"), ("idle5", "0.5")]


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


def paired_diff_per_cell(
    p4d_by_arch: dict[str, dict[int, float]],
    p4e_by_cell: dict[tuple[str, str], dict[int, float]],
) -> list[dict]:
    """Align (arch, slug, seed) rows between the 4D baseline and a 4E cell.

    The 4D side is keyed by arch alone (slug doesn't apply — 4D had no
    penalty); the 4E side is keyed by (arch, slug). For each (arch, slug)
    cell present in p4e_by_cell, one row is emitted per seed in the union
    of (p4d_by_arch[arch] ∪ p4e_by_cell[arch, slug]).

    Each row: {arch, slug, seed, p4d_pnl, p4e_pnl, delta}. delta = p4e − p4d,
    or None if either side is missing for that cell × seed.
    """
    rows: list[dict] = []
    cells = sorted(p4e_by_cell.keys())
    for (arch, slug) in cells:
        d_seeds = p4d_by_arch.get(arch, {})
        e_seeds = p4e_by_cell[(arch, slug)]
        seeds = sorted(set(d_seeds) | set(e_seeds))
        for s in seeds:
            d = d_seeds.get(s)
            e = e_seeds.get(s)
            delta = e - d if (d is not None and e is not None) else None
            rows.append(
                {
                    "arch": arch,
                    "slug": slug,
                    "seed": s,
                    "p4d_pnl": d,
                    "p4e_pnl": e,
                    "delta": delta,
                }
            )
    return rows


def _fmt_pnl(v: float | None) -> str:
    return f"${v:+,.0f}" if v is not None else "—"


def main() -> None:
    conn = connect()
    try:
        decisions: list[dict] = []
        p4d_by_arch: dict[str, dict[int, float]] = {}
        p4e_by_cell: dict[tuple[str, str], dict[int, float]] = {}

        for lb in ARCHITECTURES:
            arch = f"lb{lb}"
            p4d_pattern = f"%_lb{lb}_3em4_p4d_s%"
            p4d_pnls = fetch_holdout_pnls(conn, p4d_pattern)
            p4d_by_arch[arch] = fetch_holdout_pnls_by_seed(conn, p4d_pattern)

            for slug, _label in PENALTY_SLUGS:
                pattern = f"%_lb{lb}_3em4_{slug}_p4e_s%"
                cell_pnls = fetch_holdout_pnls(conn, pattern)
                cell_by_seed = fetch_holdout_pnls_by_seed(conn, pattern)
                # evaluate_arch's "p2_pnls" slot is used as the informational
                # baseline column in the printed table — here that's 4D.
                decisions.append(
                    {
                        "arch": arch,
                        "slug": slug,
                        **evaluate_arch(
                            arch=f"{arch}_{slug}",
                            p4a_pnls=cell_pnls,
                            p2_pnls=p4d_pnls,
                        ),
                    }
                )
                p4e_by_cell[(arch, slug)] = cell_by_seed
    finally:
        conn.close()

    # Headline decision per (arch, slug) cell.
    print(
        f"{'arch':<8} {'slug':<8} {'seeds':>6} {'pos':>4} {'median PnL':>14} "
        f"{'4D median':>14}  decision"
    )
    print("-" * 80)
    for r in decisions:
        if r["pass"] is None:
            verdict = f"INCOMPLETE ({r['incomplete_reason']})"
        else:
            verdict = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['arch']:<8} {r['slug']:<8} "
            f"{r['p4a_count']:>6} {r['count_positive']:>4} "
            f"{_fmt_pnl(r['median_pnl']):>14} {_fmt_pnl(r['p2_median_pnl']):>14}  {verdict}"
        )

    # Paired per-(arch, slug, seed) diff.
    print()
    print("Paired per-seed delta (4E = idle penalty; 4D = same env, no penalty):")
    print(
        f"{'arch':<8} {'slug':<8} {'seed':>6}  "
        f"{'4D holdout':>14}  {'4E holdout':>14}  {'delta':>14}"
    )
    print("-" * 78)
    rows = paired_diff_per_cell(p4d_by_arch, p4e_by_cell)
    for r in rows:
        print(
            f"{r['arch']:<8} {r['slug']:<8} {r['seed']:>6}  "
            f"{_fmt_pnl(r['p4d_pnl']):>14}  "
            f"{_fmt_pnl(r['p4e_pnl']):>14}  "
            f"{_fmt_pnl(r['delta']):>14}"
        )

    # Aggregate paired stats per (arch, slug) cell.
    print()
    print("Aggregate paired delta per cell:")
    print(f"{'arch':<8} {'slug':<8} {'paired':>8} {'pos delta':>10} {'median Δ':>14}")
    print("-" * 56)
    cells: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r["delta"] is not None:
            cells.setdefault((r["arch"], r["slug"]), []).append(r["delta"])
    for (arch, slug), deltas in sorted(cells.items()):
        positive = sum(1 for d in deltas if d > 0)
        med = statistics.median(deltas)
        print(
            f"{arch:<8} {slug:<8} {len(deltas):>8} {positive:>10} {_fmt_pnl(med):>14}"
        )


if __name__ == "__main__":
    main()
