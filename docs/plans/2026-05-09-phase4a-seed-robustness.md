# Phase 4A — Seed Robustness Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> Before starting Task 1, use `superpowers:using-git-worktrees` to create an isolated worktree from `main`. All edits below assume that worktree as cwd.

**Goal:** Characterize per-architecture seed variance for the three top P2 architectures (lb=100/250/500, lr=3e-4, 4h, A2C) by running 5 reproducibly-seeded training runs per architecture, then computing median holdout PnL and "≥3/5 seeds positive" so we can decide which architectures are promotable to paper trading and which are seed-luck artifacts.

**Architecture:** Two halves — a tiny code change to make seeds real, then an operational sweep that reuses the existing pipeline.
1. **Plumb seed through ModelConfig → SB3.** The current sweep scripts declare `SEEDS = [42, 123, 456]` but never pass them to the algorithm; all P1/P2/P3 variance came from non-deterministic torch/numpy init. Add `ModelConfig.seed: int | None`, persist it via `to_dict`/`from_dict`, and pass it to `algo_cls(...)` in `trainer.py`.
2. **Register and run 15 new configs.** A new `sweep_phase4a.py` registers `btc_4h_a2c_lb{100,250,500}_3em4_p4a_s{0..4}` with seeds `[1001, 2002, 3003, 4004, 5005]`. The existing `train worker` loop trains them; `infra/scripts/evaluate_winners.sh` evals them; a new `phase4a_summary.py` produces the per-architecture pass/fail decision matrix.

**Tech Stack:** Python 3.12, stable-baselines3 (PPO/A2C/SAC), psycopg 3, PostgreSQL, pytest, uv. Existing trainer code under `backend/src/trainer/`. Training runs on the remote `tradan-training` host; commands below assume invocation via the `mcp__tradan-training-ssh-mcp__exec` tool when applicable, or `cd backend && uv run …` locally for code-only steps.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `backend/src/trainer/config.py` | modify | Add `seed: int \| None = None` field; round-trip in `to_dict`/`from_dict`. |
| `backend/src/trainer/training/trainer.py` | modify | Pass `seed=config.seed` to `algo_cls(...)` in `train_model`. |
| `backend/tests/trainer/test_seed_plumbing.py` | create | Unit tests for ModelConfig round-trip + that trainer forwards seed to SB3 (mocked). |
| `backend/scripts/sweep_phase4a.py` | create | Register 15 `_p4a_` ModelConfigs (3 lookbacks × 5 explicit seeds). |
| `backend/scripts/phase4a_summary.py` | create | Aggregate holdout eval results across `_p2_` + `_p4a_` seeds per architecture; print pass/fail. |
| `backend/tests/trainer/test_phase4a_summary.py` | create | Unit-test the aggregation/pass-fail logic with synthetic rows. |
| `docs/plans/2026-05-09-phase4-training-plan.md` | modify | Update "Open decisions" + 4A section with the chosen seeds and decision-matrix output. |

The two new scripts mirror the existing `sweep_phase{1,2,3}.py` style. The summary script is new because 4A's decision criteria (median > 0, ≥3/5 positive) don't match the existing `winners.sql` filter chain.

---

## Decisions locked in (don't re-litigate)

- **Architectures to expand:** `lb=100`, `lb=250`, `lb=500` — all `lr=3e-4`, `4h`, A2C. (Top-3 by holdout sharpe in P2.)
- **Seed values:** `[1001, 2002, 3003, 4004, 5005]` — chosen to be visibly distinct from the P1/P2/P3 dummy seeds (`42, 123, 456`). Values themselves don't matter, only that they're fixed and recorded.
- **Phase tag:** `_p4a_`. New `LIKE '%_p4a_%'` filter; matches the existing pattern. Old `_p2_` runs stay as additional unseeded samples — they can be pooled in the aggregation as 3 extra observations per architecture.
- **Total runs:** 3 architectures × 5 seeds = 15 train runs + 15 eval runs.
- **Pass criteria per architecture:** median holdout PnL > 0 AND `count(holdout PnL > 0)` ≥ 3 of 5 (over the new `_p4a_` runs only — old `_p2_` samples are reported but don't gate the decision since they're unseeded and may not generalize).
- **Promotion list (from open-decisions):** Pick 1 + Pick 2. Pick 3 is conditional on `lb500` passing 4A.

---

## Task 1: Add `seed` field to ModelConfig

**Files:**
- Modify: `backend/src/trainer/config.py`
- Test: `backend/tests/trainer/test_seed_plumbing.py` (create)

- [ ] **Step 1: Write the failing test for round-trip**

Create `backend/tests/trainer/test_seed_plumbing.py`:

```python
from trainer.config import ModelConfig


def test_model_config_seed_default_is_none():
    cfg = ModelConfig(name="t", symbols=["BTCUSDT"], intervals=["4h"])
    assert cfg.seed is None


def test_model_config_seed_round_trip():
    cfg = ModelConfig(name="t", symbols=["BTCUSDT"], intervals=["4h"], seed=1001)
    d = cfg.to_dict()
    assert d["seed"] == 1001
    restored = ModelConfig.from_dict(d)
    assert restored.seed == 1001


def test_model_config_seed_omitted_in_legacy_dict():
    # Older configs persisted before the seed field existed: from_dict must not crash.
    legacy = {"name": "t", "symbols": ["BTCUSDT"], "intervals": ["4h"]}
    restored = ModelConfig.from_dict(legacy)
    assert restored.seed is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/trainer/test_seed_plumbing.py -v`
Expected: 3 FAILs — `AttributeError: 'ModelConfig' object has no attribute 'seed'` (or KeyError on `d["seed"]`).

- [ ] **Step 3: Add the field and round-trip support**

Edit `backend/src/trainer/config.py`. After `learning_rate: float = 3e-4` (around line 100), add:

```python
    # Optional integer seed for reproducibility. Forwarded to SB3 algo and to env.reset.
    # None means non-deterministic (matches pre-Phase-4A behavior). Set explicitly for
    # seed-variance studies where reruns must match.
    seed: int | None = None
```

In `to_dict` (around line 105), add `"seed": self.seed,` to the returned dict — placement next to `learning_rate` keeps related fields together:

```python
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "snapshot_interval": self.snapshot_interval,
```

`from_dict` already filters by `__dataclass_fields__` and won't need changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/trainer/test_seed_plumbing.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/trainer/config.py backend/tests/trainer/test_seed_plumbing.py
git commit -m "feat(trainer): add optional seed field to ModelConfig"
```

---

## Task 2: Forward seed from trainer to SB3 algorithm

**Files:**
- Modify: `backend/src/trainer/training/trainer.py:294-299`
- Test: `backend/tests/trainer/test_seed_plumbing.py` (extend)

- [ ] **Step 1: Write the failing test**

Source-inspection test: lighter than a full integration mock and won't leak the `PnlSnapshotCallback` DB-writer thread that starts on construction. Append to `backend/tests/trainer/test_seed_plumbing.py`:

```python
import inspect

from trainer.training import trainer as trainer_mod


def test_train_model_passes_seed_to_algo_constructor():
    """train_model must forward config.seed to algo_cls(...).

    This is a source-level check: we want a build-time guarantee that the seed
    plumbing exists. SB3 itself is trusted to consume `seed=` (well-tested upstream).
    """
    src = inspect.getsource(trainer_mod.train_model)
    assert "seed=config.seed" in src, (
        "train_model must construct the algo with seed=config.seed; "
        "either the field was renamed or the kwarg was dropped."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/trainer/test_seed_plumbing.py::test_train_model_passes_seed_to_algo_constructor -v`
Expected: FAIL — `seed=config.seed` is not in the source yet.

- [ ] **Step 3: Forward seed to algo constructor**

Edit `backend/src/trainer/training/trainer.py` around lines 294–299. Change:

```python
        model = algo_cls(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            verbose=0,
        )
```

to:

```python
        model = algo_cls(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            seed=config.seed,
            verbose=0,
        )
```

(SB3's PPO, A2C, and SAC all accept `seed: int | None`. Passing `None` matches their default and preserves existing behavior for unseeded configs.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/trainer/test_seed_plumbing.py -v`
Expected: 4 PASS (3 from Task 1 + 1 new).

- [ ] **Step 5: Run the full trainer test suite to confirm no regressions**

Run: `cd backend && uv run pytest tests/trainer/ -v`
Expected: all PASS, no new failures.

- [ ] **Step 6: Commit**

```bash
git add backend/src/trainer/training/trainer.py backend/tests/trainer/test_seed_plumbing.py
git commit -m "feat(trainer): forward ModelConfig.seed to SB3 algorithm"
```

---

## Task 3: Write `sweep_phase4a.py` to register 15 configs

**Files:**
- Create: `backend/scripts/sweep_phase4a.py`
- Test: extend `backend/tests/trainer/test_seed_plumbing.py` with a builder test

- [ ] **Step 1: Write the failing test for the config builder**

Append to `backend/tests/trainer/test_seed_plumbing.py`:

```python
def test_phase4a_builder_produces_15_configs():
    import importlib.util, pathlib

    backend_root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "sweep_phase4a", backend_root / "scripts" / "sweep_phase4a.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    configs = mod.build_phase4a_configs()
    assert len(configs) == 15

    names = sorted(c.name for c in configs)
    expected_names = sorted(
        f"btc_4h_a2c_lb{lb}_3em4_p4a_s{s}"
        for lb in (100, 250, 500)
        for s in range(5)
    )
    assert names == expected_names

    # Every config has an explicit integer seed; no two configs at the same lb share a seed.
    seeds_by_lb: dict[int, set[int]] = {}
    for c in configs:
        assert isinstance(c.seed, int)
        seeds_by_lb.setdefault(c.lookback_window, set()).add(c.seed)
    for lb, seeds in seeds_by_lb.items():
        assert len(seeds) == 5, f"lb={lb} has {len(seeds)} unique seeds"

    # All p4a configs share architecture: 4h, A2C, lr=3e-4, 1M timesteps.
    for c in configs:
        assert c.intervals == ["4h"]
        assert c.algorithm == "A2C"
        assert c.learning_rate == 3e-4
        assert c.total_timesteps == 1_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/trainer/test_seed_plumbing.py::test_phase4a_builder_produces_15_configs -v`
Expected: FAIL — `FileNotFoundError` (script doesn't exist yet).

- [ ] **Step 3: Create the script**

Create `backend/scripts/sweep_phase4a.py`:

```python
#!/usr/bin/env python3
"""
Phase 4A — Seed robustness sweep.

Registers 15 ModelConfigs (3 architectures × 5 explicit seeds) for the top P2
architectures (4h, A2C, lr=3e-4, lookback ∈ {100, 250, 500}, 1M timesteps).

Goal: characterize seed variance so we can reject architectures whose median
seed is unprofitable. Pass criterion (per architecture, applied in
phase4a_summary.py): median holdout PnL > 0 AND ≥3 of 5 seeds positive.

Run from repo: cd backend && uv run python scripts/sweep_phase4a.py
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

LOOKBACKS = [100, 250, 500]
SEEDS = [1001, 2002, 3003, 4004, 5005]
LEARNING_RATE = 3e-4
TIMESTEPS = 1_000_000
INTERVAL = "4h"
ALGO = "A2C"
PHASE = "p4a"


def build_phase4a_configs() -> list[ModelConfig]:
    """Return the 15 ModelConfigs that make up the Phase 4A sweep.

    Pure (no DB I/O); separated so tests can validate the spec without a database.
    """
    configs: list[ModelConfig] = []
    for lookback, (seed_idx, seed) in product(LOOKBACKS, enumerate(SEEDS)):
        name = f"btc_{INTERVAL}_{ALGO.lower()}_lb{lookback}_3em4_{PHASE}_s{seed_idx}"
        configs.append(
            ModelConfig(
                name=name,
                symbols=["BTCUSDT"],
                intervals=[INTERVAL],
                columns=list(ALL_KLINE_COLUMNS),
                exchange=ExchangeConfig(),
                lookback_window=lookback,
                algorithm=ALGO,
                learning_rate=LEARNING_RATE,
                total_timesteps=TIMESTEPS,
                seed=seed,
            )
        )
    return configs


def main() -> None:
    configs = build_phase4a_configs()
    print(f"Registering {len(configs)} Phase 4A configs...")
    for cfg in configs:
        save_model_config(cfg)
        print(f"  Registered: {cfg.name}  seed={cfg.seed}")
    print(f"\nDone. {len(configs)} Phase 4A configs registered.")
    print("Train: uv run train worker  |  Or: bash ../infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/trainer/test_seed_plumbing.py::test_phase4a_builder_produces_15_configs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/sweep_phase4a.py backend/tests/trainer/test_seed_plumbing.py
git commit -m "feat(scripts): add sweep_phase4a for seed-robustness sweep"
```

---

## Task 4: Write `phase4a_summary.py` (decision-matrix aggregation)

**Files:**
- Create: `backend/scripts/phase4a_summary.py`
- Test: `backend/tests/trainer/test_phase4a_summary.py`

The summary script answers one question per architecture: pass or fail? It separates the pure decision logic from DB I/O so it's unit-testable.

- [ ] **Step 1: Write the failing test for the decision logic**

Create `backend/tests/trainer/test_phase4a_summary.py`:

```python
import importlib.util
import pathlib

import pytest


def _load_module():
    backend_root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "phase4a_summary", backend_root / "scripts" / "phase4a_summary.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


def test_evaluate_arch_pass_when_median_positive_and_3_of_5_positive(mod):
    # 5 holdout PnLs: 3 positive, 2 negative; median is the middle (positive).
    pnls = [-1000.0, -500.0, 100.0, 5000.0, 8000.0]
    result = mod.evaluate_arch(arch="lb500", p4a_pnls=pnls, p2_pnls=[])
    assert result["pass"] is True
    assert result["count_positive"] == 3
    assert result["median_pnl"] == pytest.approx(100.0)


def test_evaluate_arch_fail_when_only_2_of_5_positive(mod):
    pnls = [-2000.0, -1000.0, -500.0, 200.0, 800.0]  # median is negative
    result = mod.evaluate_arch(arch="lb250", p4a_pnls=pnls, p2_pnls=[])
    assert result["pass"] is False
    assert result["count_positive"] == 2
    assert result["median_pnl"] == pytest.approx(-500.0)


def test_evaluate_arch_fail_when_median_negative_even_if_3_positive(mod):
    # Edge case: 3 of 5 positive but tiny; 2 of 5 negative but huge → median negative.
    # With sorted = [-100000, -50000, 1, 2, 3], median is 1 — actually positive.
    # Construct a real failure: 5 values where median <= 0.
    pnls = [-1000.0, -500.0, 0.0, 100.0, 200.0]
    result = mod.evaluate_arch(arch="lb100", p4a_pnls=pnls, p2_pnls=[])
    assert result["pass"] is False  # median is 0.0, not > 0
    assert result["median_pnl"] == pytest.approx(0.0)


def test_evaluate_arch_reports_p2_observations_separately(mod):
    p4a = [100.0, 200.0, 300.0, 400.0, 500.0]
    p2 = [126547.0, 3465.0, -9987.0]  # the actual P2 lb500 numbers
    result = mod.evaluate_arch(arch="lb500", p4a_pnls=p4a, p2_pnls=p2)
    assert result["pass"] is True  # decision uses p4a only
    assert result["p2_count"] == 3
    assert result["p2_median_pnl"] == pytest.approx(3465.0)


def test_evaluate_arch_handles_partial_seed_completion(mod):
    # Only 4 of 5 seeds finished; report it but don't make a final decision.
    pnls = [100.0, 200.0, 300.0, 400.0]
    result = mod.evaluate_arch(arch="lb500", p4a_pnls=pnls, p2_pnls=[])
    assert result["pass"] is None  # incomplete
    assert result["incomplete_reason"] == "only 4 of 5 p4a seeds have eval results"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/trainer/test_phase4a_summary.py -v`
Expected: FAIL — `FileNotFoundError` (script doesn't exist).

- [ ] **Step 3: Create the summary script**

Create `backend/scripts/phase4a_summary.py`:

```python
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
    rows = conn.execute(
        """
        SELECT tr_eval.total_pnl
        FROM model_configs mc
        JOIN training_runs tr_eval ON tr_eval.model_config_id = mc.id
            AND tr_eval.run_type = 'evaluate' AND tr_eval.status = 'completed'
        WHERE mc.name LIKE %s
        ORDER BY mc.name
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/trainer/test_phase4a_summary.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/phase4a_summary.py backend/tests/trainer/test_phase4a_summary.py
git commit -m "feat(scripts): add phase4a_summary decision-matrix aggregation"
```

---

## Task 5: Register the 15 configs in the database

This is operational, not code. It runs against the production trainer DB.

**Files:**
- None modified.

- [ ] **Step 1: Verify DB connectivity from the local machine**

Run: `cd backend && uv run python -c "from ingester.db import connect; conn = connect(); print(conn.execute('SELECT count(*) FROM model_configs WHERE name LIKE %s', ('%_p4a_%',)).fetchone())"`
Expected: prints `(0,)` (no _p4a_ configs registered yet).

If this fails with a connection error, check `backend/.env` has a working `DATABASE_URL` pointing at the trainer DB.

- [ ] **Step 2: Register the 15 configs**

Run: `cd backend && uv run python scripts/sweep_phase4a.py`
Expected: 15 lines `Registered: btc_4h_a2c_lb<N>_3em4_p4a_s<I>  seed=<S>`, then `Done. 15 Phase 4A configs registered.`

- [ ] **Step 3: Verify registration**

Run: `cd backend && uv run python -c "from ingester.db import connect; conn = connect(); rows = conn.execute('SELECT name, config_json->>\\'seed\\' FROM model_configs WHERE name LIKE %s ORDER BY name', ('%_p4a_%',)).fetchall(); [print(r) for r in rows]"`
Expected: 15 rows, each with a non-null `seed` column matching the values in `SEEDS = [1001, 2002, 3003, 4004, 5005]` (cycling per lookback).

- [ ] **Step 4: Stop and report — no commit needed**

(This step modifies the database, not the repo. Move on to Task 6.)

---

## Task 6: Run the training sweep on the remote training host

Each run is ~19 minutes per the existing P2 numbers; 15 runs ≈ 5 hours wall-clock if serialized on a single GPU. The existing `train worker` claims-and-trains loop handles this — no new code needed.

**Files:**
- None.

- [ ] **Step 1: Confirm no stale claims block claiming**

Use the SSH MCP for the training host to run:
```
cd /opt/tradan/backend && /root/.local/bin/uv run train release-claims --older-than-seconds 600
```
Expected: either "No stale claims found" or a list with a confirmation prompt (answer `y` only if those models are genuinely orphaned).

- [ ] **Step 2: Confirm the 15 p4a models are listed as pending**

Run on the training host:
```
cd /opt/tradan/backend && /root/.local/bin/uv run train list --status pending --names-only | grep _p4a_ | wc -l
```
Expected: `15`.

- [ ] **Step 3: Start the worker**

Run on the training host (in tmux/screen so it survives disconnect):
```
cd /opt/tradan/backend && /root/.local/bin/uv run train worker --poll-seconds 0
```
Expected: worker logs `claimed: btc_4h_a2c_lb…_p4a_s…`, then training progress every minute, then `run #N done` per claim. With `--poll-seconds 0` the worker exits when the queue drains.

- [ ] **Step 4: Monitor progress**

In a separate shell, periodically run:
```
cd /opt/tradan/backend && /root/.local/bin/uv run train list --status completed | grep _p4a_
```
Expected: count grows from 0 → 15 over ~5h.

- [ ] **Step 5: Confirm all 15 train runs completed**

Run on the training host:
```sql
SELECT count(*) FROM training_runs tr
JOIN model_configs mc ON mc.id = tr.model_config_id
WHERE mc.name LIKE '%_p4a_%'
  AND tr.run_type = 'train' AND tr.status = 'completed';
```
Expected: `15`.

If any failed (`status = 'failed'`), inspect the error column. Common causes: OOM (rare on 4h), DB transient. Re-run the worker — failed runs leave the model claimable again after `release-claims`.

- [ ] **Step 6: No commit. Move to Task 7.**

---

## Task 7: Run holdout evaluation on all 15 trained runs

**Files:**
- None modified. Reuses `infra/scripts/evaluate_winners.sh` (which uses `winners_no_eval.sql` to find the 20 best train runs without an eval and runs `train evaluate` on them in parallel — `JOBS=4`).

- [ ] **Step 1: Confirm the queue contains exactly the 15 _p4a_ runs**

`winners_no_eval.sql` returns top-20 by `total_pnl` desc, no phase filter. After Task 6 the only newly-completed unevaluated runs should be the 15 _p4a_ runs (older _p1_/_p2_/_p3_ runs already have evals). Run on the training host:

```
cd /opt/tradan/backend && psql "$DATABASE_URL" -f scripts/winners_no_eval.sql
```
Expected: ~15 rows, all matching `_p4a_` model names.

- [ ] **Step 2: Run the eval batch script**

Run on the training host:
```
cd /opt/tradan/backend && bash scripts/evaluate_winners.sh
```
Expected: 15 evaluation runs complete in parallel batches of 4. Final line: `Evaluation complete.`

If `winners_no_eval.sh` returned >15 (because some old training run also lacks an eval), evaluate-only-p4a manually instead:

```
cd /opt/tradan/backend && for name in $(uv run train list --status completed --names-only | grep _p4a_); do
  run_id=$(psql "$DATABASE_URL" -t -A -c "SELECT id FROM training_runs tr JOIN model_configs mc ON mc.id = tr.model_config_id WHERE mc.name='$name' AND tr.run_type='train' AND tr.status='completed' AND NOT EXISTS (SELECT 1 FROM training_runs ev WHERE ev.model_config_id = mc.id AND ev.run_type='evaluate') ORDER BY tr.id DESC LIMIT 1")
  [ -n "$run_id" ] && uv run train evaluate --model "$name" --run "$run_id"
done
```

- [ ] **Step 3: Verify all 15 evals completed**

Run on the training host:
```sql
SELECT count(*) FROM training_runs tr
JOIN model_configs mc ON mc.id = tr.model_config_id
WHERE mc.name LIKE '%_p4a_%'
  AND tr.run_type = 'evaluate' AND tr.status = 'completed';
```
Expected: `15`.

- [ ] **Step 4: No commit.**

---

## Task 8: Run the decision matrix and record the outcome

**Files:**
- Modify: `docs/plans/2026-05-09-phase4-training-plan.md`

- [ ] **Step 1: Run the summary**

Locally (or on the training host — both have access to the same DB):

```
cd backend && uv run python scripts/phase4a_summary.py
```
Expected output (numbers will vary):
```
arch     seeds  pos     median PnL      P2 median  decision
----------------------------------------------------------------------
lb100        5    4         $+1,200        $+8,795  PASS
lb250        5    2         $-2,400       $+39,168  FAIL
lb500        5    3         $+8,000        $+3,465  PASS
```

- [ ] **Step 2: Capture the raw seed-by-seed table**

Run on the training host (or locally with `psql` against the trainer DB):
```sql
SELECT mc.name,
       ROUND(tr_eval.total_pnl::numeric, 0) AS pnl,
       ROUND(tr_eval.sharpe_ratio::numeric, 2) AS sharpe,
       ROUND((tr_eval.max_drawdown * 100)::numeric, 1) AS dd_pct,
       tr_eval.total_trades,
       (mc.config_json->>'seed')::int AS seed
FROM model_configs mc
JOIN training_runs tr_eval ON tr_eval.model_config_id = mc.id
    AND tr_eval.run_type = 'evaluate' AND tr_eval.status = 'completed'
WHERE mc.name LIKE '%_p4a_%'
ORDER BY mc.name;
```
Save the output as a Markdown table.

- [ ] **Step 3: Update the master plan**

Edit `docs/plans/2026-05-09-phase4-training-plan.md`. In the `### 4A — Seed robustness` section, add an `**Outcome:**` subsection containing:
- The decision-matrix table from Step 1 above
- The raw seed-by-seed table from Step 2
- Per-architecture verdict and what it means for promotion

Update the "Open decisions" list:
- Cross out `[ ]` and replace with `[x]` for "Confirm the 3 models to promote" — record the final list (e.g. "Pick 1 + Pick 2; Pick 3 dropped because lb500 failed median test" or similar).

- [ ] **Step 4: Commit**

```bash
git add docs/plans/2026-05-09-phase4-training-plan.md
git commit -m "docs: record Phase 4A seed-robustness results"
```

---

## Task 9: Wrap-up checks

- [ ] **Step 1: Confirm the suite still passes**

Run: `cd backend && uv run pytest tests/trainer/ -v`
Expected: all PASS.

- [ ] **Step 2: Confirm no leaked claims**

Run on the training host: `cd /opt/tradan/backend && /root/.local/bin/uv run train release-claims --older-than-seconds 600`
Expected: "No stale claims found."

- [ ] **Step 3: Open a PR**

```bash
git push -u origin <branch-name>
gh pr create --title "Phase 4A: seed-robustness sweep + plumb seed through trainer" \
  --body "$(cat <<'EOF'
## Summary
- Plumbs `ModelConfig.seed` through to SB3 (PPO/A2C/SAC). Previously the SEEDS list in sweep_phase{1,2,3}.py was unused — variance came from non-deterministic init only.
- Adds `sweep_phase4a.py` to register 15 configs (3 lookbacks × 5 explicit seeds) and `phase4a_summary.py` for the per-architecture pass/fail decision.
- Records 4A outcome in the Phase 4 plan.

## Test plan
- [x] Unit tests for ModelConfig seed round-trip
- [x] Unit tests for trainer forwarding seed to SB3 (mocked algo)
- [x] Unit tests for the `evaluate_arch` decision logic
- [x] Manual: 15 train runs completed, 15 evals completed, summary printed
EOF
)"
```

(Skip the PR if running locally without push intent — leave the commits on the worktree branch for review.)

---

## Why this plan looks the way it does

- **The seed plumbing fix is in scope, not deferred.** It's a 4-line code change with full test coverage. Skipping it would mean 4A measures the same non-deterministic variance P2 already measured — defeating the purpose. Adding it now also unlocks reproducibility for 4B/4C/4D.
- **No SQL view, just a Python script.** `winners.sql` exists but its filter (`gen_ratio > 0.5`, `dd < 0.25`) doesn't match 4A's pass criteria (`median > 0`, `≥3/5 positive`). A separate script keeps the existing winners flow untouched while making 4A's logic unit-testable.
- **The pure builder + pure evaluator split** (`build_phase4a_configs()` in sweep_phase4a, `evaluate_arch()` in phase4a_summary) is what makes the tests practical. Don't merge them with their `main()` callers.
- **Old `_p2_` runs are reported but not part of the gate.** They were unseeded and could be replaying lucky regimes; trusting them as "extra samples" muddles the decision.
- **No new infra.** Reuses `train worker`, `evaluate_winners.sh`, the existing rsync target, the existing claim system. The plan is mostly "register 15 things, push them through the pipeline, look at the answer."

---

## Self-review checklist (run before handoff)

- [x] Spec coverage — every line of "4A — Seed robustness" in the master plan maps to a task: 5 seeds × 3 archs (Task 3), 1M timesteps (Task 3), success criterion median+positive (Task 4), reject failures from promotion list (Task 8 update).
- [x] Type consistency — `evaluate_arch` arg names (`p4a_pnls`, `p2_pnls`) match in tests and implementation; `build_phase4a_configs` returns `list[ModelConfig]` and tests check `.lookback_window`/`.seed`/`.intervals` attributes that exist on ModelConfig.
- [x] No placeholders — every test has full code, every command has expected output, no "TODO" or "implement later".
- [x] Commits frequent — each task ends in a commit (or marks itself as op-only).
