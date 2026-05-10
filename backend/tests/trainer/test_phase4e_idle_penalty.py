import importlib.util
import pathlib

import pytest


def _load(name: str):
    backend_root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        name, backend_root / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def summary_mod():
    return _load("phase4e_idle_penalty_summary")


def test_summary_reuses_phase4a_evaluate_arch(summary_mod):
    """Decision logic must come from phase4a_summary, not a reimplementation —
    same behavioral check used in the 4C/4D summary tests."""
    p4a = _load("phase4a_summary")
    summary_src = pathlib.Path(summary_mod.evaluate_arch.__code__.co_filename).name
    p4a_src = pathlib.Path(p4a.evaluate_arch.__code__.co_filename).name
    assert summary_src == "phase4a_summary.py"
    assert p4a_src == "phase4a_summary.py"
    args = dict(arch="x", p4a_pnls=[-1.0, -2.0, -3.0, -4.0, 5.0], p2_pnls=[])
    assert summary_mod.evaluate_arch(**args) == p4a.evaluate_arch(**args)


def test_paired_diff_per_cell_aligns_seeds(summary_mod):
    """paired_diff_per_cell yields one row per (arch, slug, seed) row, with
    each row carrying p4d_pnl, p4e_pnl, delta. The 4D side is keyed by arch
    only (the baseline does not vary with slug); the 4E side is keyed by
    (arch, slug) since each cell is its own configuration."""
    p4d_by_arch = {
        "lb100": {1001: -100.0, 2002: -200.0},
        "lb500": {1001: -500.0, 2002: -600.0},
    }
    p4e_by_cell = {
        ("lb100", "idle05"): {1001: -50.0, 2002: -150.0},
        ("lb100", "idle5"): {1001: -10.0, 2002: -90.0},
        ("lb500", "idle05"): {1001: -300.0, 2002: -400.0},
        ("lb500", "idle5"): {1001: -50.0, 2002: -100.0},
    }

    rows = summary_mod.paired_diff_per_cell(p4d_by_arch, p4e_by_cell)

    # 4 cells × 2 seeds each = 8 rows.
    assert len(rows) == 8

    by_key = {(r["arch"], r["slug"], r["seed"]): r for r in rows}
    # 4D – 4E delta direction: positive delta means 4E (penalty) beat 4D.
    assert by_key[("lb100", "idle05", 1001)]["delta"] == pytest.approx(50.0)   # -50 - (-100)
    assert by_key[("lb100", "idle5", 2002)]["delta"] == pytest.approx(110.0)   # -90 - (-200)
    assert by_key[("lb500", "idle5", 1001)]["delta"] == pytest.approx(450.0)   # -50 - (-500)
    assert by_key[("lb500", "idle05", 2002)]["delta"] == pytest.approx(200.0)  # -400 - (-600)


def test_paired_diff_per_cell_handles_missing_seeds(summary_mod):
    """If a (arch, slug, seed) row exists in only one side, delta is None."""
    p4d_by_arch = {"lb100": {1001: -100.0}}                     # missing 2002 in 4D
    p4e_by_cell = {("lb100", "idle05"): {1001: -50.0, 2002: -150.0}}

    rows = summary_mod.paired_diff_per_cell(p4d_by_arch, p4e_by_cell)
    by_key = {(r["arch"], r["slug"], r["seed"]): r for r in rows}

    assert by_key[("lb100", "idle05", 1001)]["delta"] == pytest.approx(50.0)
    assert by_key[("lb100", "idle05", 2002)]["p4d_pnl"] is None
    assert by_key[("lb100", "idle05", 2002)]["delta"] is None


def test_paired_diff_per_cell_handles_missing_p4e_cells(summary_mod):
    """If a (arch, slug) cell never had any 4E runs, paired_diff_per_cell must
    not crash — it just emits no rows for that cell."""
    p4d_by_arch = {"lb100": {1001: -100.0}}
    # Only one slug present; the other slug is missing entirely.
    p4e_by_cell = {("lb100", "idle05"): {1001: -50.0}}

    rows = summary_mod.paired_diff_per_cell(p4d_by_arch, p4e_by_cell)
    # Only one row, for the slug that actually exists.
    assert len(rows) == 1
    assert rows[0]["arch"] == "lb100"
    assert rows[0]["slug"] == "idle05"
    assert rows[0]["seed"] == 1001
