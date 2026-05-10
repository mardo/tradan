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
    return _load("phase4d_summary")


def test_summary_reuses_phase4a_evaluate_arch(summary_mod):
    """Decision logic comes from phase4a_summary, not a reimplementation —
    same behavioral check used in the 4C summary tests."""
    p4a = _load("phase4a_summary")
    summary_src = pathlib.Path(summary_mod.evaluate_arch.__code__.co_filename).name
    p4a_src = pathlib.Path(p4a.evaluate_arch.__code__.co_filename).name
    assert summary_src == "phase4a_summary.py"
    assert p4a_src == "phase4a_summary.py"
    args = dict(arch="x", p4a_pnls=[-1.0, -2.0, -3.0, -4.0, 5.0], p2_pnls=[])
    assert summary_mod.evaluate_arch(**args) == p4a.evaluate_arch(**args)


def test_paired_diff_per_arch_aligns_seeds(summary_mod):
    """paired_diff_per_arch returns a list-of-rows per architecture, with each
    row carrying (arch, seed, p4a_pnl, p4d_pnl, delta). Used to print the
    side-by-side comparison table."""
    # Per-arch dictionaries: {seed: pnl}
    p4a_by_arch = {
        "lb100": {1001: -596.0, 2002: -4851.0},
        "lb500": {1001: -5195.0, 2002: -10155.0},
    }
    p4d_by_arch = {
        "lb100": {1001: 100.0, 2002: -200.0},
        "lb500": {1001: -1000.0, 2002: -2000.0},
    }

    rows = summary_mod.paired_diff_per_arch(p4a_by_arch, p4d_by_arch)

    # 4 rows total: 2 archs × 2 seeds each.
    assert len(rows) == 4
    by_key = {(r["arch"], r["seed"]): r for r in rows}

    assert by_key[("lb100", 1001)]["delta"] == pytest.approx(696.0)  # 100 - (-596)
    assert by_key[("lb500", 2002)]["delta"] == pytest.approx(8155.0)  # -2000 - (-10155)


def test_paired_diff_per_arch_handles_missing_seeds(summary_mod):
    """If a seed exists in only one phase for a given arch, delta is None."""
    p4a_by_arch = {"lb100": {1001: -596.0, 2002: -4851.0}}
    p4d_by_arch = {"lb100": {1001: 100.0}}  # 2002 missing in 4D
    rows = summary_mod.paired_diff_per_arch(p4a_by_arch, p4d_by_arch)
    by_key = {(r["arch"], r["seed"]): r for r in rows}
    assert by_key[("lb100", 1001)]["delta"] == pytest.approx(696.0)
    assert by_key[("lb100", 2002)]["p4d_pnl"] is None
    assert by_key[("lb100", 2002)]["delta"] is None
