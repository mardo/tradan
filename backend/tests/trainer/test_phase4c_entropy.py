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
    return _load("phase4c_entropy_summary")


def test_summary_reuses_phase4a_evaluate_arch(summary_mod):
    """Decision logic must come from phase4a_summary, not a reimplementation.
    importlib loads both modules as fresh instances, so identity check would
    spuriously fail; instead verify the function lives in phase4a_summary's
    source file and behaves identically on a representative input."""
    p4a = _load("phase4a_summary")
    # Source file check: the function the 4C script uses must originate in 4A's file.
    summary_src = pathlib.Path(summary_mod.evaluate_arch.__code__.co_filename).name
    p4a_src = pathlib.Path(p4a.evaluate_arch.__code__.co_filename).name
    assert summary_src == "phase4a_summary.py"
    assert p4a_src == "phase4a_summary.py"

    # Behavior check: identical results on a known input.
    args = dict(arch="x", p4a_pnls=[-1.0, -2.0, -3.0, -4.0, 5.0], p2_pnls=[])
    assert summary_mod.evaluate_arch(**args) == p4a.evaluate_arch(**args)


def test_paired_diff_aligns_seeds(summary_mod):
    """paired_diff must align the two PnL series by seed, returning one row
    per (seed, p4a_pnl, p4c_pnl, delta). Missing seeds in either series
    yield None for that side."""
    p4a = {1001: -5195.0, 2002: -10155.0, 3003: -10316.0, 4004: -11703.0, 5005: -7428.0}
    p4c = {1001: -2000.0, 2002: -5000.0, 3003: 1000.0, 4004: -8000.0, 5005: -3000.0}

    rows = summary_mod.paired_diff(p4a, p4c)
    assert len(rows) == 5

    by_seed = {r["seed"]: r for r in rows}
    assert by_seed[1001]["p4a_pnl"] == pytest.approx(-5195.0)
    assert by_seed[1001]["p4c_pnl"] == pytest.approx(-2000.0)
    assert by_seed[1001]["delta"] == pytest.approx(3195.0)


def test_paired_diff_handles_missing_seeds(summary_mod):
    """If a seed is present in only one series, the other side reports None
    and delta is None (can't compute without both)."""
    p4a = {1001: -5195.0, 2002: -10155.0}
    p4c = {1001: -2000.0, 3003: 500.0}

    rows = summary_mod.paired_diff(p4a, p4c)
    by_seed = {r["seed"]: r for r in rows}

    assert by_seed[1001]["delta"] == pytest.approx(3195.0)
    assert by_seed[2002]["p4c_pnl"] is None
    assert by_seed[2002]["delta"] is None
    assert by_seed[3003]["p4a_pnl"] is None
    assert by_seed[3003]["delta"] is None
