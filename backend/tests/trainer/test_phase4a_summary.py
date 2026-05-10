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


def test_evaluate_arch_fail_when_median_zero(mod):
    # 3 of 5 positive but the median is exactly 0 — should fail (median must be > 0).
    pnls = [-1000.0, -500.0, 0.0, 100.0, 200.0]
    result = mod.evaluate_arch(arch="lb100", p4a_pnls=pnls, p2_pnls=[])
    assert result["pass"] is False
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
