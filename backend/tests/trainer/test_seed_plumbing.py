import inspect

from trainer.config import ModelConfig
from trainer.training import trainer as trainer_mod


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


def test_model_config_ent_coef_default_is_zero():
    cfg = ModelConfig(name="t", symbols=["BTCUSDT"], intervals=["4h"])
    assert cfg.ent_coef == 0.0


def test_model_config_ent_coef_round_trip():
    cfg = ModelConfig(name="t", symbols=["BTCUSDT"], intervals=["4h"], ent_coef=0.01)
    d = cfg.to_dict()
    assert d["ent_coef"] == 0.01
    restored = ModelConfig.from_dict(d)
    assert restored.ent_coef == 0.01


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


def test_train_model_conditionally_passes_ent_coef():
    """train_model must conditionally forward config.ent_coef to algo_cls(...).

    Conditional because SAC's ent_coef defaults to 'auto' (learnable) and
    forcing it to 0.0 would silently break SAC. Gating on ent_coef > 0 keeps
    the SB3 default in place for any algo when the field is unset.
    """
    src = inspect.getsource(trainer_mod.train_model)
    assert "config.ent_coef" in src, (
        "train_model must reference config.ent_coef in its algo construction."
    )
    assert "config.ent_coef > 0" in src, (
        "train_model must gate ent_coef pass-through on `config.ent_coef > 0` "
        "to preserve SB3's algo-specific defaults (especially SAC's 'auto')."
    )


def test_phase4a_builder_produces_15_configs():
    import importlib.util
    import pathlib

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


def test_phase4d_env_audit_builder_produces_15_configs():
    import importlib.util
    import pathlib

    backend_root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "sweep_phase4d_env_audit",
        backend_root / "scripts" / "sweep_phase4d_env_audit.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    configs = mod.build_phase4d_configs()
    # Same shape as 4A: 3 architectures × 5 seeds.
    assert len(configs) == 15

    expected_names = sorted(
        f"btc_4h_a2c_lb{lb}_3em4_p4d_s{s}"
        for lb in (100, 250, 500)
        for s in range(5)
    )
    assert sorted(c.name for c in configs) == expected_names

    # Seeds match 4A for paired comparison.
    expected_seeds = {1001, 2002, 3003, 4004, 5005}
    seeds_by_lb: dict[int, set[int]] = {}
    for c in configs:
        seeds_by_lb.setdefault(c.lookback_window, set()).add(c.seed)
    for lb, seeds in seeds_by_lb.items():
        assert seeds == expected_seeds, f"lb={lb} seeds: {seeds}"

    # Critical: 4D's whole point is the new env caps. Each config must carry
    # the new (post-audit) defaults baked into its ExchangeConfig.
    for c in configs:
        assert c.exchange.max_leverage == 10.0
        assert c.exchange.max_position_size_pct == 0.25
        assert c.exchange.max_drawdown_pct == 0.5

    for c in configs:
        assert c.intervals == ["4h"]
        assert c.algorithm == "A2C"
        assert c.learning_rate == 3e-4
        assert c.total_timesteps == 1_000_000
        # ent_coef stays at default (0.0) — 4D tests env-only effect.
        assert c.ent_coef == 0.0


def test_phase4e_idle_penalty_builder_produces_30_configs():
    import importlib.util
    import pathlib

    backend_root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "sweep_phase4e_idle_penalty",
        backend_root / "scripts" / "sweep_phase4e_idle_penalty.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    configs = mod.build_phase4e_configs()
    # 3 architectures × 2 penalty values × 5 seeds.
    assert len(configs) == 30

    # Slug rule: 0.05 -> "idle05", 0.5 -> "idle5".
    expected_names = sorted(
        f"btc_4h_a2c_lb{lb}_3em4_{slug}_p4e_s{s}"
        for lb in (100, 250, 500)
        for slug in ("idle05", "idle5")
        for s in range(5)
    )
    assert sorted(c.name for c in configs) == expected_names

    # Each config carries the env audit defaults from Phase 4D.
    for c in configs:
        assert c.exchange.max_leverage == 10.0
        assert c.exchange.max_position_size_pct == 0.25
        assert c.exchange.max_drawdown_pct == 0.5
        assert c.intervals == ["4h"]
        assert c.algorithm == "A2C"
        assert c.learning_rate == 3e-4
        assert c.total_timesteps == 1_000_000
        # ent_coef stays at default (0.0) — 4E tests idle-penalty effect alone.
        assert c.ent_coef == 0.0

    # Penalty value is encoded in the slug AND on the ExchangeConfig.
    for c in configs:
        if "idle05" in c.name:
            assert c.exchange.idle_step_penalty_usd == 0.05
        elif "idle5" in c.name:
            assert c.exchange.idle_step_penalty_usd == 0.5
        else:
            raise AssertionError(f"unexpected slug in {c.name}")

    # Seeds match 4A/4D for paired comparison; one seed per slot per (arch, slug).
    expected_seeds = {1001, 2002, 3003, 4004, 5005}
    seeds_by_cell: dict[tuple[int, str], set[int]] = {}
    for c in configs:
        slug = "idle05" if "idle05" in c.name else "idle5"
        seeds_by_cell.setdefault((c.lookback_window, slug), set()).add(c.seed)
    for (lb, slug), seeds in seeds_by_cell.items():
        assert seeds == expected_seeds, f"({lb},{slug}) seeds: {seeds}"


def test_phase4c_entropy_builder_produces_5_configs():
    import importlib.util
    import pathlib

    backend_root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "sweep_phase4c_entropy", backend_root / "scripts" / "sweep_phase4c_entropy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    configs = mod.build_phase4c_entropy_configs()
    assert len(configs) == 5

    expected_names = sorted(
        f"btc_4h_a2c_lb500_3em4_ent01_p4c_s{s}" for s in range(5)
    )
    assert sorted(c.name for c in configs) == expected_names

    # Seeds match the 4A run for paired comparison; ent_coef and timesteps fixed.
    expected_seeds = {1001, 2002, 3003, 4004, 5005}
    assert {c.seed for c in configs} == expected_seeds
    for c in configs:
        assert c.lookback_window == 500
        assert c.intervals == ["4h"]
        assert c.algorithm == "A2C"
        assert c.learning_rate == 3e-4
        assert c.ent_coef == 0.01
        assert c.total_timesteps == 2_000_000
