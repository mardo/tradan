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
