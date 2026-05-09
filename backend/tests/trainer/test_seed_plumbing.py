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
