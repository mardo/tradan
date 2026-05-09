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
