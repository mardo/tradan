import gymnasium as gym
import numpy as np
import pytest

from trainer.config import ExchangeConfig, ModelConfig
from trainer.env.data_feed import DataFeed
from trainer.env.trading_env import TradingEnv


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        name="test",
        symbols=["BTCUSDT"],
        intervals=["1h"],
        columns=["open", "high", "low", "close", "volume"],
        exchange=ExchangeConfig(max_open_orders=5, max_open_positions=5),
        lookback_window=50,
        num_tp_levels=3,
    )


@pytest.fixture
def feed() -> DataFeed:
    n = 200
    rng = np.random.default_rng(42)
    base_price = 50_000.0
    closes = base_price + rng.normal(0, 100, n).cumsum()
    highs = closes + rng.uniform(50, 300, n)
    lows = closes - rng.uniform(50, 300, n)
    opens = closes + rng.normal(0, 50, n)
    volume = rng.uniform(100, 1000, n)
    features = np.column_stack([opens, highs, lows, closes, volume])
    timestamps = np.arange(n, dtype=np.int64)
    return DataFeed(
        timestamps=timestamps, features=features, lookback=50,
        price_columns={"open": 0, "high": 1, "low": 2, "close": 3},
    )


@pytest.fixture
def env(config: ModelConfig, feed: DataFeed) -> TradingEnv:
    return TradingEnv(config=config, data_feed=feed)


def test_env_is_gymnasium_compliant(env: TradingEnv):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "market" in obs
    assert "account" in obs
    assert "orders" in obs
    assert "positions" in obs


def test_observation_shapes(env: TradingEnv, config: ModelConfig):
    obs, _ = env.reset()
    assert obs["market"].shape == (50, 5)
    assert obs["account"].shape == (5,)
    assert obs["orders"].shape == (5, 11)
    assert obs["positions"].shape == (5, 6)


def test_action_space_shape(env: TradingEnv):
    assert env.action_space.shape == (env.config.action_size,)


def test_do_nothing_action(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert isinstance(reward, float)


def test_step_advances_candle(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    env.step(action)
    assert env._current_step == 1


def test_episode_truncates_at_end(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    terminated = False
    truncated = False
    steps = 0
    while not terminated and not truncated:
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
    assert truncated or terminated
    assert steps == env.data_feed.total_steps


def test_open_order_action(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    action[0] = 0.9
    action[1] = 0.8
    action[2] = -0.1
    action[3] = 0.3
    action[4] = 0.3
    action[5] = 0.3
    action[6] = 0.3
    action[7] = 0.33
    action[8] = 0.33
    action[9] = 0.34
    action[10] = 0.1
    obs, _, _, _, info = env.step(action)
    assert info.get("orders_placed", 0) >= 0


def test_reset_returns_fresh_state(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(10):
        env.step(action)
    obs, _ = env.reset()
    assert env._current_step == 0
