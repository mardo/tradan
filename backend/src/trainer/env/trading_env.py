from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trainer.config import ModelConfig
from trainer.env.account import Account
from trainer.env.data_feed import DataFeed
from trainer.env.exchange_sim import ExchangeSim


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ModelConfig, data_feed: DataFeed) -> None:
        super().__init__()
        self.config = config
        self.data_feed = data_feed

        self.account = Account(initial_balance=config.initial_balance)
        self.exchange = ExchangeSim(config=config.exchange, account=self.account)

        exc = config.exchange
        num_features = data_feed.num_features
        lookback = config.lookback_window

        self.observation_space = spaces.Dict({
            "market": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback, num_features), dtype=np.float32,
            ),
            "account": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(5,), dtype=np.float32,
            ),
            "orders": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(exc.max_open_orders, 11), dtype=np.float32,
            ),
            "positions": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(exc.max_open_positions, 6), dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(config.action_size,), dtype=np.float32,
        )

        self._current_step = 0
        self._prev_equity = config.initial_balance
        self.pnl_history: list[dict] = []

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._current_step = 0
        self._prev_equity = self.config.initial_balance
        self.exchange.reset()
        self.pnl_history.clear()
        return self._build_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}

        prices = self.data_feed.get_candle_prices(self._current_step)
        high = prices.get("high", prices.get("close", 0.0))
        low = prices.get("low", prices.get("close", 0.0))
        close = prices.get("close", 0.0)

        if close <= 0:
            close = 1.0
            high = max(high, 1.0)
            low = max(low, 0.1)

        events = self.exchange.process_candle(high=high, low=low, close=close)
        info["fill_events"] = len(events)

        self._process_actions(action, close, info)

        unrealized = self.exchange.total_unrealized_pnl(close)
        equity = self.account.equity(unrealized)
        reward = float(equity - self._prev_equity)
        self._prev_equity = equity

        self.pnl_history.append({
            "step": self._current_step,
            "candle_time": self.data_feed.get_timestamp(self._current_step),
            "balance": self.account.balance,
            "equity": equity,
            "unrealized_pnl": unrealized,
            "open_position_count": len(self.exchange.open_positions),
            "open_order_count": len(self.exchange.open_orders),
        })

        terminated = equity <= 0
        self._current_step += 1
        truncated = self._current_step >= self.data_feed.total_steps

        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    def _process_actions(self, action: np.ndarray, close: float, info: dict) -> None:
        from trainer.env.action_decoder import DecoderState, decode_action
        state = DecoderState(
            close=close,
            available_balance=self.account.available_balance,
            num_open_orders=len(self.exchange.open_orders),
            num_open_positions=len(self.exchange.open_positions),
        )
        intent = decode_action(action, state, self.config)
        info.update(self.exchange.apply_intent(intent, current_price=close))

    def _build_observation(self) -> dict[str, np.ndarray]:
        from trainer.env.observation import (
            ObservationConfig, ObservationInputs, build_observation,
        )
        step = min(self._current_step, self.data_feed.total_steps - 1)
        market = self.data_feed.get_observation(step)
        raw = self.data_feed.get_current_raw(step)
        close = float(raw[self.data_feed.price_columns.get("close", 3)])
        if close <= 0:
            close = 1.0

        unrealized = self.exchange.total_unrealized_pnl(close)
        inputs = ObservationInputs(
            market=market,
            balance=self.account.balance,
            equity=self.account.equity(unrealized),
            unrealized_pnl=unrealized,
            margin_used=self.account.margin_used,
            available_balance=self.account.available_balance,
            open_orders=self.exchange.open_orders,
            open_positions=self.exchange.open_positions,
            close=close,
        )
        cfg = ObservationConfig(
            lookback=self.config.lookback_window,
            num_features=self.data_feed.num_features,
            max_open_orders=self.config.exchange.max_open_orders,
            max_open_positions=self.config.exchange.max_open_positions,
            max_leverage=self.config.exchange.max_leverage,
            initial_balance=self.config.initial_balance,
        )
        return build_observation(inputs, cfg)
