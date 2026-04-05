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

    def _process_actions(
        self, action: np.ndarray, close: float, info: dict[str, Any]
    ) -> None:
        num_tp = self.config.num_tp_levels
        exc = self.config.exchange

        idx = 1 + 1 + 1 + 1 + num_tp + num_tp + 1
        cancel_start = idx
        cancel_end = cancel_start + exc.max_open_orders
        close_start = cancel_end
        close_end = close_start + exc.max_open_positions

        cancelled = 0
        cancel_signals = action[cancel_start:cancel_end]
        indices_to_cancel = [
            i for i in range(len(self.exchange.open_orders))
            if i < len(cancel_signals) and cancel_signals[i] > 0.0
        ]
        for i in sorted(indices_to_cancel, reverse=True):
            self.exchange.cancel_order(i)
            cancelled += 1
        info["orders_cancelled"] = cancelled

        closed = 0
        close_signals = action[close_start:close_end]
        for i in range(min(len(self.exchange.open_positions), len(close_signals))):
            frac = float(max(0.0, min(1.0, (close_signals[i] + 1.0) / 2.0)))
            if frac > 0.05:
                self.exchange.close_position(i, frac, close)
                closed += 1
        info["positions_closed"] = closed

        open_conf = (action[0] + 1.0) / 2.0
        placed = 0
        if open_conf > 0.5:
            direction = 1 if action[1] > 0.0 else -1

            offset_pct = action[2] * self.config.max_trigger_offset_pct / 100.0
            trigger_price = close * (1.0 + offset_pct)

            sl_raw = (action[3] + 1.0) / 2.0
            sl_dist_pct = (
                self.config.min_sl_pct
                + sl_raw * (self.config.max_sl_pct - self.config.min_sl_pct)
            ) / 100.0
            if direction == 1:
                sl_price = trigger_price * (1.0 - sl_dist_pct)
            else:
                sl_price = trigger_price * (1.0 + sl_dist_pct)

            tp_prices: list[float] = []
            tp_size_pcts: list[float] = []
            raw_tp_sizes: list[float] = []
            for j in range(num_tp):
                tp_raw = (action[4 + j] + 1.0) / 2.0
                tp_dist_pct = tp_raw * self.config.max_tp_pct / 100.0
                tp_dist_pct = max(tp_dist_pct, 0.001)
                if direction == 1:
                    tp_price = trigger_price * (1.0 + tp_dist_pct)
                else:
                    tp_price = trigger_price * (1.0 - tp_dist_pct)
                tp_prices.append(tp_price)
                raw_size = max((action[4 + num_tp + j] + 1.0) / 2.0, 0.01)
                raw_tp_sizes.append(raw_size)

            total = sum(raw_tp_sizes)
            tp_size_pcts = [s / total for s in raw_tp_sizes]

            size_raw = (action[4 + 2 * num_tp] + 1.0) / 2.0
            margin = size_raw * self.account.available_balance

            if margin >= self.config.exchange.min_order_size_usd:
                order = self.exchange.place_order(
                    direction=direction,
                    trigger_price=trigger_price,
                    sl_price=sl_price,
                    tp_prices=tp_prices,
                    tp_size_pcts=tp_size_pcts,
                    margin=margin,
                )
                if order is not None:
                    placed = 1
        info["orders_placed"] = placed

    def _build_observation(self) -> dict[str, np.ndarray]:
        step = min(self._current_step, self.data_feed.total_steps - 1)
        market = self.data_feed.get_observation(step)

        raw = self.data_feed.get_current_raw(step)
        close = float(raw[self.data_feed.price_columns.get("close", 3)])
        if close <= 0:
            close = 1.0

        init = self.config.initial_balance
        unrealized = self.exchange.total_unrealized_pnl(close)
        account_state = np.array([
            self.account.balance / init,
            self.account.equity(unrealized) / init,
            unrealized / init,
            self.account.margin_used / init,
            self.account.available_balance / init,
        ], dtype=np.float32)

        exc = self.config.exchange
        orders = np.zeros((exc.max_open_orders, 11), dtype=np.float32)
        for i, order in enumerate(self.exchange.open_orders[:exc.max_open_orders]):
            orders[i, 0] = 1.0
            orders[i, 1] = float(order.direction)
            orders[i, 2] = order.trigger_price / close
            orders[i, 3] = order.sl_price / close
            for j, tp in enumerate(order.tp_prices[:3]):
                orders[i, 4 + j] = tp / close
            for j, pct in enumerate(order.tp_size_pcts[:3]):
                orders[i, 7 + j] = pct
            orders[i, 10] = order.margin / init

        positions = np.zeros((exc.max_open_positions, 6), dtype=np.float32)
        for i, pos in enumerate(self.exchange.open_positions[:exc.max_open_positions]):
            positions[i, 0] = 1.0
            positions[i, 1] = float(pos.direction)
            positions[i, 2] = pos.entry_price / close
            positions[i, 3] = pos.size * pos.entry_price / init
            positions[i, 4] = pos.unrealized_pnl(close) / init
            positions[i, 5] = pos.leverage / exc.max_leverage

        return {
            "market": market,
            "account": account_state,
            "orders": orders,
            "positions": positions,
        }
