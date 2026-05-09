"""LiveRunner — orchestrates feature pipeline → model → action decoder
→ adapter dispatch.

This file ships ONLY the replay entry point. The production loop (run_live)
is implemented in Phase F.
"""
from __future__ import annotations

from dataclasses import dataclass

from live.action_decoder import RiskClampConfig, clamp_intent
from live.exchange.replay import ReplayAdapter
from live.feature_pipeline import build_live_observation
from live.model_runner import ModelRunner
from trainer.config import ModelConfig
from trainer.env.action_decoder import DecoderState, decode_action
from trainer.env.normalization import NormalizationStats
from trainer.env.observation import ObservationConfig


@dataclass
class ReplayResult:
    final_equity: float
    total_steps: int


def run_replay(
    *,
    adapter: ReplayAdapter,
    model_runner: ModelRunner,
    model_config: ModelConfig,
    stats: NormalizationStats,
    max_position_size_pct: float = 1.0,
    max_leverage: float = 125.0,
) -> ReplayResult:
    """Drive the replay adapter forward one candle at a time.

    Mirrors TradingEnv.step's control flow:
      1. process the current candle (adapter.advance does this)
      2. observe (build_live_observation)
      3. predict (model_runner.predict)
      4. decode + clamp (decode_action, clamp_intent)
      5. apply intent via sim.apply_intent (replay path uses the trainer's
         simulator directly, bypassing the adapter's place_order/cancel_order
         which are NotImplementedError on ReplayAdapter)
    """
    obs_cfg = ObservationConfig(
        lookback=model_config.lookback_window,
        num_features=len(model_config.columns),
        max_open_orders=model_config.exchange.max_open_orders,
        max_open_positions=model_config.exchange.max_open_positions,
        max_leverage=model_config.exchange.max_leverage,
        initial_balance=model_config.initial_balance,
    )

    # Prime the adapter so it has at least `lookback` klines available.
    while adapter.cursor < model_config.lookback_window:
        adapter.advance()

    steps = 0
    total_features = len(adapter._state.features)
    while adapter.cursor < total_features:
        klines = adapter.fetch_klines(
            adapter._state.symbol,
            adapter._state.interval,
            limit=model_config.lookback_window,
        )
        balance = adapter.fetch_balance()
        positions = adapter.fetch_positions(adapter._state.symbol)
        open_orders = adapter.fetch_open_orders(adapter._state.symbol)

        obs = build_live_observation(
            klines=klines, columns=model_config.columns,
            balance=balance, positions=positions, open_orders=open_orders,
            stats=stats, obs_cfg=obs_cfg,
        )
        result = model_runner.predict(obs)

        state = DecoderState(
            close=float(klines[-1].close),
            available_balance=balance.available,
            num_open_orders=len(open_orders),
            num_open_positions=len(positions),
        )
        intent = decode_action(result.action, state, model_config)
        intent = clamp_intent(intent, RiskClampConfig(
            equity=balance.total,
            max_position_size_pct=max_position_size_pct,
            max_leverage=max_leverage,
        ))

        # Replay path: bypass the adapter's place_order and call sim.apply_intent
        # directly — the same code TradingEnv.step uses, ensuring 0% divergence.
        adapter.sim.apply_intent(intent, current_price=state.close)
        adapter.advance()
        steps += 1

    final_balance = adapter.fetch_balance()
    return ReplayResult(final_equity=final_balance.total, total_steps=steps)
