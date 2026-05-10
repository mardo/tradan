from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExchangeConfig:
    """Simulated exchange parameters: fees, leverage rules, and position limits."""

    # Fee charged on limit order fills (entry, TP). Percentage of notional value.
    maker_fee_pct: float = 0.02
    # Fee charged on market-like fills (SL, manual close, liquidation). Percentage of notional.
    taker_fee_pct: float = 0.04
    # Fixed dollar fee added on top of percentage fee per trade.
    flat_fee_usd: float = 0.0

    # Absolute maximum leverage. Lowered from 125 to 10 in the Phase 4 env audit
    # (docs/plans/2026-05-10-phase4-env-audit-design.md) — tight-SL gaming auto-leveraged
    # policies to ~100x and was the proximate cause of >100% holdout drawdowns.
    # Saved configs from before the audit retain their old (125) default via the
    # JSON snapshot in model_configs.config_json.
    max_leverage: float = 10.0
    # Safety margin between the SL price and the liquidation price, as % of entry price.
    # Ensures the SL triggers before liquidation even with slippage.
    liquidation_buffer_pct: float = 0.5
    # Maintenance margin rate. Used in the liquidation price formula to determine
    # at what price the exchange force-closes the position.
    maintenance_margin_pct: float = 0.4

    # Maximum number of unfilled limit orders the model can have at once.
    max_open_orders: int = 20
    # Maximum number of open positions (filled trades) the model can hold at once.
    max_open_positions: int = 20
    # Minimum notional value (margin × leverage) for an order to be accepted.
    min_order_size_usd: float = 10.0
    # Maximum fraction of available balance committed to a single position. Added
    # in the Phase 4 env audit (Fix 2): the prior behavior allowed margin =
    # size_raw * available_balance, i.e. a model could put 100% on one trade.
    # Combined with high leverage, that wiped accounts on a single move. The cap
    # is enforced in TradingEnv._process_actions before place_order.
    max_position_size_pct: float = 0.25
    # Maximum trailing drawdown from peak equity before the episode terminates.
    # Added in the Phase 4 env audit (Fix 3): the prior env terminated only at
    # equity <= 0, giving the policy no early signal that a strategy was failing.
    # Enforced in TradingEnv.step() against a peak_equity tracker.
    max_drawdown_pct: float = 0.5
    # Per-step penalty (USD) subtracted from the Δ-equity reward when the env
    # is idle — no open positions and no live orders. Default 0 preserves the
    # baseline reward exactly. Phase 4E uses positive values (e.g. 0.05, 0.5)
    # to discourage policy collapse to "do nothing", which has been the
    # recurring lb100 failure mode (entropy decay → flat reward between idle
    # and small-loss states). Enforced in TradingEnv.step().
    idle_step_penalty_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "maker_fee_pct": self.maker_fee_pct,
            "taker_fee_pct": self.taker_fee_pct,
            "flat_fee_usd": self.flat_fee_usd,
            "max_leverage": self.max_leverage,
            "liquidation_buffer_pct": self.liquidation_buffer_pct,
            "maintenance_margin_pct": self.maintenance_margin_pct,
            "max_open_orders": self.max_open_orders,
            "max_open_positions": self.max_open_positions,
            "min_order_size_usd": self.min_order_size_usd,
            "max_position_size_pct": self.max_position_size_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "idle_step_penalty_usd": self.idle_step_penalty_usd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExchangeConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


ALL_KLINE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
]


@dataclass
class ModelConfig:
    """Full configuration for a trading model: data sources, exchange params, and training settings."""

    # Unique identifier for this model configuration (e.g. "btc_v1", "sol_v1").
    name: str
    # Which trading pairs to feed as input. Single (["BTCUSDT"]) or multi-symbol (["BTCUSDT", "SOLUSDT"]).
    # Multi-symbol models receive all symbols' data concatenated per candle as features.
    symbols: list[str]
    # Kline intervals to use (e.g. ["1h"], ["1h", "4h"]). The environment steps through the finest interval.
    intervals: list[str]
    # Which kline columns to include as features per symbol. Defaults to all 9 available columns.
    columns: list[str] = field(default_factory=lambda: list(ALL_KLINE_COLUMNS))
    # Exchange simulation parameters (fees, leverage, limits).
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)

    # How many past candles the model sees as input on each step.
    # 500 candles of 1h data = ~20 days of market history.
    lookback_window: int = 500
    # Starting virtual account balance in USD for training and evaluation.
    initial_balance: float = 10_000.0
    # Number of take-profit levels per order. Each TP closes a fraction of the position.
    num_tp_levels: int = 3

    # --- Action space scaling ---
    # Max % the trigger price can deviate from current close. Model output [-1,1] maps to ±this%.
    max_trigger_offset_pct: float = 5.0
    # Minimum stop-loss distance from trigger as %. Model output 0 maps to this.
    min_sl_pct: float = 0.1
    # Maximum stop-loss distance from trigger as %. Model output 1 maps to this.
    max_sl_pct: float = 10.0
    # Maximum take-profit distance from trigger as %. Model output 1 maps to this.
    max_tp_pct: float = 20.0

    # --- Training hyperparameters ---
    # RL algorithm to use. Options: "PPO" (default, stable), "SAC" (off-policy), "A2C" (faster but less stable).
    algorithm: str = "PPO"
    # Total number of environment steps the RL agent takes during training.
    # Higher = more learning but longer training time. 1M steps ≈ 1M candles processed.
    total_timesteps: int = 1_000_000
    # Neural network learning rate. Controls how much weights update per gradient step.
    # Lower = more stable but slower convergence. Higher = faster but risk of instability.
    learning_rate: float = 3e-4
    # Optional integer seed. Forwarded to the SB3 algorithm constructor, which seeds
    # Python/NumPy/Torch globals and the env SB3 owns. None preserves prior
    # non-deterministic behavior. Same-seed reruns on GPU are statistically very close
    # but not bit-identical (CUDA non-determinism is not disabled here); the field still
    # gives a useful handle for seed-variance studies.
    seed: int | None = None
    # Coefficient on the entropy bonus added to the policy-gradient loss. SB3's
    # default is 0.0 for A2C/PPO and 'auto' (a learnable schedule) for SAC; this
    # field is conditionally forwarded by train_model only when > 0, so the
    # default 0.0 preserves SB3 defaults across all algorithms. Set positive
    # (e.g. 0.01) to prevent A2C entropy collapse — see Phase 4C study.
    ent_coef: float = 0.0

    # How often (in env steps) to record a PnL snapshot to the database for tracking performance.
    snapshot_interval: int = 100

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbols": self.symbols,
            "intervals": self.intervals,
            "columns": self.columns,
            "exchange": self.exchange.to_dict(),
            "lookback_window": self.lookback_window,
            "initial_balance": self.initial_balance,
            "num_tp_levels": self.num_tp_levels,
            "max_trigger_offset_pct": self.max_trigger_offset_pct,
            "min_sl_pct": self.min_sl_pct,
            "max_sl_pct": self.max_sl_pct,
            "max_tp_pct": self.max_tp_pct,
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "ent_coef": self.ent_coef,
            "snapshot_interval": self.snapshot_interval,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        d = dict(d)
        if "exchange" in d and isinstance(d["exchange"], dict):
            d["exchange"] = ExchangeConfig.from_dict(d["exchange"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def num_features_per_candle(self) -> int:
        return len(self.symbols) * len(self.columns) * len(self.intervals)

    @property
    def action_size(self) -> int:
        base = 1 + 1 + 1 + 1 + self.num_tp_levels + self.num_tp_levels + 1
        return base + self.exchange.max_open_orders + self.exchange.max_open_positions
