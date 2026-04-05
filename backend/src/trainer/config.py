from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExchangeConfig:
    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.04
    flat_fee_usd: float = 0.0

    max_leverage: float = 125.0
    liquidation_buffer_pct: float = 0.5
    maintenance_margin_pct: float = 0.4

    max_open_orders: int = 20
    max_open_positions: int = 20
    min_order_size_usd: float = 10.0

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
    name: str
    symbols: list[str]
    intervals: list[str]
    columns: list[str] = field(default_factory=lambda: list(ALL_KLINE_COLUMNS))
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)

    lookback_window: int = 500
    initial_balance: float = 10_000.0
    num_tp_levels: int = 3

    max_trigger_offset_pct: float = 5.0
    min_sl_pct: float = 0.1
    max_sl_pct: float = 10.0
    max_tp_pct: float = 20.0

    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4

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
