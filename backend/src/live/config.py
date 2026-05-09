"""Pydantic schemas for live runner YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ExchangeCfg(BaseModel):
    name: Literal["bingx"]
    mode: Literal["demo", "paper", "live"]
    api_key_env: str
    api_secret_env: str


class MarketCfg(BaseModel):
    symbol: str
    interval: str

    @field_validator("interval")
    @classmethod
    def _interval_known(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
        if v not in allowed:
            raise ValueError(f"interval {v!r} not in {sorted(allowed)}")
        return v


class ModelCfg(BaseModel):
    name: str


class RiskCfg(BaseModel):
    starting_equity_quote: float = Field(gt=0)
    max_drawdown_pct: float = Field(ge=0.0, le=1.0)
    max_position_size_pct: float = Field(gt=0.0, le=1.0)
    max_leverage: float = Field(gt=0)
    kill_switch_env: str


class LoggingCfg(BaseModel):
    pnl_snapshot_interval_minutes: int = Field(ge=1)


class LiveConfig(BaseModel):
    exchange: ExchangeCfg
    market: MarketCfg
    model: ModelCfg
    risk: RiskCfg
    logging: LoggingCfg

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LiveConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
