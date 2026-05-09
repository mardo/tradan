"""Live action decoder: trainer pure decoder + per-run risk clamps."""
from __future__ import annotations

from dataclasses import dataclass, replace

from trainer.env.action_decoder import OrderIntent


@dataclass(frozen=True)
class RiskClampConfig:
    equity: float                 # current account equity
    max_position_size_pct: float  # max margin as fraction of equity
    max_leverage: float           # cap on per-order leverage (informational here)


def clamp_intent(intent: OrderIntent, cfg: RiskClampConfig) -> OrderIntent:
    if intent.open is None:
        return intent
    cap = cfg.equity * cfg.max_position_size_pct
    margin = min(intent.open.margin, cap)
    new_open = replace(intent.open, margin=margin)
    return replace(intent, open=new_open)
