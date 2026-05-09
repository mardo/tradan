"""Lazy registry mapping exchange name → adapter class."""
from __future__ import annotations

import importlib

from live.exchange.base import ExchangeAdapter


_REGISTRY: dict[str, str] = {
    "bingx": "live.exchange.bingx.BingXAdapter",
    "replay": "live.exchange.replay.ReplayAdapter",
}


def get_adapter_class(name: str) -> type[ExchangeAdapter]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown exchange adapter: {name!r}")
    dotted = _REGISTRY[name]
    module_path, _, cls_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)
