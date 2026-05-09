from __future__ import annotations

import pytest

from live.exchange.base import ExchangeAdapter
from live.exchange.registry import get_adapter_class


def test_known_adapter_returned():
    cls = get_adapter_class("bingx")
    assert issubclass(cls, ExchangeAdapter)


def test_replay_adapter_returned():
    cls = get_adapter_class("replay")
    assert issubclass(cls, ExchangeAdapter)


def test_unknown_adapter_raises():
    with pytest.raises(KeyError):
        get_adapter_class("nope")
