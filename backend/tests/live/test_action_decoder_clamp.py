from __future__ import annotations

from live.action_decoder import RiskClampConfig, clamp_intent
from trainer.env.action_decoder import OpenIntent, OrderIntent


def test_open_margin_clamped_to_max_position_pct():
    intent = OrderIntent(
        open=OpenIntent(
            direction=1, trigger_price=100.0, sl_price=99.0,
            tp_prices=[101.0, 102.0, 103.0],
            tp_size_pcts=[1/3, 1/3, 1/3], margin=8_000.0,
        ),
        cancels=[], closes=[],
    )
    cfg = RiskClampConfig(
        equity=10_000.0, max_position_size_pct=0.50, max_leverage=3.0,
    )
    clamped = clamp_intent(intent, cfg)
    assert clamped.open is not None
    assert clamped.open.margin == 5_000.0


def test_no_open_unaffected():
    intent = OrderIntent(open=None, cancels=[1], closes=[])
    cfg = RiskClampConfig(
        equity=10_000.0, max_position_size_pct=0.50, max_leverage=3.0,
    )
    assert clamp_intent(intent, cfg) == intent
