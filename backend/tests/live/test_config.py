from __future__ import annotations

from pathlib import Path

import pytest

from live.config import LiveConfig


VALID = """
exchange:
  name: bingx
  mode: demo
  api_key_env: BINGX_VST_S1_API_KEY
  api_secret_env: BINGX_VST_S1_API_SECRET
market:
  symbol: BTC/USDT:USDT
  interval: 4h
model:
  name: btc_4h_a2c_lb500_3em4_p2_s1
risk:
  starting_equity_quote: 10000
  max_drawdown_pct: 0.20
  max_position_size_pct: 0.50
  max_leverage: 3
  kill_switch_env: TRADAN_KILL_SWITCH_S1
logging:
  pnl_snapshot_interval_minutes: 60
"""


def test_valid_config_loads(tmp_path: Path):
    p = tmp_path / "live-s1.yaml"
    p.write_text(VALID)
    cfg = LiveConfig.from_yaml(p)
    assert cfg.exchange.name == "bingx"
    assert cfg.exchange.mode == "demo"
    assert cfg.market.symbol == "BTC/USDT:USDT"
    assert cfg.risk.max_drawdown_pct == 0.20


def test_invalid_mode_rejected(tmp_path: Path):
    bad = VALID.replace("mode: demo", "mode: production")
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        LiveConfig.from_yaml(p)


def test_invalid_drawdown_rejected(tmp_path: Path):
    bad = VALID.replace("max_drawdown_pct: 0.20", "max_drawdown_pct: 1.5")
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        LiveConfig.from_yaml(p)
