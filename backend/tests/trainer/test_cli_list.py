from __future__ import annotations

import io
from unittest.mock import patch

from trainer.cli import build_parser, cmd_list


def _run_list(args: list[str]) -> str:
    """Run `train list <args>`, capture stdout, return output string."""
    parser = build_parser()
    parsed = parser.parse_args(["list"] + args)
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        cmd_list(parsed)
    return buf.getvalue()


def test_names_only_returns_one_name_per_line():
    fake = [
        {"name": "btc_1h_ppo_p1_s0", "created_at": None, "run_count": 0, "best_pnl": None},
        {"name": "btc_4h_sac_p1_s1", "created_at": None, "run_count": 1, "best_pnl": 100.0},
    ]
    with patch("trainer.cli.list_model_configs", return_value=fake):
        output = _run_list(["--names-only"])
    assert output.strip().splitlines() == ["btc_1h_ppo_p1_s0", "btc_4h_sac_p1_s1"]


def test_status_pending_shows_only_zero_run_count():
    fake = [
        {"name": "btc_1h_ppo_p1_s0", "created_at": None, "run_count": 0, "best_pnl": None},
        {"name": "btc_4h_sac_p1_s1", "created_at": None, "run_count": 1, "best_pnl": 100.0},
    ]
    with patch("trainer.cli.list_model_configs", return_value=fake):
        output = _run_list(["--names-only", "--status", "pending"])
    assert output.strip().splitlines() == ["btc_1h_ppo_p1_s0"]


def test_status_completed_shows_only_nonzero_run_count():
    fake = [
        {"name": "btc_1h_ppo_p1_s0", "created_at": None, "run_count": 0, "best_pnl": None},
        {"name": "btc_4h_sac_p1_s1", "created_at": None, "run_count": 1, "best_pnl": 100.0},
    ]
    with patch("trainer.cli.list_model_configs", return_value=fake):
        output = _run_list(["--names-only", "--status", "completed"])
    assert output.strip().splitlines() == ["btc_4h_sac_p1_s1"]


def test_default_list_still_prints_table_header():
    fake = [
        {"name": "btc_1h_ppo_p1_s0", "created_at": None, "run_count": 0, "best_pnl": None},
    ]
    with patch("trainer.cli.list_model_configs", return_value=fake):
        output = _run_list([])
    assert "Name" in output
    assert "btc_1h_ppo_p1_s0" in output


def test_names_only_empty_result_prints_nothing():
    with patch("trainer.cli.list_model_configs", return_value=[]):
        output = _run_list(["--names-only"])
    assert output.strip() == ""
