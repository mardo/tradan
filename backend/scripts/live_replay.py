"""Shim for the live-replay console script. Real entry point: live.replay_cli.replay_main"""
from __future__ import annotations

import sys

from live.replay_cli import replay_main


if __name__ == "__main__":
    sys.exit(replay_main())
