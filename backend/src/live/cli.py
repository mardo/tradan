"""Live runner CLI. Full implementation in Phase F."""
from __future__ import annotations

import sys


def main() -> int:
    print("live-test: not yet implemented (Phase F)", file=sys.stderr)
    return 2


def replay_main() -> int:
    from live.replay_cli import replay_main as _replay
    return _replay()


if __name__ == "__main__":
    sys.exit(main())
