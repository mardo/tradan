#!/usr/bin/env bash
# Respawn `uv run train worker` whenever the child exits (OOM, SIGKILL, crash, etc.).
# Does not touch DB claims — only supervises the process.
#
# Usage (from repo root or anywhere):
#   bash infra/scripts/train_worker_watchdog.sh
#   bash infra/scripts/train_worker_watchdog.sh --poll-seconds 60 --cpu-usage 85
#
# Optional env:
#   TRAIN_WORKER_RESTART_DELAY_SEC   seconds before restart (default: 5)
#   TRAIN_WORKER_STOP_ON_SUCCESS=1   if train worker exits 0, stop the supervisor too
#                                    (use with --poll-seconds 0 when the queue can empty)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="$ROOT/backend"
cd "$BACKEND"

DELAY="${TRAIN_WORKER_RESTART_DELAY_SEC:-5}"
STOP_OK="${TRAIN_WORKER_STOP_ON_SUCCESS:-0}"

on_sig() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') supervisor: caught signal, exiting"
  exit 130
}
trap on_sig INT TERM

while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') starting: uv run train worker $*"
  set +e
  uv run train worker "$@"
  code=$?
  set -e
  echo "$(date '+%Y-%m-%d %H:%M:%S') train worker exited (code=$code)"

  if [[ "$code" -eq 0 && "$STOP_OK" == "1" ]]; then
    echo "TRAIN_WORKER_STOP_ON_SUCCESS=1 and exit 0 — stopping supervisor."
    exit 0
  fi

  echo "restarting in ${DELAY}s..."
  sleep "$DELAY"
done
