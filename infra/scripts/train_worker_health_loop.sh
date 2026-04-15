#!/usr/bin/env bash
# Periodically logs "last checked" and ensures train_worker_watchdog.sh is running
# (starts it with nohup if the supervisor process is missing).
#
# Env:
#   TRAIN_WORKER_HEALTH_INTERVAL_SEC     sleep between checks (default: 300)
#   TRAIN_WORKER_HEALTH_LOG              this script's log (default: /var/log/train_worker_health.log)
#   TRAIN_WORKER_HEALTH_CHECK_PROCESS    if 1 (default), log train_worker=up|down
#   TRAIN_WORKER_SUPERVISE_RESTART       if 1 (default), start watchdog when absent
#   TRAIN_WORKER_WATCHDOG_LOG            stdout/stderr for watchdog+nohup (default: /var/log/train_worker_watchdog.log)
#   TRAIN_WORKER_WATCHDOG_ARGS           optional extra args for watchdog / `uv run train worker` (word-split on spaces)
#
# Install (example):
#   sudo cp infra/systemd/tradan-train-worker-health.service /etc/systemd/system/
#   sudo systemctl daemon-reload && sudo systemctl enable --now tradan-train-worker-health.service

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WATCHDOG_SCRIPT="$ROOT/infra/scripts/train_worker_watchdog.sh"

INTERVAL_SEC="${TRAIN_WORKER_HEALTH_INTERVAL_SEC:-300}"
LOG_FILE="${TRAIN_WORKER_HEALTH_LOG:-/var/log/train_worker_health.log}"
CHECK_PROC="${TRAIN_WORKER_HEALTH_CHECK_PROCESS:-1}"
SUPERVISE="${TRAIN_WORKER_SUPERVISE_RESTART:-1}"
WATCHDOG_LOG="${TRAIN_WORKER_WATCHDOG_LOG:-/var/log/train_worker_watchdog.log}"
# shellcheck disable=SC2206
WATCHDOG_EXTRA=( ${TRAIN_WORKER_WATCHDOG_ARGS-} )

on_sig() {
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "${ts} health_loop: caught signal, exiting" >>"$LOG_FILE" 2>/dev/null || true
  exit 130
}
trap on_sig INT TERM

watchdog_running() {
  pgrep -f 'train_worker_watchdog\.sh' >/dev/null 2>&1
}

worker_running() {
  pgrep -f 'uv run train worker' >/dev/null 2>&1
}

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  suffix=""
  action="none"

  if watchdog_running; then
    wd="watchdog=up"
  else
    wd="watchdog=down"
  fi

  if [[ "$CHECK_PROC" == "1" ]]; then
    if worker_running; then
      suffix=" train_worker=up"
    else
      suffix=" train_worker=down"
    fi
  fi

  if [[ "$SUPERVISE" == "1" ]] && ! watchdog_running; then
    nohup bash "$WATCHDOG_SCRIPT" "${WATCHDOG_EXTRA[@]}" >>"$WATCHDOG_LOG" 2>&1 &
    action="start_watchdog"
    sleep 1
    if watchdog_running; then
      wd="watchdog=up"
    else
      wd="watchdog=down"
    fi
  fi

  echo "${ts} last checked ${wd}${suffix} action=${action}" >>"$LOG_FILE" ||
    echo "${ts} last checked: failed to write ${LOG_FILE}" >&2

  sleep "$INTERVAL_SEC"
done
