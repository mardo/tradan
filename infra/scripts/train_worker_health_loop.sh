#!/usr/bin/env bash
# Append a periodic "last checked" line so you can tell this monitor (and optionally the worker) is alive.
#
# Env:
#   TRAIN_WORKER_HEALTH_INTERVAL_SEC  sleep between checks (default: 300)
#   TRAIN_WORKER_HEALTH_LOG           log file path (default: /var/log/train_worker_health.log)
#   TRAIN_WORKER_HEALTH_CHECK_PROCESS  if 1 (default), append train_worker=up|down using pgrep
#
# Install (example):
#   sudo cp infra/systemd/tradan-train-worker-health.service /etc/systemd/system/
#   sudo systemctl daemon-reload && sudo systemctl enable --now tradan-train-worker-health.service

set -u

INTERVAL_SEC="${TRAIN_WORKER_HEALTH_INTERVAL_SEC:-300}"
LOG_FILE="${TRAIN_WORKER_HEALTH_LOG:-/var/log/train_worker_health.log}"
CHECK_PROC="${TRAIN_WORKER_HEALTH_CHECK_PROCESS:-1}"

on_sig() {
  # shellcheck disable=SC2034
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "${ts} health_loop: caught signal, exiting" >>"$LOG_FILE" 2>/dev/null || true
  exit 130
}
trap on_sig INT TERM

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ "$CHECK_PROC" == "1" ]]; then
    if pgrep -f 'uv run train worker' >/dev/null 2>&1; then
      suffix=" train_worker=up"
    else
      suffix=" train_worker=down"
    fi
  else
    suffix=""
  fi
  echo "${ts} last checked${suffix}" >>"$LOG_FILE" || echo "${ts} last checked: failed to write ${LOG_FILE}" >&2
  sleep "$INTERVAL_SEC"
done
