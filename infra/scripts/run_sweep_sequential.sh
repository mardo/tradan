#!/usr/bin/env bash
set -euo pipefail

BACKEND=/opt/tradan/backend
LOCKFILE=/tmp/run_sweep.lock

# Prevent concurrent sweep runs — only one sweep at a time.
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "ERROR: Another sweep is already running (lock held: $LOCKFILE). Aborting."
  echo "       Kill the existing sweep first, then re-run."
  exit 1
fi

cd "$BACKEND"

# Let PyTorch use all available cores — no artificial thread cap.
export TRADAN_FULL_THREADS=1

echo "Starting sequential sweep (one model at a time, full CPU per run)..."

/root/.local/bin/uv run train list --names-only --status pending > /tmp/pending_models.txt
PENDING=$(wc -l < /tmp/pending_models.txt | tr -d ' ')
echo "Total pending: $PENDING"

if [ "$PENDING" -eq 0 ]; then
  echo "No pending models found. Register configs first (sweep_phase1.py etc.)"
  exit 0
fi

PROCESSED=0
FAILED=0

while IFS= read -r model; do
  PROCESSED=$((PROCESSED + 1))
  echo ""
  echo "=== [$PROCESSED/$PENDING] Training: $model ==="

  if /root/.local/bin/uv run train start --model "$model"; then
    echo "=== [$PROCESSED/$PENDING] Done: $model ==="
  else
    FAILED=$((FAILED + 1))
    echo "=== [$PROCESSED/$PENDING] FAILED: $model ==="
  fi
done < /tmp/pending_models.txt

echo ""
echo "Sequential sweep complete — $FAILED failed / $PENDING total. Check results: uv run train list"
