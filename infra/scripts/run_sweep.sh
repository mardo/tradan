#!/usr/bin/env bash
set -euo pipefail

BACKEND=/opt/tradan/backend
WORKER_COUNT_FILE=/etc/tradan/worker_count

cd "$BACKEND"

echo "Starting sweep (batched mode — worker count read from $WORKER_COUNT_FILE each batch)..."

/root/.local/bin/uv run train list --names-only --status pending > /tmp/pending_models.txt
PENDING=$(wc -l < /tmp/pending_models.txt | tr -d ' ')
echo "Total pending: $PENDING"

if [ "$PENDING" -eq 0 ]; then
  echo "No pending models found. Register configs first (sweep_phase1.py etc.)"
  exit 0
fi

PROCESSED=0
BATCH_NUM=0

while [ "$PROCESSED" -lt "$PENDING" ]; do
  # Re-read worker count before each batch so live adjustments take effect
  WORKERS=$(cat "$WORKER_COUNT_FILE")

  # Slice the next batch of model names from the pending list
  mapfile -t BATCH < <(tail -n +"$((PROCESSED + 1))" /tmp/pending_models.txt | head -n "$WORKERS")

  if [ "${#BATCH[@]}" -eq 0 ]; then
    break
  fi

  BATCH_NUM=$((BATCH_NUM + 1))
  echo ""
  echo "=== Batch $BATCH_NUM: launching ${#BATCH[@]} workers (workers cap: $WORKERS) ==="

  PIDS=()
  for model in "${BATCH[@]}"; do
    /root/.local/bin/uv run train start --model "$model" &
    PIDS+=($!)
    echo "  + started: $model (pid $!)"
  done

  # Wait for EVERY worker in this batch before moving on
  BATCH_FAILED=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || BATCH_FAILED=$((BATCH_FAILED + 1))
  done

  PROCESSED=$((PROCESSED + ${#BATCH[@]}))
  echo "=== Batch $BATCH_NUM done — $BATCH_FAILED failed | processed $PROCESSED / $PENDING ==="
done

echo ""
echo "Sweep complete. Check results: uv run train list"
