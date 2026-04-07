#!/usr/bin/env bash
set -euo pipefail

BACKEND=/opt/tradan/backend
WORKER_COUNT_FILE=/etc/tradan/worker_count
LOCKFILE=/tmp/run_sweep.lock

# Prevent concurrent sweep runs — only one sweep at a time.
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "ERROR: Another sweep is already running (lock held: $LOCKFILE). Aborting."
  echo "       Kill the existing sweep first, then re-run."
  exit 1
fi

cd "$BACKEND"

CPUS=$(nproc)
echo "Starting sweep (batched mode — worker count read from $WORKER_COUNT_FILE each batch, CPUs: $CPUS)..."

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

  # Limit CPU threads per worker to target ~80% CPU utilisation.
  # Using 80% headroom prevents the kernel scheduler from thrashing and leaves
  # room for the OS, DB client, and checkpoint I/O.  Minimum 1 thread.
  THREADS_PER_WORKER=$(( (CPUS * 80 + 99) / 100 / WORKERS ))
  [ "$THREADS_PER_WORKER" -lt 1 ] && THREADS_PER_WORKER=1
  export OMP_NUM_THREADS=$THREADS_PER_WORKER
  export MKL_NUM_THREADS=$THREADS_PER_WORKER

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
