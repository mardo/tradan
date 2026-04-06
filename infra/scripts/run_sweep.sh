#!/usr/bin/env bash
set -euo pipefail

WORKERS=$(cat /etc/tradan/worker_count)
BACKEND=/opt/tradan/backend

echo "Starting sweep with $WORKERS parallel workers..."

cd "$BACKEND"
/root/.local/bin/uv run train list --names-only --status pending | tee /tmp/pending_models.txt
PENDING=$(wc -l < /tmp/pending_models.txt)
echo "Total pending: $PENDING"

if [ "$PENDING" -eq 0 ]; then
  echo "No pending models found. Register configs first (sweep_phase1.py etc.)"
  exit 0
fi

cat /tmp/pending_models.txt \
  | parallel --jobs "$WORKERS" --joblog /tmp/sweep_joblog.txt \
      "cd $BACKEND && /root/.local/bin/uv run train start --model {}"

echo ""
echo "Sweep complete. Check results: uv run train list"
echo "Job log: /tmp/sweep_joblog.txt"
