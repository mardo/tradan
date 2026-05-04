#!/usr/bin/env bash
set -euo pipefail

BACKEND=/opt/tradan/backend
SQL="$BACKEND/scripts/winners_no_eval.sql"
JOBS=4

cd "$BACKEND"
# shellcheck source=/dev/null
set -a; source "$BACKEND/.env"; set +a

echo "Finding top 20 trained models without evaluation runs..."

psql "$DATABASE_URL" \
  --tuples-only \
  --no-align \
  --field-separator $'\t' \
  -f "$SQL" > /tmp/eval_queue.txt

QUEUED=$(grep -c . /tmp/eval_queue.txt 2>/dev/null || echo 0)
echo "Queued for evaluation: $QUEUED"

if [ "$QUEUED" -eq 0 ]; then
  echo "Nothing to evaluate."
  exit 0
fi

active=0
while IFS=$'\t' read -r model run_id; do
  (
    cd "$BACKEND"
    /root/.local/bin/uv run train evaluate --model "$model" --run "$run_id"
  ) &
  active=$((active + 1))
  if [ "$active" -ge "$JOBS" ]; then
    wait
    active=0
  fi
done < /tmp/eval_queue.txt
wait

echo ""
echo "Evaluation complete. Run: psql \$DATABASE_URL -f $BACKEND/scripts/winners.sql"
