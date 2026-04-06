#!/usr/bin/env bash
set -euo pipefail

BACKEND=/opt/tradan/backend
SQL=/opt/tradan/infra/scripts/winners_no_eval.sql

echo "Finding top 20 trained models without evaluation runs..."
cd "$BACKEND"

# Query returns: model_name<TAB>run_id per line
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

# Run 4 parallel evals on the base droplet (4 vCPUs available)
cat /tmp/eval_queue.txt \
  | parallel --jobs 4 --colsep $'\t' \
      "cd $BACKEND && /root/.local/bin/uv run train evaluate --model {1} --run {2}"

echo ""
echo "Evaluation complete. Run: psql \$DATABASE_URL -f /opt/tradan/infra/scripts/winners.sql"
