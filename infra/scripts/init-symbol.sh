#!/usr/bin/env bash
# init-symbol.sh <SYMBOL>
# Full kline ingest pipeline for one symbol. Run by systemd tradan-ingest@.service.
# Logs go to /var/log/tradan/ingest-<SYMBOL>.log (captured by systemd).
# Persistent failures appended to /var/log/tradan/errors.log.
set -euo pipefail

SYMBOL="${1:?Usage: init-symbol.sh SYMBOL}"
if [[ ! "$SYMBOL" =~ ^[A-Z0-9]+$ ]]; then
  echo "ERROR: SYMBOL must contain only uppercase letters and digits, got: $SYMBOL" >&2
  exit 1
fi
LOG_DIR="/var/log/tradan"
ERR_LOG="$LOG_DIR/errors.log"
ENV_FILE="/etc/tradan/ingest-${SYMBOL}.env"
BACKEND_DIR="/opt/tradan/backend"
UV="/root/.local/bin/uv"
if [ ! -x "$UV" ]; then
  echo "ERROR: uv not found at $UV" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

# Source DATABASE_URL
if [ ! -f "/opt/tradan/backend/.env" ]; then
  echo "ERROR: backend env file /opt/tradan/backend/.env not found" >&2
  exit 1
fi
# shellcheck source=/dev/null
source /opt/tradan/backend/.env

# Source per-symbol INGEST_START and INGEST_END
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: env file $ENV_FILE not found" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting ingest for $SYMBOL ($INGEST_START → $INGEST_END)"

cd "$BACKEND_DIR"

# Step 1: Enqueue jobs
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Enqueueing jobs..."
$UV run ingest enqueue --symbol "$SYMBOL" --start "$INGEST_START" --end "$INGEST_END"

# Step 2: Run workers
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running ingest workers (10)..."
$UV run ingest run --workers 10

# Step 3: Verify + retry loop (up to 3 attempts)
MAX_RETRIES=3
for attempt in $(seq 1 $MAX_RETRIES); do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Verify attempt $attempt/$MAX_RETRIES..."
  if $UV run ingest verify --symbol "$SYMBOL"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $SYMBOL: all data contiguous. Done."
    exit 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Gaps found. Running fill-gaps + retry (attempt $attempt)..."
  $UV run ingest fill-gaps --symbol "$SYMBOL"
  $UV run ingest retry --workers 4
done

# Step 4: Final verify after all retries
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Final verify after $MAX_RETRIES retries..."
if $UV run ingest verify --symbol "$SYMBOL"; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $SYMBOL: all data contiguous after retries. Done."
  exit 0
fi

# Permanent failure
MSG="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: $SYMBOL still has gaps after $MAX_RETRIES retries. Manual intervention required."
echo "$MSG"
echo "$MSG" >> "$ERR_LOG"
exit 1
