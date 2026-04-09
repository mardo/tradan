#!/usr/bin/env bash
# Minimal host setup to run tradan/backend (Postgres + uv + common build deps).
# Intended for a fresh Debian/Ubuntu server over SSH (same family as infra droplets).
#
# Usage (as root):
#   curl -fsSL https://raw.githubusercontent.com/.../bootstrap-backend-host.sh | bash
#   # or copy this file and:  bash bootstrap-backend-host.sh
#
# Optional: create app DB user/database (omit if you use Neon or already have a DB):
#   export TRADAN_DB_USER=tradan
#   export TRADAN_DB_NAME=tradan
#   export TRADAN_DB_PASSWORD='your-secure-password'
#   bash bootstrap-backend-host.sh
#
# Afterward:
#   export PATH="$HOME/.local/bin:$PATH"
#   git clone <your-repo-url> /opt/tradan   # or any path
#   cd /opt/tradan/backend
#   cp .env.example .env   # set DATABASE_URL
#   uv sync
#   uv run ingest migrate
set -euo pipefail

if [[ "${EUID:-}" -ne 0 ]]; then
  echo "Run as root (e.g. sudo -i)." >&2
  exit 1
fi

if ! command -v apt-get >/dev/null; then
  echo "This script only supports Debian/Ubuntu (apt-get)." >&2
  echo "Install PostgreSQL 14+, git, curl, rsync, then: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  rsync \
  postgresql \
  postgresql-client \
  build-essential

systemctl enable --now postgresql

UV_BIN="${HOME}/.local/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Match infra droplets / Makefile (non-login shells over SSH)
if ! grep -q '\.local/bin' "${HOME}/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
fi

TRADAN_DB_NAME="${TRADAN_DB_NAME:-tradan}"
TRADAN_DB_USER="${TRADAN_DB_USER:-tradan}"

if [[ -n "${TRADAN_DB_PASSWORD:-}" ]]; then
  # Escape single quotes for use inside a SQL string literal ('' per PostgreSQL rules).
  esc_pass="$(printf '%s' "${TRADAN_DB_PASSWORD}" | sed "s/'/''/g")"
  if sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${TRADAN_DB_USER}'" | grep -q 1; then
    echo "Postgres role '${TRADAN_DB_USER}' already exists — skipping CREATE USER."
  else
    sudo -u postgres psql -v ON_ERROR_STOP=1 -c \
      "CREATE USER ${TRADAN_DB_USER} WITH PASSWORD '${esc_pass}';"
  fi
  if sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${TRADAN_DB_NAME}'" | grep -q 1; then
    echo "Database '${TRADAN_DB_NAME}' already exists — skipping CREATE DATABASE."
  else
    sudo -u postgres psql -v ON_ERROR_STOP=1 -c \
      "CREATE DATABASE ${TRADAN_DB_NAME} OWNER ${TRADAN_DB_USER};"
  fi
  echo ""
  echo "Put this in backend/.env:"
  echo "DATABASE_URL=postgresql://${TRADAN_DB_USER}:${TRADAN_DB_PASSWORD}@localhost:5432/${TRADAN_DB_NAME}"
  echo ""
fi

echo "Bootstrap finished."
echo "  uv:  ${UV_BIN}"
echo "  Add to PATH for this session:  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  Then clone the repo, cd backend, uv sync, uv run ingest migrate"
