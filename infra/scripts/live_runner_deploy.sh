#!/usr/bin/env bash
# Install systemd unit template and create env files (templates) for s1/s2/s3.
# Idempotent: existing env files are not overwritten.
#
# Run on the tradan server (or any host with /etc/tradan/ writable):
#     bash infra/scripts/live_runner_deploy.sh
#
# After running, edit each /etc/tradan/live-sN.env to fill in real secrets,
# then enable per-pick:  systemctl enable --now tradan-live@s1.service

set -euo pipefail

UNIT_SRC="$(cd "$(dirname "$0")"/.. && pwd)/systemd/tradan-live@.service"
UNIT_DST="/etc/systemd/system/tradan-live@.service"
ENV_DIR="/etc/tradan"

if [[ ! -f "$UNIT_SRC" ]]; then
    echo "error: unit file not found at $UNIT_SRC" >&2
    exit 1
fi

install -m 0644 "$UNIT_SRC" "$UNIT_DST"
install -d -m 0755 "$ENV_DIR"

for instance in s1 s2 s3; do
    env_file="$ENV_DIR/live-${instance}.env"
    if [[ ! -f "$env_file" ]]; then
        upper="${instance^^}"
        cat > "$env_file" <<EOF
# Tradan live runner env for instance ${instance}
BINGX_VST_${upper}_API_KEY=
BINGX_VST_${upper}_API_SECRET=
TRADAN_KILL_SWITCH_${upper}=false
DATABASE_URL=
MODELS_DIR=/var/lib/tradan/models
EOF
        chmod 0600 "$env_file"
        echo "[install] created template ${env_file} (fill in secrets manually)"
    else
        echo "[skip]    ${env_file} exists, leaving as-is"
    fi
done

systemctl daemon-reload
echo "[ok] systemd unit installed; enable individual instances with:"
echo "     systemctl enable --now tradan-live@s1.service"
