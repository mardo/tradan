# Symbol Ingest Automation Design

**Date:** 2026-04-06  
**Scope:** Automate kline data ingestion for configured symbols on the `base` droplet, with Makefile commands for manual re-triggering and log tailing.

---

## Goals

1. Terraform `variables.tf` declares which symbols to ingest and their date ranges.
2. On `base-up`, the `base` droplet automatically starts ingesting all configured symbols in the background.
3. A `make init-symbol SYMBOL=BTCUSDT` command lets operators re-trigger ingest for a single symbol.
4. A `make logs` command tails ingest logs on `base` (or a future `train` log) via SSH.
5. After 3 retry attempts, persistent failures are written to a shared error log.

---

## Architecture

### Terraform Variable

`variables.tf` gains a new `symbols` variable:

```hcl
variable "symbols" {
  description = "Symbols to ingest, each with start/end month (YYYY-MM)"
  type = map(object({
    start = string
    end   = string
  }))
  default = {
    BTCUSDT = { start = "2020-01", end = "2026-04" }
  }
}
```

`base.tf` passes `symbols = var.symbols` into the `cloud-init-base.yaml` `templatefile()` call.

### Cloud-Init Changes (`cloud-init-base.yaml`)

After the existing `uv sync` + `uv run ingest migrate` steps, cloud-init:

1. Creates `/var/log/tradan/` directory.
2. Writes a systemd template unit file to `/etc/systemd/system/tradan-ingest@.service`.
3. For each symbol in the `symbols` map:
   - Writes `/etc/tradan/ingest-<SYMBOL>.env` containing `INGEST_START` and `INGEST_END`.
   - Runs `systemctl enable --now tradan-ingest@<SYMBOL>` to start the service in the background.

### Shell Script (`infra/scripts/init-symbol.sh`)

Single script that handles the full ingest pipeline for one symbol. Called by systemd as `init-symbol.sh <SYMBOL>`.

```
SYMBOL=$1
LOG=/var/log/tradan/ingest-$SYMBOL.log
ERR=/var/log/tradan/errors.log

source /opt/tradan/backend/.env
source /etc/tradan/ingest-$SYMBOL.env   # provides INGEST_START, INGEST_END

cd /opt/tradan/backend

1. uv run ingest enqueue --symbol $SYMBOL --start $INGEST_START --end $INGEST_END
2. uv run ingest run --workers 10

3. for attempt in 1 2 3:
     uv run ingest verify --symbol $SYMBOL
     → exit 0 if no gaps
     → uv run ingest fill-gaps --symbol $SYMBOL
     → uv run ingest retry --workers 4

4. Final verify:
   → if gaps remain: log ERROR to $ERR, exit 1
   → else: log OK, exit 0
```

The verify step detects gaps by checking `ingest verify` stdout for the string `"gap(s) found"` (the CLI currently exits 0 regardless; the script captures output and branches on that string). As part of implementation, `cmd_verify` in `backend/src/ingester/cli.py` will be updated to `sys.exit(1)` when gaps are found so the script can use `$?` directly.

### Systemd Template Unit (`tradan-ingest@.service`)

Written to `/etc/systemd/system/tradan-ingest@.service` by cloud-init:

```ini
[Unit]
Description=Tradan kline ingest for %i
After=network-online.target postgresql.service

[Service]
Type=oneshot
RemainAfterExit=yes
EnvironmentFile=/etc/tradan/ingest-%i.env
ExecStart=/opt/tradan/infra/scripts/init-symbol.sh %i
StandardOutput=append:/var/log/tradan/ingest-%i.log
StandardError=append:/var/log/tradan/ingest-%i.log

[Install]
WantedBy=multi-user.target
```

`Type=oneshot` + `RemainAfterExit=yes` means `systemctl status tradan-ingest@BTCUSDT` shows whether the pipeline completed successfully or failed.

### Makefile Additions

```makefile
## init-symbol: Re-trigger ingest for a symbol on the base droplet
## Usage: make init-symbol SYMBOL=BTCUSDT
init-symbol:
    @if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
    @if [ -z "$(SYMBOL)" ]; then echo "Usage: make init-symbol SYMBOL=BTCUSDT"; exit 1; fi
    ssh $(SSH_OPTS) root@$(BASE_IP) "systemctl restart tradan-ingest@$(SYMBOL)"

## logs: Tail logs on base or train droplet
## Usage: make logs                          (all base ingest + errors)
##        make logs SYMBOL=BTCUSDT           (specific symbol)
##        make logs TARGET=train             (train droplet)
TARGET ?= base
logs:
    @if [ "$(TARGET)" = "train" ]; then \
        if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet not running."; exit 1; fi; \
        ssh $(SSH_OPTS) root@$(TRAIN_IP) "tail -f /var/log/tradan/train.log 2>/dev/null || journalctl -f"; \
    else \
        if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi; \
        if [ -n "$(SYMBOL)" ]; then \
            ssh $(SSH_OPTS) root@$(BASE_IP) "tail -f /var/log/tradan/ingest-$(SYMBOL).log"; \
        else \
            ssh $(SSH_OPTS) root@$(BASE_IP) "tail -f /var/log/tradan/ingest-*.log /var/log/tradan/errors.log"; \
        fi; \
    fi
```

---

## Log Files on Base Droplet

| Path | Contents |
|------|----------|
| `/var/log/tradan/ingest-BTCUSDT.log` | Full stdout+stderr for BTCUSDT pipeline |
| `/var/log/tradan/ingest-ETHUSDT.log` | Full stdout+stderr for ETHUSDT pipeline |
| `/var/log/tradan/errors.log` | Symbols that failed all 3 retry attempts |

---

## Error Handling

- The script exits 0 on clean completion, 1 on permanent failure.
- `systemctl status tradan-ingest@BTCUSDT` reflects the exit code.
- `/var/log/tradan/errors.log` is a human-readable record of permanent failures with timestamps.
- `make logs` (no SYMBOL) tails all ingest logs + errors.log in one stream so the operator sees everything.

---

## Files Changed / Created

| File | Change |
|------|--------|
| `infra/variables.tf` | Add `symbols` variable |
| `infra/base.tf` | Pass `symbols` to cloud-init templatefile |
| `infra/scripts/cloud-init-base.yaml` | Add log dir, systemd unit, per-symbol env files, service start |
| `infra/scripts/init-symbol.sh` | New — full ingest pipeline script |
| `infra/Makefile` | Add `init-symbol` and `logs` targets, update `.PHONY` |
| `infra/.env.example` | Document BTCUSDT as example in `TF_VAR_symbols` |

---

## Out of Scope

- Automatic re-ingestion on schedule (cron). This is a one-time init.
- Train droplet log infrastructure (placeholder hook added to `make logs TARGET=train` only).
- Alerting beyond writing to `errors.log`.
