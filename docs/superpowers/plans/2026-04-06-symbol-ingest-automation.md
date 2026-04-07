# Symbol Ingest Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate kline data ingestion for configured symbols on first boot of the base droplet, with retry logic, error logging, and Makefile commands for manual re-triggering and log tailing.

**Architecture:** A new `infra/scripts/init-symbol.sh` shell script runs the full ingest pipeline (enqueue → run → verify → retry loop) for one symbol. Cloud-init writes a systemd template unit that starts one service instance per configured symbol in the background. The `symbols` Terraform variable (map of symbol → start/end dates) drives which services get started.

**Tech Stack:** Terraform (DigitalOcean provider), cloud-init (YAML), bash, systemd, Python/uv (`ingest` CLI)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `infra/variables.tf` | Modify | Add `symbols` variable |
| `infra/base.tf` | Modify | Pass `symbols` to cloud-init templatefile |
| `infra/scripts/cloud-init-base.yaml` | Modify | Write systemd unit, per-symbol env files, start services |
| `infra/scripts/init-symbol.sh` | Create | Full ingest pipeline for one symbol with retry logic |
| `infra/Makefile` | Modify | Add `init-symbol`, `logs` targets; update `.PHONY` |
| `infra/.env.example` | Modify | Document `TF_VAR_symbols` |
| `infra/README.md` | Modify | Document `symbols` variable, `make init-symbol`, `make logs` |
| `backend/src/ingester/cli.py` | Modify | Make `cmd_verify` exit 1 when gaps are found |

---

## Task 1: Make `ingest verify` exit 1 on gaps

**Files:**
- Modify: `backend/src/ingester/cli.py` (around line 278)

The `init-symbol.sh` script relies on a non-zero exit code from `ingest verify` to detect gaps. Currently `cmd_verify` always exits 0. Fix that first so the script can use `$?`.

- [ ] **Step 1: Open `backend/src/ingester/cli.py` and locate `cmd_verify`**

Find the block at the end of `cmd_verify` (around line 277):

```python
        if total_gaps:
            print(f"Result: {total_gaps} gap(s) found across {total_checked} checked series.")
        else:
            print(f"Result: all {total_checked} checked series are contiguous.")
```

- [ ] **Step 2: Add `sys.exit(1)` when gaps exist**

Replace that block with:

```python
        if total_gaps:
            print(f"Result: {total_gaps} gap(s) found across {total_checked} checked series.")
            sys.exit(1)
        else:
            print(f"Result: all {total_checked} checked series are contiguous.")
```

Also add `import sys` at the top of the file if it isn't already imported. Check line 1–20; `sys` is already imported via `import os` area — search for `import sys`. If missing, add it after `import os`.

- [ ] **Step 3: Verify the change manually**

```bash
cd backend
grep -n "sys.exit" src/ingester/cli.py
```

Expected output: one line showing `sys.exit(1)` in `cmd_verify`.

- [ ] **Step 4: Commit**

```bash
git add backend/src/ingester/cli.py
git commit -m "fix: ingest verify exits 1 when gaps are found"
```

---

## Task 2: Add `symbols` Terraform variable

**Files:**
- Modify: `infra/variables.tf` (append after line 70)
- Modify: `infra/.env.example`

- [ ] **Step 1: Add variable to `infra/variables.tf`**

Append after the last variable (`github_token`):

```hcl
variable "symbols" {
  description = "Symbols to ingest, keyed by symbol name. Each entry specifies the start and end month (YYYY-MM) for kline data."
  type = map(object({
    start = string
    end   = string
  }))
  default = {
    BTCUSDT = { start = "2020-01", end = "2026-04" }
  }
}
```

- [ ] **Step 2: Document in `.env.example`**

Append to `infra/.env.example`:

```bash
# Symbols to ingest (Terraform map as JSON). Add more symbols as needed.
# TF_VAR_symbols='{"BTCUSDT":{"start":"2020-01","end":"2026-04"},"ETHUSDT":{"start":"2021-06","end":"2026-04"}}'
```

- [ ] **Step 3: Validate Terraform syntax**

```bash
cd infra
terraform validate
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 4: Commit**

```bash
git add infra/variables.tf infra/.env.example
git commit -m "feat: add symbols terraform variable with per-symbol date ranges"
```

---

## Task 3: Create `infra/scripts/init-symbol.sh`

**Files:**
- Create: `infra/scripts/init-symbol.sh`

This script is invoked by systemd as `init-symbol.sh BTCUSDT`. It sources env files, runs the pipeline, and retries up to 3 times on gaps.

- [ ] **Step 1: Create the script**

Create `infra/scripts/init-symbol.sh`:

```bash
#!/usr/bin/env bash
# init-symbol.sh <SYMBOL>
# Full kline ingest pipeline for one symbol. Run by systemd tradan-ingest@.service.
# Logs go to /var/log/tradan/ingest-<SYMBOL>.log (captured by systemd).
# Persistent failures appended to /var/log/tradan/errors.log.
set -euo pipefail

SYMBOL="${1:?Usage: init-symbol.sh SYMBOL}"
LOG_DIR="/var/log/tradan"
ERR_LOG="$LOG_DIR/errors.log"
ENV_FILE="/etc/tradan/ingest-${SYMBOL}.env"
BACKEND_DIR="/opt/tradan/backend"
UV="/root/.local/bin/uv"

mkdir -p "$LOG_DIR"

# Source DATABASE_URL
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
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x infra/scripts/init-symbol.sh
```

- [ ] **Step 3: Commit**

```bash
git add infra/scripts/init-symbol.sh
git commit -m "feat: add init-symbol.sh pipeline script with retry logic"
```

---

## Task 4: Update `cloud-init-base.yaml`

**Files:**
- Modify: `infra/scripts/cloud-init-base.yaml`
- Modify: `infra/base.tf`

Cloud-init needs to: (a) pass `symbols` from Terraform into the template, (b) create `/var/log/tradan/`, (c) write the systemd template unit, (d) write per-symbol env files, (e) start one service per symbol.

- [ ] **Step 1: Update `infra/base.tf` to pass `symbols`**

In `infra/base.tf`, the existing `templatefile(...)` call looks like:

```hcl
  user_data = templatefile("${path.module}/scripts/cloud-init-base.yaml", {
    db_password        = var.db_password
    db_name            = var.db_name
    db_user            = var.db_user
    git_repo_url       = local.git_clone_url
    vpc_cidr           = digitalocean_vpc.tradan.ip_range
    pgdata_volume_name = digitalocean_volume.pgdata.name
  })
```

Add `symbols = var.symbols` to that map:

```hcl
  user_data = templatefile("${path.module}/scripts/cloud-init-base.yaml", {
    db_password        = var.db_password
    db_name            = var.db_name
    db_user            = var.db_user
    git_repo_url       = local.git_clone_url
    vpc_cidr           = digitalocean_vpc.tradan.ip_range
    pgdata_volume_name = digitalocean_volume.pgdata.name
    symbols            = var.symbols
  })
```

- [ ] **Step 2: Append ingest bootstrap steps to `cloud-init-base.yaml`**

At the end of the `runcmd` section in `infra/scripts/cloud-init-base.yaml`, directly after the `touch /etc/tradan/setup_complete` line, add:

```yaml
  # Create log directory for ingest services
  - mkdir -p /var/log/tradan

  # Write systemd template unit for per-symbol ingest
  - |
    cat > /etc/systemd/system/tradan-ingest@.service << 'UNIT'
    [Unit]
    Description=Tradan kline ingest for %i
    After=network-online.target postgresql.service
    Wants=network-online.target

    [Service]
    Type=oneshot
    RemainAfterExit=yes
    EnvironmentFile=/etc/tradan/ingest-%i.env
    WorkingDirectory=/opt/tradan/backend
    ExecStart=/opt/tradan/infra/scripts/init-symbol.sh %i
    StandardOutput=append:/var/log/tradan/ingest-%i.log
    StandardError=append:/var/log/tradan/ingest-%i.log

    [Install]
    WantedBy=multi-user.target
    UNIT
    systemctl daemon-reload

  # Make init-symbol.sh executable (git may not preserve +x)
  - chmod +x /opt/tradan/infra/scripts/init-symbol.sh

  # Write per-symbol env files and start ingest services
  # Terraform templatefile() renders one block per symbol via for loop
%{ for symbol, cfg in symbols ~}
  - echo -e "INGEST_START=${cfg.start}\nINGEST_END=${cfg.end}" > /etc/tradan/ingest-${symbol}.env
  - systemctl enable --now tradan-ingest@${symbol}
%{ endfor ~}
```

> **Note on Terraform template syntax:** `%{ for ... }` / `%{ endfor }` is Terraform's templatefile loop syntax, not bash. Terraform renders this before cloud-init ever sees the file — the resulting YAML has one `echo` line and one `systemctl` line per symbol, with the symbol names and dates substituted as literals.

- [ ] **Step 3: Validate Terraform can render the template**

```bash
cd infra
terraform validate
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 4: Commit**

```bash
git add infra/base.tf infra/scripts/cloud-init-base.yaml
git commit -m "feat: cloud-init starts per-symbol ingest services on first boot"
```

---

## Task 5: Add `init-symbol` and `logs` Makefile targets

**Files:**
- Modify: `infra/Makefile`

- [ ] **Step 1: Update `.PHONY` line**

The existing `.PHONY` line is:

```makefile
.PHONY: init base-up train-up train-down update-ip base-ssh train-ssh \
        sweep-phase1 sweep-phase2 sweep-phase3 \
        winners evaluate progress status
```

Replace with:

```makefile
.PHONY: init base-up train-up train-down update-ip base-ssh train-ssh \
        sweep-phase1 sweep-phase2 sweep-phase3 \
        winners evaluate progress status \
        init-symbol logs
```

- [ ] **Step 2: Add `TARGET` default and the two new targets**

Append to the end of `infra/Makefile`:

```makefile
TARGET ?= base

## init-symbol: Re-trigger ingest for a symbol on the base droplet (idempotent)
## Usage: make init-symbol SYMBOL=BTCUSDT
init-symbol:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
	@if [ -z "$(SYMBOL)" ]; then echo "Usage: make init-symbol SYMBOL=BTCUSDT"; exit 1; fi
	ssh $(SSH_OPTS) root@$(BASE_IP) "systemctl restart tradan-ingest@$(SYMBOL)"

## logs: Tail logs on base or train droplet
## Usage: make logs                       (all base ingest logs + errors)
##        make logs SYMBOL=BTCUSDT        (single symbol on base)
##        make logs TARGET=train          (train droplet journal)
logs:
	@if [ "$(TARGET)" = "train" ]; then \
		if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet not running. Run: make train-up"; exit 1; fi; \
		ssh $(SSH_OPTS) root@$(TRAIN_IP) "tail -f /var/log/tradan/train.log 2>/dev/null || journalctl -f"; \
	else \
		if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi; \
		if [ -n "$(SYMBOL)" ]; then \
			ssh $(SSH_OPTS) root@$(BASE_IP) "tail -f /var/log/tradan/ingest-$(SYMBOL).log"; \
		else \
			ssh $(SSH_OPTS) root@$(BASE_IP) "tail -f /var/log/tradan/ingest-*.log /var/log/tradan/errors.log 2>/dev/null"; \
		fi; \
	fi
```

> **Important:** The indentation inside Makefile recipe lines must use a real TAB character, not spaces. Ensure your editor doesn't convert tabs to spaces in this file.

- [ ] **Step 3: Verify Makefile syntax**

```bash
cd infra
make --dry-run init-symbol SYMBOL=BTCUSDT BASE_IP=1.2.3.4 2>&1 | head -5
```

Expected: prints the `ssh` command without executing it.

- [ ] **Step 4: Commit**

```bash
git add infra/Makefile
git commit -m "feat: add init-symbol and logs make targets"
```

---

## Task 6: Update `infra/README.md`

**Files:**
- Modify: `infra/README.md`

- [ ] **Step 1: Add `symbols` variable to the environment variables section**

Find the "Export environment variables" section (around line 71–83). After the existing exports block, add a note about symbols:

Add this paragraph and code block after the existing exports block:

```
To ingest data for specific symbols, set TF_VAR_symbols (JSON map). The default is BTCUSDT from 2020-01:

    # Default (already set — no action needed for BTCUSDT)
    # TF_VAR_symbols='{"BTCUSDT":{"start":"2020-01","end":"2026-04"}}'

    # To add ETHUSDT as well:
    export TF_VAR_symbols='{"BTCUSDT":{"start":"2020-01","end":"2026-04"},"ETHUSDT":{"start":"2021-06","end":"2026-04"}}'
```

- [ ] **Step 2: Add a "Monitoring ingest" section after the base-up verification section**

Find the "### Step 3: Verify the base droplet is ready" section (around line 110). After that section and before the "Running a training batch" heading, insert:

```markdown
### Step 4: Monitor symbol ingest

After `make base-up`, the base droplet automatically starts ingesting kline data for all configured symbols in the background. Each symbol runs as a systemd service (`tradan-ingest@BTCUSDT`).

**Tail all ingest logs + error log:**

```bash
make logs
```

**Tail a specific symbol:**

```bash
make logs SYMBOL=BTCUSDT
```

**Check service status:**

```bash
make base-ssh
# On the droplet:
systemctl status tradan-ingest@BTCUSDT
```

Ingest for BTCUSDT from 2020 to 2026 takes roughly 30–60 minutes depending on network conditions. Once the service exits with code 0, data is ready for training.

**Re-trigger ingest for a symbol** (e.g. after adding a new symbol to `TF_VAR_symbols` and running `make base-up`):

```bash
make init-symbol SYMBOL=ETHUSDT
```

**If ingest fails after 3 retries,** the error is written to `/var/log/tradan/errors.log` on the base droplet. You'll see it in `make logs` output. To investigate and retry manually:

```bash
make base-ssh
# On the droplet:
cat /var/log/tradan/errors.log
systemctl restart tradan-ingest@BTCUSDT
```
```

- [ ] **Step 3: Update TL;DR at top of README**

Find the TL;DR block at the top (lines 3–19). After `make base-up`, add:

```markdown
make logs               # monitor ingest (runs in background, ~30–60min)
```

So the full TL;DR becomes:

```markdown
TL;DR
```
export TF_VAR_do_token="..."
export TF_VAR_db_password="..."
export TF_VAR_ssh_key_fingerprint="..."
export TF_VAR_git_repo_url="git@github.com:yourorg/tradan.git"
export TF_VAR_operator_ip="$(curl -s ifconfig.me)/32"
cd infra
make init       # terraform init (downloads DO provider)
make base-up    # provision base droplet + volume (~$34/mo ongoing)
make logs       # monitor symbol ingest (runs in background, ~30–60min)
make train-up   # spin up c-32 training droplet ($1/hr)
make sweep-phase1   # register 63 configs + fire up 28 workers
# ... wait ~5h ...
make evaluate       # eval winners on base droplet
make winners        # print ranked table
make train-down
```
```

- [ ] **Step 4: Commit**

```bash
git add infra/README.md
git commit -m "docs: document symbols variable, make logs, make init-symbol"
```

---

## Self-Review Checklist

After completing all tasks, verify:

- [ ] `terraform validate` passes in `infra/`
- [ ] `ingest verify` exits 1 when gaps exist: `echo $?` after running with known-gap data
- [ ] `make --dry-run init-symbol SYMBOL=BTCUSDT BASE_IP=1.2.3.4` prints ssh command
- [ ] `make --dry-run logs BASE_IP=1.2.3.4` prints tail command
- [ ] `make --dry-run logs TARGET=train TRAIN_IP=1.2.3.4` prints the train tail command
- [ ] `infra/scripts/init-symbol.sh` is executable: `ls -la infra/scripts/init-symbol.sh`
- [ ] All 6 commits are present: `git log --oneline -6`
