# DigitalOcean training infrastructure
#
# Variables are loaded automatically from .env if it exists.
# Copy .env.example to .env and fill in your values before running any target.
#
# operator_ip is auto-detected from your current public IP on every apply.
# No need to set it — the Makefile handles it automatically.

# Auto-load .env so TF_VAR_* are always in sync with the file.
# Strips surrounding double-quotes so make and bash source both work.
ifneq (,$(wildcard .env))
  $(foreach kv,$(shell grep -v '^\#' .env | grep -v '^$$' | sed 's/^\([^=]*\)="\(.*\)"$$/\1=\2/'),$(eval export $(kv)))
endif

TF      := terraform
SCRIPTS := $(CURDIR)/scripts
BACKEND_SCRIPTS := $(CURDIR)/../backend/scripts

# Auto-detect current public IP for the SSH firewall rule.
# Resolved once per make invocation so all targets use a consistent IP.
MY_IP = $(shell curl -4sf https://ifconfig.me 2>/dev/null || curl -4sf https://api.ipify.org 2>/dev/null || echo "0.0.0.0")
MY_IP_CIDR = $(MY_IP)/32

# Shared terraform apply flags: always pass current IP + allow target overrides
TF_APPLY = $(TF) apply -var="operator_ip=$(MY_IP_CIDR)"

# Read live Terraform outputs (empty string if resource doesn't exist yet)
BASE_IP  = $(shell $(TF) output -raw base_ip 2>/dev/null || echo "")
TRAIN_IP = $(shell $(TF) output -raw train_ip 2>/dev/null || echo "")

# SSH options. Override SSH_KEY to point at a specific private key, e.g.:
#   make train-ssh SSH_KEY=~/.ssh/id_ed25519
SSH_KEY  ?=
SSH_OPTS  = -o StrictHostKeyChecking=accept-new $(if $(SSH_KEY),-i $(SSH_KEY),)

.PHONY: init base-up train-up train-down update-ip base-ssh train-ssh \
        register-phase1 sweep-phase1 sweep-phase1-concurrent sweep-phase2 sweep-phase3 \
        winners evaluate progress status \
        init-symbol logs tunnel db-worker-access

## init: Initialize Terraform (run once after cloning)
init:
	$(TF) init

## base-up: Provision base droplet + block volumes (first-time setup)
base-up:
	@echo "Your current IP: $(MY_IP_CIDR)"
	$(TF_APPLY) \
	  -target=digitalocean_vpc.tradan \
	  -target=digitalocean_firewall.tradan \
	  -target=digitalocean_volume.pgdata \
	  -target=digitalocean_volume.models \
	  -target=digitalocean_droplet.base \
	  -target=digitalocean_volume_attachment.pgdata \
	  -target=digitalocean_volume_attachment.models_base
	@ip=$$($(TF) output -raw base_ip 2>/dev/null); \
	  if [ -n "$$ip" ]; then python3 $(SCRIPTS)/update_cursor_mcp_ssh.py base "$$ip"; fi

## train-up: Create training droplet and attach block volume
train-up:
	@echo "Your current IP: $(MY_IP_CIDR)"
	$(TF_APPLY) -var="train_enabled=true"
	@ip=$$($(TF) output -raw train_ip 2>/dev/null); \
	  if [ -n "$$ip" ]; then python3 $(SCRIPTS)/update_cursor_mcp_ssh.py train "$$ip"; fi

## train-down: Destroy training droplet (block volume persists)
train-down:
	@echo "Your current IP: $(MY_IP_CIDR)"
	$(TF_APPLY) -var="train_enabled=false"

## update-ip: Re-apply firewall with your current IP (run after IP change)
update-ip:
	@echo "Updating firewall to allow SSH from: $(MY_IP_CIDR)"
	$(TF_APPLY)

## base-ssh: Open SSH session to base droplet
base-ssh:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
	ssh $(SSH_OPTS) root@$(BASE_IP)

## train-ssh: Open SSH session to training droplet
train-ssh:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	ssh $(SSH_OPTS) root@$(TRAIN_IP)

## register-phase1: Register Phase 1 model configs into the DB (no train droplet needed).
## Run from local machine when using distributed workers (uv run train worker).
register-phase1:
	cd $(CURDIR)/../backend && /usr/bin/env uv run python scripts/sweep_phase1.py

## sweep-phase1: Register Phase 1 configs and start training (63 runs)
## Waits for cloud-init to complete before running (up to 3 min on first boot)
sweep-phase1:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	@echo "Waiting for cloud-init to complete..."
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "until [ -f /etc/tradan/setup_complete ]; do sleep 10; echo 'Waiting...'; done; echo 'Ready.'"
	scp $(SSH_OPTS) $(BACKEND_SCRIPTS)/sweep_phase1.py root@$(TRAIN_IP):/tmp/sweep_phase1.py
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "cd /opt/tradan/backend && /root/.local/bin/uv run python /tmp/sweep_phase1.py"
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "bash /opt/tradan/infra/scripts/run_sweep.sh"

## sweep-phase1-concurrent: Register Phase 1 configs and train sequentially (one model at a time, full CPU per run)
sweep-phase1-concurrent:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	@echo "Waiting for cloud-init to complete..."
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "until [ -f /etc/tradan/setup_complete ]; do sleep 10; echo 'Waiting...'; done; echo 'Ready.'"
	scp $(SSH_OPTS) $(BACKEND_SCRIPTS)/sweep_phase1.py root@$(TRAIN_IP):/tmp/sweep_phase1.py
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "cd /opt/tradan/backend && /root/.local/bin/uv run python /tmp/sweep_phase1.py"
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "bash /opt/tradan/infra/scripts/run_sweep_sequential.sh"

## sweep-phase2: Register Phase 2 configs and start training (requires Phase 1 winners evaluated)
sweep-phase2:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	scp $(SSH_OPTS) $(BACKEND_SCRIPTS)/sweep_phase2.py root@$(TRAIN_IP):/tmp/sweep_phase2.py
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "cd /opt/tradan/backend && /root/.local/bin/uv run python /tmp/sweep_phase2.py"
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "bash /opt/tradan/infra/scripts/run_sweep.sh"

## sweep-phase3: Register Phase 3 configs and start training (requires Phase 2 winners evaluated)
sweep-phase3:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	scp $(SSH_OPTS) $(BACKEND_SCRIPTS)/sweep_phase3.py root@$(TRAIN_IP):/tmp/sweep_phase3.py
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "cd /opt/tradan/backend && /root/.local/bin/uv run python /tmp/sweep_phase3.py"
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "bash /opt/tradan/infra/scripts/run_sweep.sh"

## winners: Show ranked winner table (runs query on base droplet)
winners:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
	ssh $(SSH_OPTS) root@$(BASE_IP) "psql \$$DATABASE_URL -f /opt/tradan/infra/scripts/winners.sql"

## evaluate: Run holdout evaluation on top 20 unevaluated models (runs on base droplet)
evaluate:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
	ssh $(SSH_OPTS) root@$(BASE_IP) "bash /opt/tradan/infra/scripts/evaluate_winners.sh"

## progress: Show training progress — completed/failed/pending counts + recent completions
progress:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running."; exit 1; fi
	ssh $(SSH_OPTS) root@$(BASE_IP) "psql \$$DATABASE_URL -c \"\
	  SELECT status, count(*) FROM training_runs WHERE run_type='train' GROUP BY status ORDER BY status;\
	  SELECT mc.name, tr.total_pnl, tr.sharpe_ratio, tr.total_trades, tr.completed_at \
	  FROM training_runs tr JOIN model_configs mc ON mc.id = tr.model_config_id \
	  WHERE tr.run_type='train' AND tr.status='completed' \
	  ORDER BY tr.completed_at DESC LIMIT 10;\""

## status: Show current Terraform output (IPs, worker count, etc.)
status:
	$(TF) output

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

## tunnel: Open SSH tunnel to base droplet's PostgreSQL on localhost:5432
## Run in a separate terminal, then use DATABASE_URL pointing at localhost
## Usage: make tunnel
tunnel:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
	@echo "PostgreSQL tunnel open: postgresql://$(TF_VAR_db_user):***@localhost:5432/$(TF_VAR_db_name)"
	@echo "DATABASE_URL=postgresql://$(TF_VAR_db_user):$(TF_VAR_db_password)@localhost:5432/$(TF_VAR_db_name)"
	@echo "Press Ctrl+C to close."
	ssh $(SSH_OPTS) -N -L 5433:localhost:5432 root@$(BASE_IP)

## db-worker-access: Whitelist your current IP for direct PostgreSQL access from outside the VPC.
##
## Runs two steps:
##   1. terraform apply  — updates DO firewall to allow port 5432 from your IP
##   2. SSH patch        — adds pg_hba.conf entry + reloads PostgreSQL on the running base droplet
##
## Usage:
##   make db-worker-access              # auto-detects your current public IP
##   make db-worker-access IP=1.2.3.4   # explicit IP (CIDR /32 is appended automatically)
##
## For persistent multi-IP setups add TF_VAR_db_worker_ips='["x.x.x.x/32","y.y.y.y/32"]' to infra/.env
## and run `make update-ip` to re-apply.
IP ?= $(MY_IP)
IP_CIDR = $(IP)/32

db-worker-access:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
	@echo "Whitelisting $(IP_CIDR) for PostgreSQL access..."
	$(TF_APPLY) -var='db_worker_ips=["$(IP_CIDR)"]'
	@echo "Patching pg_hba.conf on base droplet for $(IP_CIDR)..."
	ssh $(SSH_OPTS) root@$(BASE_IP) " \
	  PG_CONF=\$$(find /etc/postgresql -name postgresql.conf | head -1) && \
	  grep -q \"listen_addresses = '\*'\" \"\$$PG_CONF\" || \
	    sed -i \"/^#*listen_addresses/c\\listen_addresses = '*'\" \"\$$PG_CONF\"; \
	  PG_HBA=\$$(find /etc/postgresql -name pg_hba.conf | head -1) && \
	  grep -qF '$(IP_CIDR)' \"\$$PG_HBA\" || \
	    echo 'host $(TF_VAR_db_name) $(TF_VAR_db_user) $(IP_CIDR) scram-sha-256' >> \"\$$PG_HBA\"; \
	  systemctl reload postgresql && echo 'Done. PostgreSQL reloaded.'"
	@echo ""
	@echo "Set these in backend/.env on your local machine:"
	@echo "  DATABASE_URL=postgresql://$(TF_VAR_db_user):$(TF_VAR_db_password)@$(BASE_IP):5432/$(TF_VAR_db_name)"
	@echo "  MODELS_RSYNC_TARGET=root@$(BASE_IP):/mnt/models"
