# DigitalOcean training infrastructure
#
# Prerequisites — export these before running any target:
#   export TF_VAR_do_token="..."
#   export TF_VAR_db_password="..."
#   export TF_VAR_ssh_key_name="your-do-ssh-key-name"
#
# operator_ip is auto-detected from your current public IP on every apply.
# No need to export it — the Makefile handles it automatically.
# Code is deployed via rsync (make sync-train / make sync-base) — no git credentials needed.

TF       := terraform
SCRIPTS  := $(CURDIR)/scripts
REPO_ROOT := $(abspath $(CURDIR)/..)

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
        sync-train sync-base \
        sweep-phase1 sweep-phase2 sweep-phase3 \
        winners evaluate progress status

## init: Initialize Terraform (run once after cloning)
init:
	$(TF) init

## base-up: Provision base droplet + block volume (first-time setup)
base-up:
	@echo "Your current IP: $(MY_IP_CIDR)"
	$(TF_APPLY) \
	  -target=digitalocean_vpc.tradan \
	  -target=digitalocean_firewall.tradan \
	  -target=digitalocean_droplet.base \
	  -target=digitalocean_volume.models

## train-up: Create training droplet and attach block volume
train-up:
	@echo "Your current IP: $(MY_IP_CIDR)"
	$(TF_APPLY) -var="train_enabled=true"

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

## sync-train: Rsync backend/ + infra/ to training droplet, then install Python deps
sync-train:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	@echo "Waiting for cloud-init to complete..."
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "until [ -f /etc/tradan/setup_complete ]; do sleep 10; echo 'Waiting...'; done; echo 'Ready.'"
	@echo "Syncing backend/ and infra/ to training droplet..."
	rsync -az --delete \
	  --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
	  --exclude 'trained_models' --exclude '.git' \
	  -e "ssh $(SSH_OPTS)" \
	  $(REPO_ROOT)/backend/ root@$(TRAIN_IP):/opt/tradan/backend/
	rsync -az --delete \
	  --exclude '.terraform' --exclude 'terraform.tfstate*' --exclude '.env' \
	  -e "ssh $(SSH_OPTS)" \
	  $(REPO_ROOT)/infra/ root@$(TRAIN_IP):/opt/tradan/infra/
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "cd /opt/tradan/backend && /root/.local/bin/uv sync"
	@echo "Training droplet is ready."

## sync-base: Rsync backend/ + infra/ to base droplet, install deps, run migrations
sync-base:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
	@echo "Waiting for cloud-init to complete..."
	ssh $(SSH_OPTS) root@$(BASE_IP) "until [ -f /etc/tradan/setup_complete ]; do sleep 10; echo 'Waiting...'; done; echo 'Ready.'"
	@echo "Syncing backend/ and infra/ to base droplet..."
	rsync -az --delete \
	  --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
	  --exclude 'trained_models' --exclude '.git' \
	  -e "ssh $(SSH_OPTS)" \
	  $(REPO_ROOT)/backend/ root@$(BASE_IP):/opt/tradan/backend/
	rsync -az --delete \
	  --exclude '.terraform' --exclude 'terraform.tfstate*' --exclude '.env' \
	  -e "ssh $(SSH_OPTS)" \
	  $(REPO_ROOT)/infra/ root@$(BASE_IP):/opt/tradan/infra/
	ssh $(SSH_OPTS) root@$(BASE_IP) "cd /opt/tradan/backend && /root/.local/bin/uv sync && /root/.local/bin/uv run ingest migrate"
	@echo "Base droplet is ready."

## sweep-phase1: Sync code, register Phase 1 configs, start training (63 runs)
sweep-phase1:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	$(MAKE) sync-train
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "cd /opt/tradan/backend && /root/.local/bin/uv run python /opt/tradan/infra/scripts/sweep_phase1.py"
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "bash /opt/tradan/infra/scripts/run_sweep.sh"

## sweep-phase2: Sync code, register Phase 2 configs (requires Phase 1 winners evaluated)
sweep-phase2:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	$(MAKE) sync-train
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "cd /opt/tradan/backend && /root/.local/bin/uv run python /opt/tradan/infra/scripts/sweep_phase2.py"
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "bash /opt/tradan/infra/scripts/run_sweep.sh"

## sweep-phase3: Sync code, register Phase 3 configs (requires Phase 2 winners evaluated)
sweep-phase3:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	$(MAKE) sync-train
	ssh $(SSH_OPTS) root@$(TRAIN_IP) "cd /opt/tradan/backend && /root/.local/bin/uv run python /opt/tradan/infra/scripts/sweep_phase3.py"
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
