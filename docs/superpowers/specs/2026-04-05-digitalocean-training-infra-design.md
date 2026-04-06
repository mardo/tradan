# DigitalOcean Training Infrastructure — Design Spec

## Overview

Terraform-managed infrastructure on DigitalOcean for training, evaluating, and persisting RL trading models. The training workload is CPU-bound (exchange simulation in Python), so the architecture uses CPU-optimized droplets — not GPUs.

Three components: an always-on **base droplet** (PostgreSQL + eval/ops), an ephemeral **training droplet** (created per batch, destroyed when done), and a persistent **block volume** (model .zip files survive across training droplet lifecycles).

A `Makefile` wraps `terraform` and SSH commands into single-command operations. Python sweep scripts generate model configs in the DB; GNU parallel fans out N workers per droplet.

**Stack:** Terraform, DigitalOcean, PostgreSQL 15, Python 3.12 + uv, GNU parallel, Stable-Baselines3.

---

## Infrastructure Architecture

```
                        VPC (10.0.0.0/16, NYC3)
                               │
          ┌────────────────────┼────────────────────┐
          │                                          │
  ┌───────▼───────┐                        ┌────────▼────────┐
  │  base droplet │  ← always-on           │ train droplet   │  ← ephemeral
  │  s-4vcpu-8gb  │                        │ c-32 (default)  │
  │  $24/mo       │                        │ $1.00/hr        │
  │               │                        │                 │
  │  PostgreSQL   │  ← private IP only     │ 28 workers      │
  │  eval / ops   │                        │ GNU parallel    │
  └───────────────┘                        └────────┬────────┘
                                                    │ attach / detach
                                           ┌────────▼────────┐
                                           │  Block Volume   │  ← persistent, created once
                                           │  100 GB         │
                                           │  /mnt/models    │
                                           │  $10/mo         │
                                           └─────────────────┘
```

### Droplets

| Droplet | Size | Always-on | Purpose |
|---|---|---|---|
| base | s-4vcpu-8gb | Yes | PostgreSQL, eval, ops queries |
| train | c-32 (default) | No | Parallel training workers |

### Networking
- VPC in NYC3 — all resources in the same region and private network
- Firewall: SSH (22) restricted to operator IP; Postgres (5432) only accessible within VPC (private IP)
- Training droplets connect to PostgreSQL via base droplet's **private IP**

### Block Volume
- 100 GB, created once, never destroyed
- Attached to training droplet when it exists, detached when it is destroyed
- Mount point: `/mnt/models` — all trained model `.zip` files written here
- Training droplet's `MODELS_DIR` env var points to `/mnt/models`

---

## Terraform File Structure

```
infra/
├── main.tf          # Provider config, VPC, firewall rules
├── variables.tf     # All input variables with defaults
├── outputs.tf       # IPs, SSH commands, worker_count
├── base.tf          # Base droplet + cloud-init (postgres + eval)
├── train.tf         # Training droplet + volume attachment (conditional)
├── volume.tf        # Block storage volume (always exists)
└── scripts/
    ├── cloud-init-base.yaml     # Base droplet provisioning
    └── cloud-init-train.yaml    # Train droplet provisioning
```

### Key Variables (`variables.tf`)

```hcl
variable "do_token"           {}  # DigitalOcean API token (from env: TF_VAR_do_token)
variable "ssh_key_fingerprint" {} # DO SSH key fingerprint
variable "operator_ip"        {}  # Your IP for SSH firewall rule
variable "db_password"        {}  # PostgreSQL password (from env: TF_VAR_db_password)
variable "region"             { default = "nyc3" }
variable "train_enabled"      { default = false }
variable "train_droplet_size" { default = "c-32" }  # c-16 | c-32 | c-48
```

### Worker Count Lookup

```hcl
locals {
  worker_counts = {
    "c-16" = 14
    "c-32" = 28
    "c-48" = 44
  }
  worker_count = local.worker_counts[var.train_droplet_size]
}
```

Worker counts leave 2 vCPUs free for OS/overhead. Start at these values and increase by 2 at a time while monitoring CPU/memory until the droplet runs at ~90% utilization without freezing.

### Ephemeral Training Droplet Pattern

```hcl
resource "digitalocean_droplet" "train" {
  count  = var.train_enabled ? 1 : 0
  size   = var.train_droplet_size
  image  = "ubuntu-22-04-x64"
  region = var.region
  user_data = templatefile("scripts/cloud-init-train.yaml", {
    db_host      = digitalocean_droplet.base.ipv4_address_private
    db_password  = var.db_password
    worker_count = local.worker_count
  })
}

resource "digitalocean_volume_attachment" "models" {
  count      = var.train_enabled ? 1 : 0
  droplet_id = digitalocean_droplet.train[0].id
  volume_id  = digitalocean_volume.models.id
}
```

### Outputs (`outputs.tf`)

```hcl
output "base_ssh"      { value = "ssh root@${digitalocean_droplet.base.ipv4_address}" }
output "base_private_ip" { value = digitalocean_droplet.base.ipv4_address_private }
output "train_ssh"     { value = var.train_enabled ? "ssh root@${digitalocean_droplet.train[0].ipv4_address}" : "train droplet not running" }
output "worker_count"  { value = local.worker_count }
output "volume_id"     { value = digitalocean_volume.models.id }
```

---

## Cloud-Init Provisioning

### Base Droplet (`cloud-init-base.yaml`)

On first boot:
1. Install PostgreSQL 15
2. Configure `postgresql.conf`: `listen_addresses = 'localhost,<private_ip>'`
3. Configure `pg_hba.conf`: allow connections from VPC subnet (`10.0.0.0/16`)
4. Create database `tradan` and user `tradan` with the configured password
5. Run migrations: `uv run ingest migrate`
6. Install `uv`, `git`, `postgresql-client`
7. Clone the repo to `/opt/tradan`
8. Write `/opt/tradan/backend/.env` with `DATABASE_URL`

### Training Droplet (`cloud-init-train.yaml`)

On first boot:
1. Install `uv`, `git`, `parallel` (GNU parallel)
2. Mount block volume at `/mnt/models`:
   - Device path on DigitalOcean: `/dev/disk/by-id/scsi-0DO_Volume_<volume-name>`
   - Guard `mkfs.ext4` with a check (`blkid`) so it only formats on the first-ever attach — subsequent attaches skip formatting and mount the existing filesystem
   - Add entry to `/etc/fstab` for persistence across reboots
3. Clone the repo to `/opt/tradan`
4. Write `/opt/tradan/backend/.env`:
   ```
   DATABASE_URL=postgresql://tradan:<password>@<base_private_ip>:5432/tradan
   ```
5. Create symlink: `ln -sfn /mnt/models /opt/tradan/backend/trained_models`
   — This is how the trainer finds the persistent volume. The trainer code uses a hardcoded path relative to the repo (`backend/trained_models`); the symlink redirects it to the block volume without modifying any Python code.
6. Run `uv sync` in `/opt/tradan/backend`
7. Write `/etc/tradan/worker_count` with the computed worker count value

---

## Scripts & Makefile

### Scripts Layout

```
infra/scripts/
├── sweep_phase1.py        # Generate & register Phase 1 BTC configs in DB
├── sweep_phase2.py        # Query Phase 1 winners → generate Phase 2 configs
├── sweep_phase3.py        # Query Phase 2 winners → generate Phase 3 configs
├── run_sweep.sh           # Run N parallel training workers via GNU parallel
├── evaluate_winners.sh    # Run `train evaluate` on top N models
└── winners.sql            # Ranked winner query with filter chain
```

### `run_sweep.sh`

```bash
#!/usr/bin/env bash
# Reads worker count from /etc/tradan/worker_count
# Pulls pending model names from DB, fans them out with GNU parallel
WORKERS=$(cat /etc/tradan/worker_count)
cd /opt/tradan/backend
uv run train list --names-only --status pending \
  | parallel -j"$WORKERS" "uv run train start --model {}"
```

> **Required CLI addition:** `train list` must be extended with `--names-only` (print only model names, one per line) and `--status <pending|completed|failed>` (filter by run status) flags. These are needed for `run_sweep.sh` to pipe model names into GNU parallel.

### `evaluate_winners.sh`

```bash
#!/usr/bin/env bash
# Runs `train evaluate` on top 20 models from winners.sql that don't yet have an eval run
cd /opt/tradan/backend
psql "$DATABASE_URL" -f /opt/tradan/infra/scripts/winners_no_eval.sql --tuples-only --no-align \
  | parallel -j4 "uv run train evaluate --model {1} --run {2}"
```

A companion `winners_no_eval.sql` queries for completed training runs that have no evaluation run yet, limited to the top 20 by training PnL. Evaluation is less CPU-intensive so 4 parallel workers is sufficient on the base droplet.

### `winners.sql`

```sql
SELECT
    mc.name,
    tr_train.total_pnl       AS train_pnl,
    tr_eval.total_pnl        AS holdout_pnl,
    tr_eval.sharpe_ratio,
    tr_eval.max_drawdown,
    tr_eval.total_trades,
    tr_eval.win_rate,
    tr_eval.total_pnl / NULLIF(tr_train.total_pnl, 0) AS generalization_ratio
FROM model_configs mc
JOIN training_runs tr_train ON tr_train.model_config_id = mc.id
    AND tr_train.run_type = 'train' AND tr_train.status = 'completed'
JOIN training_runs tr_eval  ON tr_eval.model_config_id  = mc.id
    AND tr_eval.run_type  = 'evaluate' AND tr_eval.status = 'completed'
WHERE tr_eval.total_trades > 10
  AND tr_eval.total_pnl    > 0
  AND tr_eval.max_drawdown < 0.25
  AND tr_eval.total_pnl / NULLIF(tr_train.total_pnl, 0) > 0.5
ORDER BY tr_eval.sharpe_ratio DESC
LIMIT 20;
```

### `Makefile`

```makefile
# Usage:
#   make base-up              provision base droplet (first time)
#   make train-up             provision training droplet + attach volume
#   make sweep-phase1         generate phase 1 configs + run training
#   make winners              print ranked winner table
#   make evaluate             evaluate top 20 models on holdout data
#   make train-down           destroy training droplet (volume persists)

TF_DIR       := $(CURDIR)
TRAIN_SSH    := $(shell cd $(TF_DIR) && terraform output -raw train_ssh 2>/dev/null)
BASE_SSH     := $(shell cd $(TF_DIR) && terraform output -raw base_ssh 2>/dev/null)
BASE_PRIV_IP := $(shell cd $(TF_DIR) && terraform output -raw base_private_ip 2>/dev/null)

base-up:
	terraform apply -target=digitalocean_droplet.base -target=digitalocean_volume.models

train-up:
	terraform apply -var="train_enabled=true"

train-down:
	terraform apply -var="train_enabled=false"

train-ssh:
	$(TRAIN_SSH)

base-ssh:
	$(BASE_SSH)

sweep-phase1:
	scp scripts/sweep_phase1.py root@$$(terraform output -raw train_ssh | awk '{print $$NF}'):/tmp/
	$(TRAIN_SSH) "cd /opt/tradan/backend && uv run python /tmp/sweep_phase1.py && bash /opt/tradan/infra/scripts/run_sweep.sh"

sweep-phase2:
	scp scripts/sweep_phase2.py root@$$(terraform output -raw train_ssh | awk '{print $$NF}'):/tmp/
	$(TRAIN_SSH) "cd /opt/tradan/backend && uv run python /tmp/sweep_phase2.py && bash /opt/tradan/infra/scripts/run_sweep.sh"

sweep-phase3:
	scp scripts/sweep_phase3.py root@$$(terraform output -raw train_ssh | awk '{print $$NF}'):/tmp/
	$(TRAIN_SSH) "cd /opt/tradan/backend && uv run python /tmp/sweep_phase3.py && bash /opt/tradan/infra/scripts/run_sweep.sh"

winners:
	$(BASE_SSH) "psql \$$DATABASE_URL -f /opt/tradan/infra/scripts/winners.sql"

evaluate:
	$(BASE_SSH) "bash /opt/tradan/infra/scripts/evaluate_winners.sh"

.PHONY: base-up train-up train-down train-ssh base-ssh sweep-phase1 sweep-phase2 sweep-phase3 winners evaluate
```

---

## Training Plan (BTCUSDT)

### Phase 1 — Baseline Sweep

Find which interval + algorithm combinations work at all.

| Dimension | Values | Count |
|---|---|---|
| Symbol | BTCUSDT | 1 (fixed) |
| Intervals | 1m, 5m, 15m, 30m, 1h, 4h, 1d | 7 |
| Algorithms | PPO, SAC, A2C | 3 |
| Seeds | 42, 123, 456 | 3 |
| Lookback | 500 | fixed |
| Learning rate | 3e-4 | fixed |
| Timesteps | 1,000,000 | fixed |

**Total: 63 runs.** Expected wall time on c-32 (28 workers): ~4–5h. Cost: ~$5.

→ Advance **top 5 combos** by holdout Sharpe (filtered by winner criteria).

### Phase 2 — Hyperparameter Expansion

Take Phase 1 winners, vary training knobs.

| Dimension | Values | Count |
|---|---|---|
| Configs | Top 5 from Phase 1 | 5 |
| Lookback | 100, 250, 500, 1000 | 4 |
| Learning rate | 1e-4, 3e-4, 1e-3 | 3 |
| Seeds | 3 per combo | 3 |
| Timesteps | 1,000,000 | fixed |

**Total: ~180 runs.** Expected wall time: ~7h. Cost: ~$7.

→ Advance **top 5 configs** by holdout Sharpe.

### Phase 3 — Long Training

Top 5 from Phase 2, trained much longer.

| Dimension | Values |
|---|---|
| Configs | Top 5 from Phase 2 |
| Seeds | 3 per config |
| Timesteps | 5,000,000 |

**Total: 15 runs.** Expected wall time: ~12h. Cost: ~$12.

→ **Final keeper models.** Stored on block volume at `/mnt/models`.

### Winner Selection Criteria (applied at every phase transition)

```
total_trades > 10               -- excludes "do nothing" models
holdout_pnl > 0                 -- profitable on unseen data
max_drawdown < 0.25             -- not recklessly risky
holdout_pnl / train_pnl > 0.5  -- not overfit
ORDER BY sharpe_ratio DESC      -- best risk-adjusted return first
```

### Cost Summary

| Component | Cost |
|---|---|
| Phase 1 compute | ~$5 |
| Phase 2 compute | ~$7 |
| Phase 3 compute | ~$12 |
| Base droplet | $24/mo |
| Block volume | $10/mo |
| **Total compute** | **~$24 one-time** |
| **Total recurring** | **~$34/mo** |

---

## Worker Tuning Process

Start at the conservative default for each droplet size, then increase by 2 until you see ~90% CPU utilization without the droplet becoming unresponsive.

| Droplet | vCPUs | Default workers | Max to try |
|---|---|---|---|
| c-16 | 16 | 14 | 16 |
| c-32 | 32 | 28 | 32 |
| c-48 | 48 | 44 | 48 |

Monitor with `htop` or `vmstat 2` during a sweep. If memory pressure appears (swap usage), reduce workers by 2. Update `/etc/tradan/worker_count` and re-run without reprovisioning.

---

## Secrets Management

All secrets passed via environment variables — never committed to git.

```bash
export TF_VAR_do_token="..."
export TF_VAR_db_password="..."
export TF_VAR_operator_ip="$(curl -s ifconfig.me)/32"
export TF_VAR_ssh_key_fingerprint="..."
```

Terraform state stored locally (`terraform.tfstate`). Add `infra/terraform.tfstate*` to `.gitignore`.

---

## File Structure (Final)

```
infra/
├── main.tf
├── variables.tf
├── outputs.tf
├── base.tf
├── train.tf
├── volume.tf
├── Makefile
└── scripts/
    ├── cloud-init-base.yaml
    ├── cloud-init-train.yaml
    ├── sweep_phase1.py
    ├── sweep_phase2.py
    ├── sweep_phase3.py
    ├── run_sweep.sh
    ├── evaluate_winners.sh
    └── winners.sql
```
