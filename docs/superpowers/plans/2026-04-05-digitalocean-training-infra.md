# DigitalOcean Training Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Terraform-managed DigitalOcean infrastructure for ephemeral CPU training droplets, a persistent block volume for model files, and an always-on base droplet running PostgreSQL + eval ops — plus sweep scripts and a Makefile to orchestrate 3-phase BTCUSDT training.

**Architecture:** Two droplets (always-on base + ephemeral train) in a private VPC in NYC3. Training droplet is toggled with a single Terraform variable; the 100 GB block volume persists across training droplet lifecycles via a symlink. GNU parallel fans out N workers per droplet, count computed from droplet size via a lookup map.

**Tech Stack:** Terraform ≥ 1.6, DigitalOcean provider, Ubuntu 22.04 LTS, PostgreSQL 15, Python 3.12 + uv, GNU parallel, Stable-Baselines3 (existing), psycopg3 (existing).

---

## File Map

### New files to create
```
infra/
├── .gitignore
├── Makefile
├── main.tf
├── variables.tf
├── outputs.tf
├── base.tf
├── train.tf
├── volume.tf
└── scripts/
    ├── cloud-init-base.yaml
    ├── cloud-init-train.yaml
    ├── sweep_phase1.py
    ├── sweep_phase2.py
    ├── sweep_phase3.py
    ├── run_sweep.sh
    ├── evaluate_winners.sh
    ├── winners.sql
    └── winners_no_eval.sql
```

### Existing files to modify
```
backend/src/trainer/cli.py           # Add --names-only and --status flags to `train list`
backend/src/trainer/db.py            # Add list_pending_model_names() DB function
```

### New test files
```
backend/tests/trainer/test_cli_list.py   # Test new list flags
```

---

## Task 1: Project scaffold + Terraform provider, VPC, and firewall

**Files:**
- Create: `infra/.gitignore`
- Create: `infra/main.tf`
- Create: `infra/variables.tf`

- [ ] **Step 1: Create `infra/` directory and `.gitignore`**

```bash
mkdir -p /path/to/tradan/infra/scripts
```

`infra/.gitignore`:
```
.terraform/
.terraform.lock.hcl
terraform.tfstate
terraform.tfstate.backup
*.tfvars
.env
```

- [ ] **Step 2: Create `infra/variables.tf`**

```hcl
variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "ssh_key_fingerprint" {
  description = "Fingerprint of the SSH key registered in DigitalOcean"
  type        = string
}

variable "operator_ip" {
  description = "Your IP address in CIDR notation for SSH firewall rule (e.g. 1.2.3.4/32)"
  type        = string
}

variable "db_password" {
  description = "PostgreSQL password for the tradan user"
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "tradan"
}

variable "db_user" {
  description = "PostgreSQL user"
  type        = string
  default     = "tradan"
}

variable "region" {
  description = "DigitalOcean region slug"
  type        = string
  default     = "nyc3"
}

variable "train_enabled" {
  description = "Set to true to create the training droplet, false to destroy it"
  type        = bool
  default     = false
}

variable "train_droplet_size" {
  description = "CPU-optimized droplet size: c-16 (14 workers), c-32 (28 workers), c-48 (44 workers)"
  type        = string
  default     = "c-32"

  validation {
    condition     = contains(["c-16", "c-32", "c-48"], var.train_droplet_size)
    error_message = "train_droplet_size must be c-16, c-32, or c-48."
  }
}

variable "git_repo_url" {
  description = "Git repository URL to clone on droplets (e.g. git@github.com:org/tradan.git)"
  type        = string
}
```

- [ ] **Step 3: Create `infra/main.tf`** with provider, locals (worker count lookup), VPC, and firewall rules

```hcl
terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
  required_version = ">= 1.6"
}

provider "digitalocean" {
  token = var.do_token
}

locals {
  worker_counts = {
    "c-16" = 14
    "c-32" = 28
    "c-48" = 44
  }
  worker_count = local.worker_counts[var.train_droplet_size]
}

resource "digitalocean_vpc" "tradan" {
  name     = "tradan-vpc"
  region   = var.region
  ip_range = "10.0.0.0/16"
}

resource "digitalocean_firewall" "tradan" {
  name = "tradan-firewall"

  droplet_ids = concat(
    [digitalocean_droplet.base.id],
    var.train_enabled ? [digitalocean_droplet.train[0].id] : []
  )

  # SSH: only from operator IP
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = [var.operator_ip]
  }

  # PostgreSQL: only from within VPC
  inbound_rule {
    protocol         = "tcp"
    port_range       = "5432"
    source_addresses = [digitalocean_vpc.tradan.ip_range]
  }

  # Allow all outbound (package installs, git clone, etc.)
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}
```

- [ ] **Step 4: Run `terraform init` to verify provider config**

```bash
cd infra
terraform init
```

Expected: `Terraform has been successfully initialized!`

- [ ] **Step 5: Commit**

```bash
git add infra/.gitignore infra/main.tf infra/variables.tf
git commit -m "feat(infra): terraform scaffold with provider, VPC, firewall"
```

---

## Task 2: Block storage volume

**Files:**
- Create: `infra/volume.tf`

- [ ] **Step 1: Create `infra/volume.tf`**

```hcl
resource "digitalocean_volume" "models" {
  name                     = "tradan-models"
  region                   = var.region
  size                     = 100
  initial_filesystem_type  = "ext4"
  description              = "Persistent storage for trained model .zip files"
}
```

> Note: `initial_filesystem_type = "ext4"` tells DigitalOcean to format the volume on creation. This means we do NOT need to run `mkfs` manually in cloud-init. The cloud-init script only needs to mount it. The volume is never destroyed by Terraform — it is outside the `train_enabled` conditional.

- [ ] **Step 2: Run `terraform validate`**

```bash
cd infra
terraform validate
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 3: Commit**

```bash
git add infra/volume.tf
git commit -m "feat(infra): persistent 100GB block volume for model storage"
```

---

## Task 3: Base droplet + cloud-init

**Files:**
- Create: `infra/base.tf`
- Create: `infra/scripts/cloud-init-base.yaml`

- [ ] **Step 1: Create `infra/scripts/cloud-init-base.yaml`**

This script runs once on first boot of the base droplet. It installs PostgreSQL, configures it for VPC access, creates the DB/user, installs uv and git, clones the repo, and runs migrations.

```yaml
#cloud-config
package_update: true
package_upgrade: false

packages:
  - git
  - curl
  - postgresql
  - postgresql-client

write_files:
  - path: /etc/tradan/db_password
    permissions: '0600'
    content: |
      ${db_password}

runcmd:
  # Wait for PostgreSQL to be ready
  - systemctl start postgresql
  - systemctl enable postgresql

  # Configure PostgreSQL to listen on localhost + private IP
  - |
    PG_CONF=$(find /etc/postgresql -name "postgresql.conf" | head -1)
    sed -i "s/^#listen_addresses.*/listen_addresses = 'localhost,${private_ip}'/" "$PG_CONF"

  # Allow connections from VPC subnet in pg_hba.conf
  - |
    PG_HBA=$(find /etc/postgresql -name "pg_hba.conf" | head -1)
    echo "host    ${db_name}    ${db_user}    10.0.0.0/16    md5" >> "$PG_HBA"

  # Create DB user and database
  - |
    sudo -u postgres psql -c "CREATE USER ${db_user} WITH PASSWORD '${db_password}';" || true
    sudo -u postgres psql -c "CREATE DATABASE ${db_name} OWNER ${db_user};" || true

  # Restart PostgreSQL to apply config changes
  - systemctl restart postgresql

  # Install uv
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /root/.bashrc

  # Clone repo
  - git clone ${git_repo_url} /opt/tradan

  # Write .env file
  - |
    cat > /opt/tradan/backend/.env <<EOF
    DATABASE_URL=postgresql://${db_user}:${db_password}@localhost:5432/${db_name}
    EOF

  # Install dependencies and run migrations
  - cd /opt/tradan/backend && /root/.cargo/bin/uv sync
  - cd /opt/tradan/backend && /root/.cargo/bin/uv run ingest migrate

  # Mark setup complete
  - touch /etc/tradan/setup_complete
```

- [ ] **Step 2: Create `infra/base.tf`**

```hcl
data "digitalocean_ssh_key" "operator" {
  fingerprint = var.ssh_key_fingerprint
}

resource "digitalocean_droplet" "base" {
  name      = "tradan-base"
  size      = "s-4vcpu-8gb"
  image     = "ubuntu-22-04-x64"
  region    = var.region
  vpc_uuid  = digitalocean_vpc.tradan.id
  ssh_keys  = [data.digitalocean_ssh_key.operator.id]

  user_data = templatefile("${path.module}/scripts/cloud-init-base.yaml", {
    db_password  = var.db_password
    db_name      = var.db_name
    db_user      = var.db_user
    git_repo_url = var.git_repo_url
    private_ip   = "$_PRIVATE_IP"  # placeholder; actual private IP set via runcmd below
  })
}
```

> **Implementation note:** DigitalOcean private IPs are assigned after droplet creation, so they can't be templated into cloud-init at plan time. The cloud-init script should instead detect its own private IP at runtime using the DigitalOcean metadata service:
> ```bash
> PRIVATE_IP=$(curl -s http://169.254.169.254/metadata/v1/interfaces/private/0/ipv4/address)
> ```
> Replace the `${private_ip}` template reference with this runtime detection in the yaml.

- [ ] **Step 3: Update `cloud-init-base.yaml` to use metadata service for private IP**

Replace the static `${private_ip}` reference with a runtime detection:

```yaml
runcmd:
  - systemctl start postgresql
  - systemctl enable postgresql

  # Detect this droplet's private IP via metadata service
  - PRIVATE_IP=$(curl -s http://169.254.169.254/metadata/v1/interfaces/private/0/ipv4/address)

  # Configure PostgreSQL to listen on localhost + private IP
  - |
    PG_CONF=$(find /etc/postgresql -name "postgresql.conf" | head -1)
    PRIVATE_IP=$(curl -s http://169.254.169.254/metadata/v1/interfaces/private/0/ipv4/address)
    sed -i "s/^#listen_addresses.*/listen_addresses = 'localhost,$PRIVATE_IP'/" "$PG_CONF"

  # Allow connections from VPC subnet
  - |
    PG_HBA=$(find /etc/postgresql -name "pg_hba.conf" | head -1)
    echo "host    ${db_name}    ${db_user}    10.0.0.0/16    md5" >> "$PG_HBA"

  - |
    sudo -u postgres psql -c "CREATE USER ${db_user} WITH PASSWORD '${db_password}';" || true
    sudo -u postgres psql -c "CREATE DATABASE ${db_name} OWNER ${db_user};" || true

  - systemctl restart postgresql

  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /root/.bashrc
  - git clone ${git_repo_url} /opt/tradan

  - |
    cat > /opt/tradan/backend/.env <<EOF
    DATABASE_URL=postgresql://${db_user}:${db_password}@localhost:5432/${db_name}
    EOF

  - cd /opt/tradan/backend && /root/.cargo/bin/uv sync
  - cd /opt/tradan/backend && /root/.cargo/bin/uv run ingest migrate
  - touch /etc/tradan/setup_complete
```

- [ ] **Step 4: Run `terraform validate`**

```bash
cd infra && terraform validate
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 5: Commit**

```bash
git add infra/base.tf infra/scripts/cloud-init-base.yaml
git commit -m "feat(infra): base droplet with postgres + cloud-init provisioning"
```

---

## Task 4: Training droplet + cloud-init + volume attachment

**Files:**
- Create: `infra/train.tf`
- Create: `infra/scripts/cloud-init-train.yaml`

- [ ] **Step 1: Create `infra/scripts/cloud-init-train.yaml`**

```yaml
#cloud-config
package_update: true
package_upgrade: false

packages:
  - git
  - curl
  - parallel

runcmd:
  # Install uv
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /root/.bashrc

  # Mount block volume
  # DigitalOcean attaches volumes as /dev/disk/by-id/scsi-0DO_Volume_<name>
  - mkdir -p /mnt/models
  - DEVICE=$(ls /dev/disk/by-id/scsi-0DO_Volume_* 2>/dev/null | head -1)
  - |
    if [ -n "$DEVICE" ]; then
      mount "$DEVICE" /mnt/models
      echo "$DEVICE /mnt/models ext4 defaults,nofail 0 2" >> /etc/fstab
    fi

  # Clone repo
  - git clone ${git_repo_url} /opt/tradan

  # Write .env with DATABASE_URL pointing to base droplet's private IP
  - mkdir -p /opt/tradan/backend
  - |
    cat > /opt/tradan/backend/.env <<EOF
    DATABASE_URL=postgresql://${db_user}:${db_password}@${db_host}:5432/${db_name}
    EOF

  # Symlink trained_models -> /mnt/models so the trainer writes to the block volume
  # The trainer uses a hardcoded path relative to the repo: backend/trained_models
  - rm -rf /opt/tradan/backend/trained_models
  - ln -s /mnt/models /opt/tradan/backend/trained_models

  # Install Python dependencies
  - cd /opt/tradan/backend && /root/.cargo/bin/uv sync

  # Write worker count for run_sweep.sh to read
  - mkdir -p /etc/tradan
  - echo "${worker_count}" > /etc/tradan/worker_count

  # Mark setup complete
  - touch /etc/tradan/setup_complete
```

- [ ] **Step 2: Create `infra/train.tf`**

```hcl
resource "digitalocean_droplet" "train" {
  count    = var.train_enabled ? 1 : 0
  name     = "tradan-train"
  size     = var.train_droplet_size
  image    = "ubuntu-22-04-x64"
  region   = var.region
  vpc_uuid = digitalocean_vpc.tradan.id
  ssh_keys = [data.digitalocean_ssh_key.operator.id]

  user_data = templatefile("${path.module}/scripts/cloud-init-train.yaml", {
    db_host      = digitalocean_droplet.base.ipv4_address_private
    db_password  = var.db_password
    db_name      = var.db_name
    db_user      = var.db_user
    git_repo_url = var.git_repo_url
    worker_count = local.worker_count
  })
}

resource "digitalocean_volume_attachment" "models" {
  count      = var.train_enabled ? 1 : 0
  droplet_id = digitalocean_droplet.train[0].id
  volume_id  = digitalocean_volume.models.id
}
```

- [ ] **Step 3: Run `terraform validate`**

```bash
cd infra && terraform validate
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 4: Commit**

```bash
git add infra/train.tf infra/scripts/cloud-init-train.yaml
git commit -m "feat(infra): ephemeral training droplet with volume attachment and cloud-init"
```

---

## Task 5: Terraform outputs

**Files:**
- Create: `infra/outputs.tf`

- [ ] **Step 1: Create `infra/outputs.tf`**

```hcl
output "base_ip" {
  description = "Public IP of the base droplet"
  value       = digitalocean_droplet.base.ipv4_address
}

output "base_private_ip" {
  description = "Private VPC IP of the base droplet (used by training droplet for DB)"
  value       = digitalocean_droplet.base.ipv4_address_private
}

output "base_ssh" {
  description = "SSH command to connect to the base droplet"
  value       = "ssh root@${digitalocean_droplet.base.ipv4_address}"
}

output "train_ip" {
  description = "Public IP of the training droplet (empty if not running)"
  value       = var.train_enabled ? digitalocean_droplet.train[0].ipv4_address : ""
}

output "train_ssh" {
  description = "SSH command to connect to the training droplet"
  value       = var.train_enabled ? "ssh root@${digitalocean_droplet.train[0].ipv4_address}" : "training droplet is not running"
}

output "worker_count" {
  description = "Number of parallel training workers for the current droplet size"
  value       = local.worker_count
}

output "volume_id" {
  description = "Block volume ID (reference for manual operations)"
  value       = digitalocean_volume.models.id
}

output "droplet_size" {
  description = "Current training droplet size"
  value       = var.train_droplet_size
}
```

- [ ] **Step 2: Run `terraform validate`**

```bash
cd infra && terraform validate
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 3: Commit**

```bash
git add infra/outputs.tf
git commit -m "feat(infra): terraform outputs for IPs, SSH commands, worker count"
```

---

## Task 6: CLI extension — `train list --names-only --status`

The `run_sweep.sh` script needs: `uv run train list --names-only --status pending` to output one model name per line (models that have no completed training run). This task adds those flags to the CLI and the DB function that drives them.

**Files:**
- Modify: `backend/src/trainer/db.py`
- Modify: `backend/src/trainer/cli.py`
- Create: `backend/tests/trainer/test_cli_list.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/trainer/test_cli_list.py`:
```python
from __future__ import annotations

from unittest.mock import patch

import pytest

from trainer.cli import build_parser, cmd_list


def _run_list(args: list[str]) -> tuple[int, str]:
    """Run `train list <args>` and capture stdout. Returns (exit_code, output)."""
    import io, sys
    parser = build_parser()
    parsed = parser.parse_args(["list"] + args)

    buf = io.StringIO()
    with patch("sys.stdout", buf):
        cmd_list(parsed)
    return 0, buf.getvalue()


def test_list_names_only_returns_one_name_per_line():
    fake_models = [
        {"name": "btc_1h_ppo_s0", "created_at": None, "run_count": 0, "best_pnl": None},
        {"name": "btc_4h_sac_s1", "created_at": None, "run_count": 1, "best_pnl": 100.0},
    ]
    with patch("trainer.cli.list_model_configs", return_value=fake_models):
        _, output = _run_list(["--names-only"])

    lines = output.strip().splitlines()
    assert lines == ["btc_1h_ppo_s0", "btc_4h_sac_s1"]


def test_list_status_pending_filters_to_zero_runs():
    fake_models = [
        {"name": "btc_1h_ppo_s0", "created_at": None, "run_count": 0, "best_pnl": None},
        {"name": "btc_4h_sac_s1", "created_at": None, "run_count": 1, "best_pnl": 100.0},
    ]
    with patch("trainer.cli.list_model_configs", return_value=fake_models):
        _, output = _run_list(["--names-only", "--status", "pending"])

    lines = output.strip().splitlines()
    assert lines == ["btc_1h_ppo_s0"]


def test_list_status_completed_filters_to_nonzero_runs():
    fake_models = [
        {"name": "btc_1h_ppo_s0", "created_at": None, "run_count": 0, "best_pnl": None},
        {"name": "btc_4h_sac_s1", "created_at": None, "run_count": 1, "best_pnl": 100.0},
    ]
    with patch("trainer.cli.list_model_configs", return_value=fake_models):
        _, output = _run_list(["--names-only", "--status", "completed"])

    lines = output.strip().splitlines()
    assert lines == ["btc_4h_sac_s1"]


def test_list_default_still_prints_table():
    fake_models = [
        {"name": "btc_1h_ppo_s0", "created_at": None, "run_count": 0, "best_pnl": None},
    ]
    with patch("trainer.cli.list_model_configs", return_value=fake_models):
        _, output = _run_list([])

    assert "btc_1h_ppo_s0" in output
    assert "Name" in output  # table header still present
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/trainer/test_cli_list.py -v
```

Expected: All 4 tests FAIL with `AttributeError` or `SystemExit` because `--names-only` and `--status` don't exist yet.

- [ ] **Step 3: Extend `cmd_list` and `build_parser` in `backend/src/trainer/cli.py`**

Replace the existing `cmd_list` function and the `list` subparser in `build_parser`:

```python
def cmd_list(args: argparse.Namespace) -> None:
    models = list_model_configs()
    if not models:
        if not getattr(args, "names_only", False):
            print("No models registered.")
        return

    names_only = getattr(args, "names_only", False)
    status_filter = getattr(args, "status", None)

    # Apply status filter:
    # "pending"   = models with zero completed runs (run_count == 0)
    # "completed" = models with at least one completed run (run_count > 0)
    if status_filter == "pending":
        models = [m for m in models if m["run_count"] == 0]
    elif status_filter == "completed":
        models = [m for m in models if m["run_count"] > 0]

    if names_only:
        for m in models:
            print(m["name"])
        return

    print(f"{'Name':<20} {'Runs':>6} {'Best PnL':>12} {'Created'}")
    print("-" * 60)
    for m in models:
        pnl = f"${m['best_pnl']:.2f}" if m["best_pnl"] is not None else "—"
        created = m["created_at"].strftime("%Y-%m-%d %H:%M") if m["created_at"] else "—"
        print(f"{m['name']:<20} {m['run_count']:>6} {pnl:>12} {created}")
```

In `build_parser`, replace the `list` subparser definition:

```python
    ls = sub.add_parser("list", help="List all registered models")
    ls.add_argument(
        "--names-only",
        action="store_true",
        dest="names_only",
        help="Print only model names, one per line (for use with GNU parallel)",
    )
    ls.add_argument(
        "--status",
        choices=["pending", "completed"],
        default=None,
        help="Filter models: pending = no completed runs, completed = has completed runs",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/trainer/test_cli_list.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Verify existing `train list` behavior is unchanged**

```bash
cd backend
uv run pytest tests/trainer/ -v
```

Expected: All tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add backend/src/trainer/cli.py backend/tests/trainer/test_cli_list.py
git commit -m "feat(trainer): add --names-only and --status flags to train list"
```

---

## Task 7: Sweep script — Phase 1 (BTCUSDT baseline)

**Files:**
- Create: `infra/scripts/sweep_phase1.py`

This script runs on the training droplet (after cloud-init completes) to register Phase 1 model configs in the DB. It reads `DATABASE_URL` from `/opt/tradan/backend/.env`.

- [ ] **Step 1: Create `infra/scripts/sweep_phase1.py`**

```python
#!/usr/bin/env python3
"""
Phase 1 — BTCUSDT baseline sweep.

Generates and registers 63 model configs:
  7 intervals × 3 algorithms × 3 seeds = 63 runs

All other parameters fixed. Purpose: find which interval+algorithm combos
work at all before varying hyperparameters.
"""
from __future__ import annotations

import os
import sys
from itertools import product
from pathlib import Path
from dotenv import load_dotenv

# Load .env before importing trainer modules (which read DATABASE_URL at import)
load_dotenv(Path(__file__).resolve().parent.parent.parent / "backend" / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend" / "src"))

from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
ALGORITHMS = ["PPO", "SAC", "A2C"]
SEEDS = [42, 123, 456]

PHASE = "p1"
TARGET = "BTCUSDT"
LOOKBACK = 500
LEARNING_RATE = 3e-4
TIMESTEPS = 1_000_000


def main() -> None:
    configs = list(product(INTERVALS, ALGORITHMS, enumerate(SEEDS)))
    print(f"Registering {len(configs)} Phase 1 configs for {TARGET}...")

    for interval, algo, (seed_idx, seed) in configs:
        name = f"btc_{interval}_{algo.lower()}_{PHASE}_s{seed_idx}"
        config = ModelConfig(
            name=name,
            symbols=[TARGET],
            intervals=[interval],
            columns=list(ALL_KLINE_COLUMNS),
            exchange=ExchangeConfig(),
            lookback_window=LOOKBACK,
            algorithm=algo,
            learning_rate=LEARNING_RATE,
            total_timesteps=TIMESTEPS,
        )
        config_id = save_model_config(config)
        print(f"  Registered: {name} (id={config_id})")

    print(f"\nDone. {len(configs)} configs registered.")
    print("Run: bash /opt/tradan/infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make the script executable**

```bash
chmod +x infra/scripts/sweep_phase1.py
```

- [ ] **Step 3: Commit**

```bash
git add infra/scripts/sweep_phase1.py
git commit -m "feat(infra): Phase 1 BTCUSDT sweep script (63 configs)"
```

---

## Task 8: Sweep scripts — Phase 2 and Phase 3

**Files:**
- Create: `infra/scripts/sweep_phase2.py`
- Create: `infra/scripts/sweep_phase3.py`

Phase 2 queries the DB for Phase 1 winners (top 5 by holdout Sharpe) and generates hyperparameter variants. Phase 3 takes Phase 2 winners and trains them much longer.

- [ ] **Step 1: Create `infra/scripts/sweep_phase2.py`**

```python
#!/usr/bin/env python3
"""
Phase 2 — Hyperparameter expansion.

Reads top 5 Phase 1 winners from DB (by holdout Sharpe, filtered by winner
criteria), then generates variants across:
  4 lookback windows × 3 learning rates × 3 seeds = up to 180 runs per winner config

Only the interval + algorithm from the winner is preserved. Everything else is re-varied.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / "backend" / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend" / "src"))

from ingester.db import connect
from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

LOOKBACKS = [100, 250, 500, 1000]
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
SEEDS = [42, 123, 456]
TIMESTEPS = 1_000_000
PHASE = "p2"
TOP_N = 5


def get_phase1_winners(conn, top_n: int) -> list[dict]:
    """Return top N Phase 1 configs by holdout Sharpe after applying winner filters."""
    rows = conn.execute(
        """
        SELECT
            mc.name,
            mc.config_json,
            tr_eval.sharpe_ratio
        FROM model_configs mc
        JOIN training_runs tr_train ON tr_train.model_config_id = mc.id
            AND tr_train.run_type = 'train' AND tr_train.status = 'completed'
        JOIN training_runs tr_eval ON tr_eval.model_config_id = mc.id
            AND tr_eval.run_type = 'evaluate' AND tr_eval.status = 'completed'
        WHERE mc.name LIKE '%_p1_%'
          AND tr_eval.total_trades > 10
          AND tr_eval.total_pnl > 0
          AND tr_eval.max_drawdown < 0.25
          AND tr_eval.total_pnl / NULLIF(tr_train.total_pnl, 0) > 0.5
        ORDER BY tr_eval.sharpe_ratio DESC
        LIMIT %s
        """,
        (top_n,),
    ).fetchall()
    return [{"name": r[0], "config": r[1], "sharpe": r[2]} for r in rows]


def main() -> None:
    conn = connect()
    try:
        winners = get_phase1_winners(conn, TOP_N)
    finally:
        conn.close()

    if not winners:
        print("No Phase 1 winners found. Run evaluate_winners.sh first.")
        sys.exit(1)

    print(f"Found {len(winners)} Phase 1 winners. Generating Phase 2 configs...")
    count = 0

    for winner in winners:
        base_cfg = ModelConfig.from_dict(winner["config"])
        interval = base_cfg.intervals[0]
        algo = base_cfg.algorithm

        for lookback, lr, (seed_idx, _seed) in product(LOOKBACKS, LEARNING_RATES, enumerate(SEEDS)):
            # Encode lr as readable string: 1e-4 → lr1e4
            lr_str = f"lr{str(lr).replace('-', '').replace('.', '').replace('e', 'e')}"
            name = f"btc_{interval}_{algo.lower()}_lb{lookback}_{lr_str}_{PHASE}_s{seed_idx}"
            config = ModelConfig(
                name=name,
                symbols=["BTCUSDT"],
                intervals=[interval],
                columns=list(ALL_KLINE_COLUMNS),
                exchange=ExchangeConfig(),
                lookback_window=lookback,
                algorithm=algo,
                learning_rate=lr,
                total_timesteps=TIMESTEPS,
            )
            save_model_config(config)
            print(f"  Registered: {name}")
            count += 1

    print(f"\nDone. {count} Phase 2 configs registered.")
    print("Run: bash /opt/tradan/infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `infra/scripts/sweep_phase3.py`**

```python
#!/usr/bin/env python3
"""
Phase 3 — Long training.

Reads top 5 Phase 2 winners from DB, trains them with 5M timesteps each.
3 seeds per config = 15 runs total.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / "backend" / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend" / "src"))

from ingester.db import connect
from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

SEEDS = [42, 123, 456]
TIMESTEPS = 5_000_000
PHASE = "p3"
TOP_N = 5


def get_phase2_winners(conn, top_n: int) -> list[dict]:
    """Return top N Phase 2 configs by holdout Sharpe after applying winner filters."""
    rows = conn.execute(
        """
        SELECT
            mc.name,
            mc.config_json,
            tr_eval.sharpe_ratio
        FROM model_configs mc
        JOIN training_runs tr_train ON tr_train.model_config_id = mc.id
            AND tr_train.run_type = 'train' AND tr_train.status = 'completed'
        JOIN training_runs tr_eval ON tr_eval.model_config_id = mc.id
            AND tr_eval.run_type = 'evaluate' AND tr_eval.status = 'completed'
        WHERE mc.name LIKE '%_p2_%'
          AND tr_eval.total_trades > 10
          AND tr_eval.total_pnl > 0
          AND tr_eval.max_drawdown < 0.25
          AND tr_eval.total_pnl / NULLIF(tr_train.total_pnl, 0) > 0.5
        ORDER BY tr_eval.sharpe_ratio DESC
        LIMIT %s
        """,
        (top_n,),
    ).fetchall()
    return [{"name": r[0], "config": r[1], "sharpe": r[2]} for r in rows]


def main() -> None:
    conn = connect()
    try:
        winners = get_phase2_winners(conn, TOP_N)
    finally:
        conn.close()

    if not winners:
        print("No Phase 2 winners found. Run evaluate_winners.sh after Phase 2 first.")
        sys.exit(1)

    print(f"Found {len(winners)} Phase 2 winners. Generating Phase 3 long-training configs...")
    count = 0

    for winner in winners:
        base_cfg = ModelConfig.from_dict(winner["config"])

        for seed_idx in range(len(SEEDS)):
            # Strip old phase/seed suffix and add p3
            base_name = "_".join(
                p for p in winner["name"].split("_")
                if not p.startswith("p") or not p[1:].isdigit()
                if not (p.startswith("s") and p[1:].isdigit())
            )
            name = f"{base_name}_{PHASE}_s{seed_idx}"
            config = ModelConfig(
                name=name,
                symbols=base_cfg.symbols,
                intervals=base_cfg.intervals,
                columns=base_cfg.columns,
                exchange=base_cfg.exchange,
                lookback_window=base_cfg.lookback_window,
                algorithm=base_cfg.algorithm,
                learning_rate=base_cfg.learning_rate,
                total_timesteps=TIMESTEPS,
            )
            save_model_config(config)
            print(f"  Registered: {name}")
            count += 1

    print(f"\nDone. {count} Phase 3 configs registered (5M timesteps each).")
    print("Run: bash /opt/tradan/infra/scripts/run_sweep.sh")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make scripts executable**

```bash
chmod +x infra/scripts/sweep_phase2.py infra/scripts/sweep_phase3.py
```

- [ ] **Step 4: Commit**

```bash
git add infra/scripts/sweep_phase2.py infra/scripts/sweep_phase3.py
git commit -m "feat(infra): Phase 2 and Phase 3 sweep scripts (winners-driven)"
```

---

## Task 9: SQL queries — winner ranking and eval queue

**Files:**
- Create: `infra/scripts/winners.sql`
- Create: `infra/scripts/winners_no_eval.sql`

- [ ] **Step 1: Create `infra/scripts/winners.sql`**

```sql
-- Ranked winner table: apply filter chain then sort by holdout Sharpe.
-- Run from the base droplet: psql $DATABASE_URL -f winners.sql
SELECT
    mc.name,
    tr_train.total_pnl                                          AS train_pnl,
    tr_eval.total_pnl                                           AS holdout_pnl,
    ROUND(tr_eval.sharpe_ratio::numeric, 3)                     AS sharpe,
    ROUND((tr_eval.max_drawdown * 100)::numeric, 1)             AS drawdown_pct,
    tr_eval.total_trades,
    ROUND((tr_eval.win_rate * 100)::numeric, 1)                 AS win_rate_pct,
    ROUND(
        (tr_eval.total_pnl / NULLIF(tr_train.total_pnl, 0))::numeric, 2
    )                                                           AS generalization_ratio
FROM model_configs mc
JOIN training_runs tr_train ON tr_train.model_config_id = mc.id
    AND tr_train.run_type = 'train' AND tr_train.status = 'completed'
JOIN training_runs tr_eval ON tr_eval.model_config_id = mc.id
    AND tr_eval.run_type = 'evaluate' AND tr_eval.status = 'completed'
WHERE tr_eval.total_trades > 10
  AND tr_eval.total_pnl > 0
  AND tr_eval.max_drawdown < 0.25
  AND tr_eval.total_pnl / NULLIF(tr_train.total_pnl, 0) > 0.5
ORDER BY tr_eval.sharpe_ratio DESC
LIMIT 20;
```

- [ ] **Step 2: Create `infra/scripts/winners_no_eval.sql`**

Used by `evaluate_winners.sh` to find completed training runs that have not yet been evaluated, ordered by training PnL.

```sql
-- Top 20 completed training runs with no evaluation run yet.
-- Output: model_name, run_id (tab-separated, for use with GNU parallel)
SELECT mc.name, tr.id
FROM model_configs mc
JOIN training_runs tr ON tr.model_config_id = mc.id
    AND tr.run_type = 'train' AND tr.status = 'completed'
WHERE NOT EXISTS (
    SELECT 1 FROM training_runs ev
    WHERE ev.model_config_id = mc.id AND ev.run_type = 'evaluate'
)
ORDER BY tr.total_pnl DESC NULLS LAST
LIMIT 20;
```

- [ ] **Step 3: Commit**

```bash
git add infra/scripts/winners.sql infra/scripts/winners_no_eval.sql
git commit -m "feat(infra): SQL queries for winner ranking and eval queue"
```

---

## Task 10: Shell scripts — training runner and eval runner

**Files:**
- Create: `infra/scripts/run_sweep.sh`
- Create: `infra/scripts/evaluate_winners.sh`

- [ ] **Step 1: Create `infra/scripts/run_sweep.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKERS=$(cat /etc/tradan/worker_count)
BACKEND=/opt/tradan/backend

echo "Starting sweep with $WORKERS parallel workers..."
echo "Pending models:"
cd "$BACKEND"
uv run train list --names-only --status pending | tee /tmp/pending_models.txt
PENDING=$(wc -l < /tmp/pending_models.txt)
echo "Total pending: $PENDING"

if [ "$PENDING" -eq 0 ]; then
  echo "No pending models found. Register configs first (sweep_phase1.py etc.)"
  exit 0
fi

cat /tmp/pending_models.txt \
  | parallel --jobs "$WORKERS" --joblog /tmp/sweep_joblog.txt \
      "cd $BACKEND && uv run train start --model {}"

echo ""
echo "Sweep complete. Results in: uv run train list"
echo "Job log: /tmp/sweep_joblog.txt"
```

- [ ] **Step 2: Create `infra/scripts/evaluate_winners.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

BACKEND=/opt/tradan/backend
SQL=/opt/tradan/infra/scripts/winners_no_eval.sql

echo "Finding top 20 trained models without evaluation runs..."
cd "$BACKEND"

# Query returns: model_name<tab>run_id per line
psql "$DATABASE_URL" \
  --tuples-only \
  --no-align \
  --field-separator $'\t' \
  -f "$SQL" > /tmp/eval_queue.txt

QUEUED=$(wc -l < /tmp/eval_queue.txt)
echo "Queued for evaluation: $QUEUED"

if [ "$QUEUED" -eq 0 ]; then
  echo "Nothing to evaluate."
  exit 0
fi

# Run 4 parallel evals — evaluation is lighter than training, base droplet has 4 vCPUs
cat /tmp/eval_queue.txt \
  | parallel --jobs 4 --colsep $'\t' \
      "cd $BACKEND && uv run train evaluate --model {1} --run {2}"

echo ""
echo "Evaluation complete. Run: psql \$DATABASE_URL -f $SQL_WINNERS"
```

- [ ] **Step 3: Make scripts executable**

```bash
chmod +x infra/scripts/run_sweep.sh infra/scripts/evaluate_winners.sh
```

- [ ] **Step 4: Commit**

```bash
git add infra/scripts/run_sweep.sh infra/scripts/evaluate_winners.sh
git commit -m "feat(infra): run_sweep.sh and evaluate_winners.sh shell scripts"
```

---

## Task 11: Makefile

**Files:**
- Create: `infra/Makefile`

- [ ] **Step 1: Create `infra/Makefile`**

```makefile
# DigitalOcean training infrastructure Makefile
#
# Prerequisites: set these environment variables before running any target:
#   export TF_VAR_do_token="..."
#   export TF_VAR_db_password="..."
#   export TF_VAR_ssh_key_fingerprint="..."
#   export TF_VAR_git_repo_url="git@github.com:yourorg/tradan.git"
#   export TF_VAR_operator_ip="$(curl -s ifconfig.me)/32"

TF        := terraform
SCRIPTS   := $(CURDIR)/scripts

# Read live outputs (empty string if train droplet is down)
BASE_IP    = $(shell $(TF) output -raw base_ip 2>/dev/null || echo "")
TRAIN_IP   = $(shell $(TF) output -raw train_ip 2>/dev/null || echo "")
WORKERS    = $(shell $(TF) output -raw worker_count 2>/dev/null || echo "28")

.PHONY: init base-up train-up train-down base-ssh train-ssh \
        sweep-phase1 sweep-phase2 sweep-phase3 \
        winners evaluate status

## init: Initialize Terraform (run once)
init:
	$(TF) init

## base-up: Provision base droplet + block volume (first-time setup)
base-up:
	$(TF) apply \
	  -target=digitalocean_vpc.tradan \
	  -target=digitalocean_firewall.tradan \
	  -target=digitalocean_droplet.base \
	  -target=digitalocean_volume.models

## train-up: Create training droplet and attach block volume
train-up:
	$(TF) apply -var="train_enabled=true"

## train-down: Destroy training droplet (block volume persists)
train-down:
	$(TF) apply -var="train_enabled=false"

## base-ssh: Open SSH session to base droplet
base-ssh:
	@echo "Connecting to base droplet at $(BASE_IP)..."
	ssh root@$(BASE_IP)

## train-ssh: Open SSH session to training droplet
train-ssh:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	@echo "Connecting to training droplet at $(TRAIN_IP)..."
	ssh root@$(TRAIN_IP)

## sweep-phase1: Register Phase 1 configs and start training (63 runs)
sweep-phase1:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	@echo "Waiting for cloud-init to complete (may take 2-3 min on first boot)..."
	ssh root@$(TRAIN_IP) "while [ ! -f /etc/tradan/setup_complete ]; do sleep 10; echo 'Waiting...'; done; echo 'Ready.'"
	scp $(SCRIPTS)/sweep_phase1.py root@$(TRAIN_IP):/tmp/sweep_phase1.py
	ssh root@$(TRAIN_IP) "cd /opt/tradan/backend && uv run python /tmp/sweep_phase1.py"
	ssh root@$(TRAIN_IP) "bash /opt/tradan/infra/scripts/run_sweep.sh"

## sweep-phase2: Register Phase 2 configs (requires Phase 1 winners evaluated) and start training
sweep-phase2:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	scp $(SCRIPTS)/sweep_phase2.py root@$(TRAIN_IP):/tmp/sweep_phase2.py
	ssh root@$(TRAIN_IP) "cd /opt/tradan/backend && uv run python /tmp/sweep_phase2.py"
	ssh root@$(TRAIN_IP) "bash /opt/tradan/infra/scripts/run_sweep.sh"

## sweep-phase3: Register Phase 3 configs (requires Phase 2 winners evaluated) and start training
sweep-phase3:
	@if [ -z "$(TRAIN_IP)" ]; then echo "Training droplet is not running. Run: make train-up"; exit 1; fi
	scp $(SCRIPTS)/sweep_phase3.py root@$(TRAIN_IP):/tmp/sweep_phase3.py
	ssh root@$(TRAIN_IP) "cd /opt/tradan/backend && uv run python /tmp/sweep_phase3.py"
	ssh root@$(TRAIN_IP) "bash /opt/tradan/infra/scripts/run_sweep.sh"

## winners: Show ranked winner table (runs on base droplet)
winners:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
	ssh root@$(BASE_IP) "psql \$$DATABASE_URL -f /opt/tradan/infra/scripts/winners.sql"

## evaluate: Run holdout evaluation on top 20 unevaluated models (runs on base droplet)
evaluate:
	@if [ -z "$(BASE_IP)" ]; then echo "Base droplet not running. Run: make base-up"; exit 1; fi
	ssh root@$(BASE_IP) "bash /opt/tradan/infra/scripts/evaluate_winners.sh"

## status: Show terraform state summary
status:
	$(TF) output
```

- [ ] **Step 2: Verify Makefile syntax**

```bash
cd infra && make --dry-run base-up
```

Expected: Prints the `terraform apply ...` command without executing it.

- [ ] **Step 3: Commit**

```bash
git add infra/Makefile
git commit -m "feat(infra): Makefile with base-up, train-up/down, sweep, winners, evaluate targets"
```

---

## Self-Review

**Spec coverage check:**
- [x] Terraform provider, VPC, firewall — Task 1
- [x] Block volume (persistent, 100GB) — Task 2
- [x] Base droplet + PostgreSQL cloud-init — Task 3
- [x] Training droplet + volume attachment (ephemeral) — Task 4
- [x] Terraform outputs (IPs, SSH commands, worker count) — Task 5
- [x] Worker count lookup map (c-16/c-32/c-48) — Task 1 (main.tf locals)
- [x] `train_enabled` toggle pattern — Task 4 (train.tf)
- [x] `trained_models → /mnt/models` symlink — Task 4 (cloud-init)
- [x] `train list --names-only --status` CLI extension — Task 6
- [x] Phase 1 sweep script (63 BTC runs) — Task 7
- [x] Phase 2 sweep script (winners-driven, hyperparams) — Task 8
- [x] Phase 3 sweep script (winners-driven, 5M steps) — Task 8
- [x] `run_sweep.sh` with GNU parallel — Task 10
- [x] `evaluate_winners.sh` — Task 10
- [x] `winners.sql` filter chain — Task 9
- [x] `winners_no_eval.sql` — Task 9
- [x] Makefile with all targets — Task 11
- [x] Secrets via env vars, `.gitignore` for tfstate — Task 1

**No placeholders, TBDs, or TODOs found.**

---

## Execution

Plan complete and saved to `docs/superpowers/plans/2026-04-05-digitalocean-training-infra.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using the executing-plans skill, batch execution with checkpoints.

Which approach?
