# Training Infrastructure

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
make train-up   # spin up c-32 training droplet ($1/hr)
make sweep-phase1   # register 63 configs + fire up 28 workers
# ... wait ~5h ...
make evaluate       # eval winners on base droplet
make winners        # print ranked table
make train-down 
```

DigitalOcean infrastructure for training, evaluating, and storing RL trading models. Managed with Terraform; operated with `make`.

## Architecture

```
                       VPC (NYC3, private network)
                              │
         ┌────────────────────┼────────────────────┐
         │                                          │
 ┌───────▼───────┐                        ┌────────▼────────┐
 │  base droplet │  ← always-on           │ train droplet   │  ← ephemeral
 │  s-4vcpu-8gb  │                        │ c-32 (default)  │
 │  ~$24/mo      │                        │ $1.00/hr        │
 │               │                        │                 │
 │  PostgreSQL   │  ← private IP only     │ 28 workers      │
 │  eval / ops   │                        │ GNU parallel    │
 └───────────────┘                        └────────┬────────┘
                                                   │ attach/detach
                                          ┌────────▼────────┐
                                          │  Block Volume   │  ← persistent, never destroyed
                                          │  100 GB         │
                                          │  /mnt/models    │
                                          │  ~$10/mo        │
                                          └─────────────────┘
```

**Base droplet** runs PostgreSQL and serves as the ops node (run evaluations, query results, inspect DB). It is always on.

**Training droplet** is ephemeral — you create it when you want to train and destroy it when you're done. It costs money only while running.

**Block volume** persists across all training droplet lifecycles. Trained model `.zip` files live here and survive droplet destruction.

## Prerequisites

### 1. Install Terraform

```bash
brew install terraform
```

### 2. Create a DigitalOcean API token

Go to [DigitalOcean API settings](https://cloud.digitalocean.com/account/api/tokens) → Generate New Token → Read + Write scope.

### 3. Upload your SSH public key to DigitalOcean

Go to [Settings → Security → SSH Keys](https://cloud.digitalocean.com/account/security) → Add SSH Key.

After uploading, note the **name** you gave it (the label shown in the list, e.g. `mardo-macbook`).

### 4. Export environment variables

```bash
export TF_VAR_do_token="your-digitalocean-api-token"
export TF_VAR_db_password="choose-a-strong-password"
export TF_VAR_ssh_key_name="mardo-macbook"   # the name shown in DO Settings → SSH Keys
export TF_VAR_git_repo_url="git@github.com:yourorg/tradan.git"
```

Add these to your shell profile (`~/.zshrc` or `~/.bashrc`) so they persist across sessions.

> Your SSH firewall rule is auto-detected from your current public IP on every `make` call — no need to export `operator_ip`. If your IP changes, just run `make update-ip`.

---

## First-time setup

### Step 1: Initialize Terraform

```bash
cd infra
make init
```

Downloads the DigitalOcean Terraform provider. Run once after cloning.

### Step 2: Provision the base droplet and block volume

```bash
make base-up
```

Creates:
- VPC and firewall rules
- Base droplet (`s-4vcpu-8gb`) with PostgreSQL — takes ~3 minutes to fully provision via cloud-init
- 100 GB block volume for model storage

**Cost: ~$34/mo ongoing** (base droplet + volume). This is your only recurring cost when not training.

### Step 3: Verify the base droplet is ready

```bash
make base-ssh
```

SSH into the base droplet. Check that PostgreSQL and the repo are ready:

```bash
# On the base droplet:
systemctl status postgresql
ls /opt/tradan
cat /etc/tradan/setup_complete   # exists when cloud-init finished
```

If `setup_complete` doesn't exist yet, cloud-init is still running. Wait a minute and try again.

---

## Running a training batch

### Step 1: Spin up the training droplet

```bash
make train-up
```

Creates a `c-32` CPU-optimized droplet (32 vCPUs, 64 GB RAM) and attaches the block volume. Takes ~2 minutes to provision. **Costs $1.00/hr while running.**

To use a different size:

```bash
# 16 vCPU / $0.50/hr / 14 workers
make train-up TF_VAR_train_droplet_size=c-16

# 48 vCPU / $1.50/hr / 44 workers
make train-up TF_VAR_train_droplet_size=c-48
```

### Step 2: Run Phase 1 — Baseline sweep (63 runs)

```bash
make sweep-phase1
```

This command:
1. Waits for cloud-init to finish on the training droplet (up to ~3 min)
2. Registers 63 BTCUSDT model configs in the DB: `7 intervals × 3 algorithms × 3 seeds`
3. Starts 28 parallel training workers via GNU parallel

**Expected time: ~4–5 hours. Estimated cost: ~$5.**

You can monitor progress from another terminal:

```bash
make train-ssh
# On the training droplet:
tail -f /tmp/sweep_joblog.txt
```

### Step 3: Evaluate the trained models

Once training is done, run holdout evaluation on the top 20 models:

```bash
make evaluate
```

This runs on the base droplet (no training droplet needed). It evaluates each model against the 20% holdout data it has never seen.

### Step 4: See the winners

```bash
make winners
```

Prints a ranked table filtered by:
- `total_trades > 10` — not a "do nothing" model
- `holdout_pnl > 0` — profitable on unseen data
- `max_drawdown < 25%` — not recklessly risky
- `holdout_pnl / train_pnl > 0.5` — not overfit
- Ranked by Sharpe ratio

Example output:

```
 name                      | train_pnl | holdout_pnl | sharpe | drawdown_pct | total_trades | win_rate_pct | generalization_ratio
---------------------------+-----------+-------------+--------+--------------+--------------+--------------+---------------------
 btc_1h_ppo_p1_s2          |  12450.00 |     9870.00 |  1.842 |         18.3 |          234 |         57.3 |                 0.79
 btc_4h_ppo_p1_s0          |   8900.00 |     7100.00 |  1.611 |         21.5 |          187 |         54.1 |                 0.80
```

### Step 5: Destroy the training droplet

```bash
make train-down
```

Destroys the training droplet. **The block volume (and all trained models) are kept safe.** You stop paying $1.00/hr.

---

## Running Phase 2 and Phase 3

After Phase 1, you can run deeper hyperparameter sweeps on the winners.

### Phase 2 — Hyperparameter expansion (~180 runs)

Takes Phase 1 winners and varies lookback window (100/250/500/1000) and learning rate (1e-4/3e-4/1e-3):

```bash
make train-up            # spin up a fresh training droplet
make sweep-phase2        # register Phase 2 configs from Phase 1 winners + train
make train-down          # tear it down
make evaluate            # eval on base droplet
make winners             # see new rankings
```

### Phase 3 — Long training (~15 runs)

Takes Phase 2 winners and trains them with 5M timesteps (5× longer):

```bash
make train-up
make sweep-phase3        # register Phase 3 configs + train (takes ~12h)
make train-down
make evaluate
make winners             # final keeper models
```

---

## Tuning worker count

The default worker counts (c-16: 14, c-32: 28, c-48: 44) leave 2 vCPUs free for OS overhead. To push further:

```bash
make train-ssh
# On the training droplet:
htop                     # watch CPU and memory during a sweep
echo 30 > /etc/tradan/worker_count   # increase by 2 and rerun
bash /opt/tradan/infra/scripts/run_sweep.sh
```

Increase by 2 at a time. If you see swap usage or the droplet becomes sluggish, reduce by 2 and settle there.

---

## Makefile reference

| Command | What it does |
|---|---|
| `make init` | Initialize Terraform (run once) |
| `make base-up` | Provision base droplet + volume (first-time setup) |
| `make train-up` | Create training droplet + attach volume |
| `make train-down` | Destroy training droplet (volume persists) |
| `make update-ip` | Re-apply firewall with your current IP after IP change |
| `make base-ssh` | SSH into base droplet |
| `make train-ssh` | SSH into training droplet |
| `make sweep-phase1` | Register 63 Phase 1 configs + start training |
| `make sweep-phase2` | Register Phase 2 configs from Phase 1 winners + start training |
| `make sweep-phase3` | Register Phase 3 configs from Phase 2 winners + start training |
| `make evaluate` | Run holdout eval on top 20 unevaluated models (base droplet) |
| `make winners` | Print ranked winner table |
| `make status` | Show Terraform outputs (IPs, worker count, droplet size) |

---

## Cost summary

| Component | When | Cost |
|---|---|---|
| Base droplet (`s-4vcpu-8gb`) | Always | $24/mo |
| Block volume (100 GB) | Always | $10/mo |
| Training droplet (`c-32`) | While training | $1.00/hr |
| Phase 1 compute (63 runs, ~5h) | One-time | ~$5 |
| Phase 2 compute (~180 runs, ~7h) | One-time | ~$7 |
| Phase 3 compute (~15 runs, ~12h) | One-time | ~$12 |

**Ongoing base cost: ~$34/mo.** Training phases cost ~$24 total compute.

---

## File structure

```
infra/
├── main.tf                    # Provider, VPC, firewall, worker count locals
├── variables.tf               # All input variables
├── outputs.tf                 # IPs, SSH commands, worker count
├── base.tf                    # Always-on base droplet
├── train.tf                   # Ephemeral training droplet + volume attachment
├── volume.tf                  # Persistent 100 GB block volume
├── Makefile                   # Orchestration
└── scripts/
    ├── cloud-init-base.yaml   # Provisions base droplet on first boot
    ├── cloud-init-train.yaml  # Provisions training droplet on first boot
    ├── sweep_phase1.py        # Register 63 BTCUSDT baseline configs
    ├── sweep_phase2.py        # Register hyperparameter variants from Phase 1 winners
    ├── sweep_phase3.py        # Register long-training configs from Phase 2 winners
    ├── run_sweep.sh           # Fan out N parallel training workers
    ├── evaluate_winners.sh    # Run eval on top 20 unevaluated models
    ├── winners.sql            # Ranked winner query with filter chain
    └── winners_no_eval.sql    # Queue of models needing evaluation
```

---

## After Phase 3: continuing toward better models

After Phase 3 you have your best models. There are several directions depending on what you observe.

### Path 1: Resume training existing winners

SB3 supports loading a saved model and continuing training from where it left off — weights, optimizer state, and replay buffer all carry over. The current trainer always starts fresh, so this requires a small addition:

```python
# In backend/src/trainer/training/trainer.py
def resume_training(config: ModelConfig, model_path: str, additional_timesteps: int) -> int:
    algo_cls = ALGO_MAP[config.algorithm]
    # ... same setup as train_model ...
    model = algo_cls.load(model_path, env=env)   # load existing weights
    model.learn(total_timesteps=additional_timesteps, callback=[...])
    # ... save + record run ...
```

Then expose it as: `uv run train resume --model btc_1h_ppo_p3_s2 --run 47 --timesteps 5000000`

**When to use:** A Phase 3 winner is clearly profitable but Sharpe is still improving — it just needs more data exposure. Cheaper than restarting from scratch.

---

### Path 2: New Phase 4 sweep — unexplored config dimensions

You've covered intervals, algorithms, seeds, lookback, and learning rate. You haven't varied:

| Dimension | Values to try |
|---|---|
| **Column sets** | All 9 vs OHLCV-only (5) vs OHLC-only (4) — does removing volume help or hurt? |
| **SL/TP range** | `min_sl_pct`, `max_sl_pct`, `max_tp_pct` — tighter or looser action space |
| **Trigger offset** | `max_trigger_offset_pct` — how far limit orders can be placed from price |
| **Initial balance** | $1,000 vs $10,000 vs $100,000 — affects position sizing behavior |

Take your top 3 Phase 3 winners, apply these variations, 3 seeds each. A `sweep_phase4.py` follows the same pattern as the others: query the DB for Phase 3 winners, generate variants, register configs, run `run_sweep.sh`.

**When to use:** Phase 3 winners are good but you suspect the action space or reward signal is limiting them.

---

### Path 3: Expand to other symbols

Take your best BTCUSDT config (interval + algo + lookback + lr) and apply it to other symbols without modification:

```python
SYMBOLS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
# Use the exact ModelConfig of your Phase 3 winner, just swap symbols=[target]
```

Also test multi-symbol inputs — e.g., `symbols=["BTCUSDT", "SOLUSDT"]` to trade SOL while feeding BTC as a leading indicator.

**When to use:** You have a confident BTCUSDT winner and want to diversify or find a stronger signal on alts.

---

### Path 4: Fix the reward signal (if most models do nothing)

RL models often converge to never trading because that guarantees zero loss — technically optimal but useless. If you're still seeing many zero-trade models after Phase 3, the reward signal needs shaping.

**Option A — Inactivity penalty** in `backend/src/trainer/env/trading_env.py`:

```python
# In _calculate_reward() or step():
if self.exchange.total_trades == 0 and self.current_step > 100:
    reward -= 0.001  # small penalty for staying idle too long
```

**Option B — Episode-end bonus** for profitable activity:

```python
if done and self.exchange.total_trades > 10:
    reward += max(0, final_equity - self.config.initial_balance) * 0.01
```

Retrain your Phase 3 winners after this change — they may break out of the do-nothing local minimum.

---

### Decision tree

```
Phase 3 winners
      │
      ├─ Sharpe > 1.5, drawdown < 15%, still improving?
      │    → resume training at 10M steps (Path 1)
      │
      ├─ Good profits but too many zero-trade models?
      │    → reward shaping (Path 4), then retrain
      │
      ├─ Strong on BTC, want to diversify?
      │    → same config on other symbols (Path 3)
      │
      └─ Still room to explore?
           → vary column sets + SL/TP range in Phase 4 sweep (Path 2)
```
