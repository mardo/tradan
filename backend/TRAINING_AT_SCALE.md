# Training at Scale

How to train hundreds or thousands of model configurations, pick the winners, and what infrastructure you need.

---

## Table of Contents

1. [The Configuration Matrix](#the-configuration-matrix)
2. [Generating Configs Programmatically](#generating-configs-programmatically)
3. [Picking Winners](#picking-winners)
4. [Infrastructure & DevOps](#infrastructure--devops)
5. [Cost Estimates](#cost-estimates)
6. [Recommended Approach](#recommended-approach)

---

## The Configuration Matrix

Each model config is a combination of these dimensions:

| Dimension | Example Values | Count |
|---|---|---|
| **Target symbol** | BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT | 5 |
| **Input symbols** | single (same as target), multi (target + BTC), all 5 | 3 variants |
| **Interval** | 1m, 5m, 15m, 30m, 1h, 4h, 1d | 7 |
| **Columns** | all 9, OHLCV only (5), OHLC only (4) | 3 |
| **Algorithm** | PPO, SAC, A2C | 3 |
| **Lookback window** | 100, 250, 500, 1000 | 4 |
| **Learning rate** | 1e-4, 3e-4, 1e-3 | 3 |
| **Random seed** | 3-5 runs per config | 3-5 |

### How Big is the Matrix?

**Full grid search** (everything × everything):

```
5 targets × 3 symbol sets × 7 intervals × 3 columns × 3 algos × 4 lookbacks × 3 LRs × 3 seeds
= 5 × 3 × 7 × 3 × 3 × 4 × 3 × 3 = 34,020 training runs
```

That's too many. You need a strategy to reduce it.

### Phased Approach (Recommended)

**Phase 1 — Baseline sweep (≈150 runs)**
Fix most params, vary only the most impactful dimensions:
- 1 target (BTCUSDT)
- 1 symbol set (single)
- All 7 intervals
- All 9 columns (don't vary yet)
- 3 algorithms (PPO, SAC, A2C)
- Lookback 500 (default)
- Learning rate 3e-4 (default)
- 3 seeds each

```
7 intervals × 3 algos × 3 seeds = 63 runs
```

This tells you which interval + algorithm combinations work.

**Phase 2 — Expand winners (≈200 runs)**
Take the top 3 interval+algo combos from Phase 1, now vary:
- 5 target symbols
- 2 symbol sets (single, multi with BTC)
- 3 seeds each

```
3 combos × 5 targets × 2 symbol sets × 3 seeds = 90 runs
```

**Phase 3 — Hyperparameter tuning (≈300 runs)**
Take top 5 configs from Phase 2, now vary:
- 4 lookback windows
- 3 learning rates
- 3 column sets
- 3 seeds

```
5 configs × (4 + 3 + 3) × 3 seeds ≈ 150 runs
```

**Phase 4 — Long training (≈20 runs)**
Top 5 configs from Phase 3, train with 5-10M timesteps instead of 1M.

**Total: ≈400-500 runs** instead of 34,000.

---

## Generating Configs Programmatically

Instead of hand-writing each config file, you'd create a sweep script. Here's the pattern:

```python
# sweep.py — generate and register all configs for a phase
from itertools import product
from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig
from trainer.db import save_model_config

INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
ALGOS = ["PPO", "SAC", "A2C"]
SEEDS = [42, 123, 456]

for interval, algo in product(INTERVALS, ALGOS):
    for seed_idx, seed in enumerate(SEEDS):
        config = ModelConfig(
            name=f"btc_{interval}_{algo.lower()}_s{seed_idx}",
            symbols=["BTCUSDT"],
            intervals=[interval],
            columns=list(ALL_KLINE_COLUMNS),
            exchange=ExchangeConfig(),
            algorithm=algo,
            total_timesteps=1_000_000,
        )
        save_model_config(config)
        print(f"Registered: {config.name}")
```

Then train them all with a simple loop or job queue:

```bash
# Train all registered pending models
for model in $(uv run train list --names-only); do
    uv run train start --model "$model" &
done
```

Or better — use a proper job runner (see Infrastructure section).

---

## Picking Winners

### Metrics That Matter

| Metric | What It Tells You | Red Flag |
|---|---|---|
| **Total PnL** | Raw profitability | Negative = model loses money |
| **Sharpe Ratio** | Risk-adjusted return | < 0.5 = not worth the risk |
| **Max Drawdown** | Worst peak-to-trough loss | > 30% = too risky for real money |
| **Win Rate** | % of profitable trades | < 40% is concerning unless avg win >> avg loss |
| **Total Trades** | Activity level | 0 = model learned to do nothing (common!) |
| **Holdout PnL** | Performance on unseen data | Much worse than training PnL = overfitting |

### The Selection Process

```
                    All Trained Runs
                          │
                 Filter: total_trades > 10
                          │
                 Filter: holdout PnL > 0
                          │
              Rank by: holdout Sharpe Ratio
                          │
                  Top 20 candidates
                          │
         Check: max_drawdown < 25%
                          │
         Check: holdout PnL / training PnL > 0.5
         (if holdout is less than half of training = overfit)
                          │
              Final 5-10 winners
                          │
         Phase 4: retrain with more timesteps
```

### Overfitting Detection

The single most important check: **holdout performance vs training performance**.

- **Holdout PnL ≈ Training PnL**: Good generalization.
- **Holdout PnL << Training PnL**: Overfitting. The model memorized the training data.
- **Holdout PnL > Training PnL**: Unusual but possible if the holdout period was easier.

You can query this directly:

```sql
-- Find overfit models: training PnL much better than evaluation PnL
SELECT
    mc.name,
    train.total_pnl AS train_pnl,
    eval.total_pnl AS holdout_pnl,
    eval.total_pnl / NULLIF(train.total_pnl, 0) AS ratio
FROM model_configs mc
JOIN training_runs train ON train.model_config_id = mc.id AND train.run_type = 'train'
JOIN training_runs eval ON eval.model_config_id = mc.id AND eval.run_type = 'evaluate'
WHERE train.status = 'completed' AND eval.status = 'completed'
ORDER BY eval.sharpe_ratio DESC;
```

### The "Does Nothing" Problem

RL models often converge to doing nothing (never opening a trade) because that's the safest action — you never lose money. This is technically correct (PnL = $0, Sharpe = 0, drawdown = 0) but useless.

Filter these out: `total_trades > 10` as a minimum.

If most models learn to do nothing, it usually means:
- The reward signal needs shaping (e.g., add a small penalty for inactivity)
- The action space is too complex (model can't explore effectively)
- Training timesteps are too low (hasn't explored enough)

---

## Infrastructure & DevOps

### Key Insight: This Workload is CPU-Bound

The bottleneck is the **exchange simulation** (Python code stepping through candles), not the neural network. PPO/SAC with the MultiInputPolicy creates a small network (~100K parameters). GPU acceleration helps little because:

- The network is tiny (forward pass is microseconds even on CPU)
- The environment is pure Python (cannot be GPU-accelerated)
- PPO collects rollouts sequentially in the environment, then does a batch update

**You want many CPU cores, not GPUs.**

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Managed PostgreSQL                    │
│   (Neon / DigitalOcean Managed DB / your existing)    │
│   Stores: klines, model_configs, training_runs,       │
│           pnl_snapshots                               │
└─────────────────────┬────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
   │ Worker  │  │ Worker  │  │ Worker  │   CPU-Optimized Droplets
   │ 16 vCPU │  │ 16 vCPU │  │ 16 vCPU │   Each runs 14 parallel
   │ 14 runs │  │ 14 runs │  │ 14 runs │   training processes
   └─────────┘  └─────────┘  └─────────┘
```

Each worker droplet:
- Pulls a model config from the DB (models with `run_type='train'` and no completed run yet)
- Runs `uv run train start --model <name>` in parallel processes
- Results write back to the shared DB
- Model .zip files save to local disk (can rsync/scp to a storage server later)

### Why NOT GPUs for This

| Factor | CPU Workers | GPU Instances |
|---|---|---|
| **$/core** | ~$21/mo per vCPU (DO CPU-Optimized) | $1,368/mo per GPU (DO H100) |
| **Parallelism** | 14 runs on 16-vCPU droplet | 1 run per GPU (env is CPU-bound anyway) |
| **Cost for 50 parallel runs** | 4 × 16-vCPU @ $336/mo = **$1,344/mo** | 50 × GPU droplet = overkill |
| **Utilization** | ~95% (CPU-bound training) | ~5% GPU, ~95% idle (env stepping on CPU) |

GPUs would only matter if you:
- Had a much larger neural network (transformer-based policy)
- Used a vectorized environment (many envs on GPU, like IsaacGym)
- Needed batched inference for inference-heavy workloads

None of those apply to this system today.

### When GPUs Would Help

If you later want to:
- Use a transformer/attention-based policy network
- Process raw order book data instead of OHLCV candles
- Run thousands of environments in parallel with GPU-accelerated simulation

Then RunPod becomes attractive for its GPU price/performance.

---

## Provider Comparison

### For Training Workers (CPU-Bound)

| Provider | Instance | vCPUs | RAM | $/mo | $/hr | $/vCPU/mo |
|---|---|---|---|---|---|---|
| **DigitalOcean** CPU-Opt | c-16 | 16 | 32 GB | $336 | $0.50 | $21 |
| **DigitalOcean** CPU-Opt | c-32 | 32 | 64 GB | $672 | $1.00 | $21 |
| **DigitalOcean** CPU-Opt | c-48 | 48 | 96 GB | $1,008 | $1.50 | $21 |

DigitalOcean's per-second billing (since Jan 2026) is ideal — spin up workers, run a training batch, tear down. You only pay for the hours used.

### For Database

| Option | $/mo | Notes |
|---|---|---|
| **Neon** (existing) | Free-$19 | Already in use. Good for up to ~100 concurrent writes. |
| **DO Managed Postgres** | $15-60 | Dedicated instance, handles high write throughput. |
| **Self-hosted on worker** | $0 | Run Postgres on one of the worker droplets. Free but single point of failure. |

For heavy training (hundreds of runs writing PnL snapshots), a dedicated managed DB ($60/mo) is worth it.

### For Model Storage

Trained models are small (~5-50 MB each). 1,000 models ≈ 5-50 GB.

| Option | $/mo | Notes |
|---|---|---|
| **Worker local disk** | Included | Simple but lost if droplet is destroyed. |
| **DO Spaces (S3-compatible)** | $5/mo for 250 GB | Persistent, accessible from all workers. |
| **rsync to a storage droplet** | $4-6/mo | Cheapest permanent storage. |

### RunPod — When to Use

RunPod excels for GPU workloads at $1.19-2.69/hr for A100/H100s. For this CPU-bound trainer, it's not the right fit. However:

- **Future use**: If you add transformer policies or GPU-accelerated environments, RunPod spot instances (A100 at $0.95/hr) are hard to beat.
- **Serverless option**: RunPod Serverless can auto-scale GPU inference for a future live-trading model serving layer.

---

## Cost Estimates

### Phase 1: Baseline Sweep (63 runs)

Assuming each run takes ~2 hours at 1M timesteps on a single vCPU:

| Setup | Time | Cost |
|---|---|---|
| 1 × c-16 (14 parallel) | 63/14 × 2h = 9h | **$4.50** |
| 2 × c-16 (28 parallel) | 63/28 × 2h = 4.5h | **$4.50** |

Yes, Phase 1 costs about **$5**.

### Phase 2: Expand Winners (90 runs)

| Setup | Time | Cost |
|---|---|---|
| 2 × c-16 (28 parallel) | 90/28 × 2h = 6.4h | **$6.40** |

### Phase 3: Hyperparameter Tuning (150 runs)

| Setup | Time | Cost |
|---|---|---|
| 2 × c-32 (60 parallel) | 150/60 × 2h = 5h | **$10.00** |

### Phase 4: Long Training (20 runs at 5M steps)

Each run takes ~10 hours at 5M timesteps:

| Setup | Time | Cost |
|---|---|---|
| 2 × c-16 (28 parallel) | 20/28 × 10h = 7h | **$7.00** |

### Total All Phases

| Phase | Runs | Compute Cost | DB Cost |
|---|---|---|---|
| Phase 1 | 63 | $5 | — |
| Phase 2 | 90 | $7 | — |
| Phase 3 | 150 | $10 | — |
| Phase 4 | 20 | $7 | — |
| **Database** | — | — | $15-60/mo |
| **Storage** | — | — | $5/mo |
| **Total** | **323** | **~$29** | **~$25-80/mo** |

The compute is dirt cheap because it's CPU-only and per-second billed. The recurring cost is the database and storage.

### At Larger Scale (1,000+ runs)

If you go wider (more symbols, more combos):

| Runs | Workers | Wall Clock | Compute Cost |
|---|---|---|---|
| 1,000 | 4 × c-32 (120 parallel) | ~17 hours | ~$68 |
| 5,000 | 8 × c-32 (240 parallel) | ~42 hours | ~$336 |
| 10,000 | 8 × c-48 (368 parallel) | ~54 hours | ~$648 |

---

## Recommended Approach

### Phase 0: Local Testing

Run a few training sessions locally to verify the pipeline works:

```bash
uv run train create-model --config btc
uv run train start --model btc_v1 --timesteps 10000  # tiny run, <1 min
uv run train list
uv run train status --run 1
```

### Phase 1: First Cloud Batch

1. **Spin up** 1 × DigitalOcean CPU-Optimized c-16 ($0.50/hr)
2. **Clone the repo**, install deps, configure DATABASE_URL to your Neon DB
3. **Ingest data** if not already done (or point to existing DB)
4. **Generate Phase 1 configs** with the sweep script
5. **Run training** with GNU parallel or a simple bash loop:

```bash
# Run 14 jobs in parallel on 16-vCPU machine
cat models.txt | parallel -j14 "uv run train start --model {}"
```

6. **Evaluate winners** on holdout data
7. **Destroy the droplet** when done

### Phase 2+: Scale Up

- Spin up more/larger droplets as needed
- All workers share the same Neon/managed-DB
- Use DO Spaces or rsync for model persistence
- Destroy workers after each batch — per-second billing means you pay only for hours used

### What You Don't Need

- **Kubernetes / Docker orchestration** — Overkill for batch training. SSH + GNU parallel works fine for <1,000 runs. Add a proper job queue (Redis/Celery) only if you go above that.
- **GPUs** — Not until you change the network architecture.
- **Persistent servers** — Spin up, train, tear down. Per-second billing makes ephemeral workers the cheapest option.
- **Multiple regions** — All workers and DB in one region to minimize latency.

---

## Future Upgrades (When to Reconsider)

| Trigger | Upgrade |
|---|---|
| > 1,000 runs per batch | Add a job queue (Redis + Celery or similar) |
| Environment stepping too slow | Rewrite exchange sim in Cython or Rust |
| Want transformer-based policy | Move to GPU workers (RunPod spot A100s at $0.95/hr) |
| Want live trading | Add inference server (RunPod Serverless or a small always-on droplet) |
| Too many PnL snapshots | Reduce snapshot_interval or move to TimescaleDB |
| Need team access | Add a simple web dashboard reading from the training_runs/pnl_snapshots tables |
