# Trading Model Trainer — Design Spec

## Overview

An RL-based trading model training system in the backend. Models learn to trade symbols by replaying historical kline data through a simulated exchange, using reinforcement learning (trial and error) to discover profitable strategies.

Each model is configured with its own symbols, kline columns, intervals, and exchange parameters. Models are trained with Stable-Baselines3 (PPO/SAC) against a custom Gymnasium environment that simulates order fills, positions, fees, leverage, and liquidation. Successful models (by PnL) are kept.

**Stack:** Python 3.12+, Gymnasium, Stable-Baselines3, PyTorch, NumPy, pandas, psycopg (existing).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Model Configurations                   │
│  (symbols, columns, intervals, exchange params, etc.)    │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │      Data Feed          │
          │  (reads klines from DB, │
          │   aligns multi-symbol,  │
          │   builds 500-candle     │
          │   feature windows)      │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Trading Environment   │  ← Gymnasium env
          │  ┌───────────────────┐  │
          │  │  Exchange Sim     │  │  (order matching, position mgmt,
          │  │  + Account        │  │   fees, leverage, liquidation)
          │  └───────────────────┘  │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │    Training Pipeline    │  ← SB3 (PPO / SAC)
          │  (train, checkpoint,    │
          │   evaluate, rank)       │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │      Database           │
          │  (model configs, runs,  │
          │   PnL snapshots)        │
          └─────────────────────────┘
```

### Directory Structure

```
backend/src/trainer/
├── cli.py                # CLI entrypoint (train, evaluate, list-models)
├── config.py             # ModelConfig, ExchangeConfig dataclasses
├── env/
│   ├── trading_env.py    # Gymnasium TradingEnv
│   ├── exchange_sim.py   # Order matching, position tracking, fills, leverage, liquidation
│   ├── account.py        # Balance, equity, margin, PnL tracking
│   └── data_feed.py      # Reads klines from DB, prepares observation windows
├── models/
│   ├── btc_config.py     # BTC model config (BTCUSDT, all columns, 1h)
│   └── sol_config.py     # SOL model config (BTCUSDT + SOLUSDT, 1h)
├── training/
│   ├── trainer.py        # SB3 training loop, checkpointing
│   └── evaluator.py      # Run trained model on holdout data, compute metrics
└── db.py                 # DB operations for trainer tables
```

The Trading Environment is framework-agnostic (pure Gymnasium). The Training Pipeline is where SB3 lives. Swapping SB3 for RLlib or custom PyTorch changes only the training layer — the environment is reusable.

## Trading Environment

### Observation Space

A dictionary with three parts:

**1. Market data** — shape `(500, num_features)`

The last 500 candles for each configured symbol/interval. Features are the configured kline columns per symbol, concatenated per candle row. All values normalized (z-score per column over the window).

- BTC model (1 symbol × 9 columns): `(500, 9)`
- SOL model (2 symbols × 9 columns): `(500, 18)`

For multi-symbol configs, candles are inner-joined on `open_time` — only timestamps where all symbols have data are used.

For multi-interval configs (e.g., `["1h", "4h"]`), the environment steps through the **finest interval** (1h). At each step, each coarser interval contributes its most recent completed candle's data. This means the 4h columns update every 4 steps while the 1h columns update every step. Features per row = `num_symbols × num_columns × num_intervals`.

**2. Account state** — flat vector of 5 floats:

`[balance, equity, unrealized_pnl, margin_used, margin_available]`

Normalized relative to starting balance ($10,000).

**3. Open orders & positions** — fixed-size padded arrays:

- Up to `MAX_OPEN_ORDERS` (default 20) order slots, each: `[active, direction, trigger_price, sl_price, tp1, tp2, tp3, tp1_pct, tp2_pct, tp3_pct, size]` (11 floats)
- Up to `MAX_OPEN_POSITIONS` (default 20) position slots, each: `[active, direction, entry_price, current_size, unrealized_pnl, leverage]` (6 floats)
- Prices normalized relative to current close. Unused slots are zeroed.

### Action Space

A single `Box` (continuous vector) of fixed size. With default limits (20 orders, 20 positions): **51 floats**.

| Indices | Field | Range | Meaning |
|---|---|---|---|
| 0 | `open_confidence` | [0,1] | Place new order if > 0.5 |
| 1 | `direction` | [0,1] | >0.5 = long, ≤0.5 = short |
| 2 | `trigger_offset` | [-1,1] | % offset from current close (scaled by `max_trigger_offset_pct`, default 5%) |
| 3 | `sl_distance` | [0,1] | % distance from trigger for SL (scaled between `min_sl_pct` 0.1% and `max_sl_pct` 10%) |
| 4-6 | `tp1/tp2/tp3_distance` | [0,1] | % distance from trigger for each TP |
| 7-9 | `tp1/tp2/tp3_size_pct` | [0,1] | Softmaxed to distribute 100% of position across TPs |
| 10 | `position_size_pct` | [0,1] | % of available balance to use as margin |
| 11 to 11+MAX_ORDERS | `cancel_order_i` | [0,1] | >0.5 = cancel order at slot i |
| next MAX_POSITIONS | `close_pos_i_frac` | [0,1] | 0 = hold, >0 = fraction of position i to close |

The model can simultaneously open a new order, cancel existing orders, and partially/fully close positions in a single step.

### Step Logic

Each step corresponds to one candle. Processing order within a step:

1. Advance to next candle.
2. **Check liquidations** — any position whose mark price hit liquidation price → force-close, deduct remaining margin from balance.
3. **Check SL triggers** — any position whose candle low (for longs) or high (for shorts) crossed SL → close at SL price, apply taker fee.
4. **Check TP triggers** — any TP whose candle high (for longs) or low (for shorts) crossed TP price → partial close at TP price, apply maker fee.
5. **Check order fills** — any limit order whose candle range covers trigger price → open position (calculate leverage from SL distance), apply maker fee.
6. **Process model actions:**
   a. Cancel flagged orders (cancel_order_i > 0.5).
   b. Close flagged positions (close_pos_i_frac > 0 → close that fraction).
   c. If open_confidence > 0.5 and sufficient margin → place new limit order.
7. **Compute reward** — change in equity since last step.
8. **Build new observation.**

### Reward

Change in equity (balance + unrealized PnL) from previous step to current step. Directly optimizes for growing the account.

### Episode Termination

- All candles exhausted (truncation).
- Account equity ≤ 0 (liquidated / blown up).

## Exchange Simulation

### ExchangeConfig

```python
@dataclass
class ExchangeConfig:
    maker_fee_pct: float = 0.02        # 0.02% per fill (limit orders = maker)
    taker_fee_pct: float = 0.04        # 0.04% per fill (SL/liquidation/manual close = taker)
    flat_fee_usd: float = 0.0          # flat fee per trade

    max_leverage: float = 125.0        # absolute cap
    liquidation_buffer_pct: float = 0.5  # liquidation triggers 0.5% before bankruptcy
    maintenance_margin_pct: float = 0.4  # maintenance margin rate

    max_open_orders: int = 20
    max_open_positions: int = 20
    min_order_size_usd: float = 10.0   # minimum notional per order
```

### Leverage Calculation

When an order fills, leverage is auto-calculated to maximize leverage while keeping liquidation price beyond the SL:

1. Compute distance from entry to SL as a percentage.
2. Compute max leverage where liquidation price is still beyond the SL:
   - For long: `liq_price = entry × (1 - 1/leverage + maintenance_margin_pct/leverage)`. Constraint: `liq_price < sl_price`.
   - For short: `liq_price = entry × (1 + 1/leverage - maintenance_margin_pct/leverage)`. Constraint: `liq_price > sl_price`.
3. Clamp to `max_leverage`.
4. Margin required = notional_value / leverage.

Tighter SLs → lower leverage (liquidation would be too close). Wider SLs → higher leverage (safely).

### Fee Application

- **Order fills** (limit): maker fee on notional + flat fee.
- **TP fills** (limit): maker fee on notional.
- **SL fills** (market-like): taker fee on notional.
- **Manual close**: taker fee on notional.
- **Liquidation**: taker fee + remaining margin is lost.

### Position Mechanics

- Both long and short positions supported.
- Multiple concurrent positions (unlimited within MAX_OPEN_POSITIONS).
- Each position is independent with its own entry price, SL, TPs, leverage, and size.
- Partial closes reduce position size proportionally.
- When all TPs hit, position is fully closed.

## Model Configuration

### ModelConfig

```python
@dataclass
class ModelConfig:
    name: str                              # "btc_v1", "sol_v1"
    symbols: list[str]                     # ["BTCUSDT"] or ["BTCUSDT", "SOLUSDT"]
    intervals: list[str]                   # ["1h"] or ["1h", "4h"]
    columns: list[str]                     # which kline columns as features
    exchange: ExchangeConfig
    lookback_window: int = 500
    initial_balance: float = 10_000.0
    num_tp_levels: int = 3

    # Action scaling
    max_trigger_offset_pct: float = 5.0    # trigger_offset [-1,1] maps to ±5%
    min_sl_pct: float = 0.1               # sl_distance 0 maps to 0.1%
    max_sl_pct: float = 10.0              # sl_distance 1 maps to 10%
    max_tp_pct: float = 20.0              # tp distances scaled to 0-20%

    # Training params
    algorithm: str = "PPO"                 # "PPO", "SAC", "A2C"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
```

### Initial Implementations

**BTC model** (`btc_config.py`):
- symbols: `["BTCUSDT"]`
- intervals: `["1h"]`
- columns: all 9 kline columns (open, high, low, close, volume, quote_volume, num_trades, taker_buy_base_vol, taker_buy_quote_vol)
- exchange: defaults

**SOL model** (`sol_config.py`):
- symbols: `["BTCUSDT", "SOLUSDT"]`
- intervals: `["1h"]`
- columns: all 9 kline columns (for both symbols → 18 features per row)
- exchange: defaults

## Database Schema

Migration: `003_trainer_tables.sql`

### model_configs

| Column | Type | Notes |
|---|---|---|
| id | SERIAL PK | |
| name | TEXT UNIQUE NOT NULL | e.g. "btc_v1" |
| config_json | JSONB NOT NULL | Full ModelConfig + ExchangeConfig serialized |
| created_at | TIMESTAMPTZ DEFAULT now() | |

### training_runs

| Column | Type | Notes |
|---|---|---|
| id | SERIAL PK | |
| model_config_id | INT FK → model_configs.id | |
| run_type | TEXT NOT NULL | "train" or "evaluate" |
| algorithm | TEXT NOT NULL | "PPO", "SAC", etc. |
| status | TEXT NOT NULL DEFAULT 'running' | running / completed / failed |
| started_at | TIMESTAMPTZ DEFAULT now() | |
| completed_at | TIMESTAMPTZ | |
| total_timesteps | INT | |
| final_balance | NUMERIC | |
| final_equity | NUMERIC | |
| total_pnl | NUMERIC | final_equity - initial_balance |
| total_trades | INT | |
| win_rate | NUMERIC | % of profitable trades |
| max_drawdown | NUMERIC | worst peak-to-trough equity drop % |
| sharpe_ratio | NUMERIC | risk-adjusted return |
| model_path | TEXT | filesystem path to saved .zip |
| error | TEXT | |

### pnl_snapshots

| Column | Type | Notes |
|---|---|---|
| id | BIGSERIAL PK | |
| training_run_id | INT FK → training_runs.id | |
| step | INT NOT NULL | env step number |
| candle_time | BIGINT NOT NULL | open_time of candle at this step |
| balance | NUMERIC NOT NULL | |
| equity | NUMERIC NOT NULL | |
| unrealized_pnl | NUMERIC NOT NULL | |
| open_position_count | INT NOT NULL | |
| open_order_count | INT NOT NULL | |

Index on `(training_run_id, step)`.

Snapshots recorded every N steps (configurable, default 100) to keep the table manageable.

## Training Pipeline

### CLI Commands

```
uv run train create-model --config btc      # register btc_v1 in model_configs
uv run train create-model --config sol      # register sol_v1 in model_configs
uv run train start --model btc_v1           # train with default algo (PPO)
uv run train start --model sol_v1 --algo SAC --timesteps 2000000
uv run train evaluate --model btc_v1 --run 3  # evaluate run on holdout data
uv run train list                             # list models + best runs
uv run train status --run 3                   # detailed stats for a run
```

### Training Flow

1. Load ModelConfig from DB (registered via `create-model`).
2. Query klines from DB for configured symbols/intervals.
3. Chronological split: 80% train, 20% holdout (no random shuffle — prevents lookahead bias).
4. Build Gymnasium TradingEnv with training data.
5. Create SB3 model (PPO/SAC/A2C) with the env.
6. Train for total_timesteps, saving checkpoints every 100k steps.
7. After training: auto-evaluate on holdout data, record metrics to training_runs.
8. Save PnL snapshots during both training and evaluation.

### Model Storage

```
backend/trained_models/{model_name}/{run_id}/
├── model.zip              # final SB3 model
├── checkpoint_100k.zip
├── checkpoint_200k.zip
└── ...
```

`model_path` in training_runs points to the final .zip for reload/re-evaluation.

## Dependencies (additions to pyproject.toml)

```
gymnasium>=1.0.0
stable-baselines3>=2.4.0
torch>=2.2.0
numpy>=1.26.0
pandas>=2.2.0
```

These are added alongside the existing dependencies (psycopg, fastapi, etc.). The `train` CLI entrypoint is registered separately from the existing `ingest` entrypoint.
