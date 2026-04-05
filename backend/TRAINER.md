# Trading Model Trainer

Trains RL models to trade crypto symbols by replaying historical kline data through a simulated exchange. Models learn by trial and error (reinforcement learning) to open/close positions profitably.

---

## Quick Start

```bash
cd backend

# 1. Install dependencies
uv sync

# 2. Run database migrations (creates trainer tables)
uv run ingest migrate

# 3. Register a model configuration
uv run train create-model --config btc    # BTC model (BTCUSDT data only)
uv run train create-model --config sol    # SOL model (BTCUSDT + SOLUSDT data)

# 4. Train (requires kline data already ingested for the symbols)
uv run train start --model btc_v1 --timesteps 100000

# 5. Check results
uv run train list
uv run train status --run 1

# 6. Evaluate on holdout data
uv run train evaluate --model btc_v1 --run 1
```

---

## How It Works

### Architecture

```
Historical Klines (PostgreSQL)
        │
        ▼
   Data Feed ──────── loads & aligns multi-symbol kline data
        │
        ▼
 Trading Environment ── Gymnasium env that simulates an exchange
   ├── Account ──────── tracks balance, margin, equity
   ├── Exchange Sim ─── processes orders, positions, fills, SL/TP, liquidation
   │
        ▼
 Training Pipeline ──── Stable-Baselines3 (PPO / SAC / A2C)
        │
        ▼
 Database ───────────── stores model configs, training runs, PnL snapshots
```

### Training Loop

1. **Load klines** from the database for the model's configured symbols and intervals.
2. **Split data** chronologically: 80% for training, 20% held out for evaluation.
3. **Create the Gymnasium environment** — a simulated exchange that replays candles one by one.
4. **Train the RL agent** (PPO by default) — the agent interacts with the environment for N timesteps, learning from rewards (equity changes).
5. **Save the trained model** to disk and record performance metrics in the database.

### What the Model Sees (Observation)

On each candle, the model receives:

| Component | Shape | Description |
|---|---|---|
| Market data | (500, features) | Last 500 candles of normalized kline data (OHLCV + other columns) |
| Account state | (5,) | Balance, equity, unrealized PnL, margin used, margin available |
| Open orders | (20, 11) | Pending limit orders with trigger/SL/TP prices and sizes |
| Open positions | (20, 6) | Active positions with entry price, size, unrealized PnL, leverage |

For a single-symbol model (BTC), each candle row has 9 features. For multi-symbol (SOL using BTC+SOL data), each row has 18 features (9 per symbol concatenated).

### What the Model Decides (Actions)

On each candle, the model outputs a 51-float vector that is interpreted as:

- **Open a new order?** — confidence threshold, direction (long/short), trigger price, stop loss, 3 take-profit levels with size distribution, position size as % of balance
- **Cancel existing orders?** — one signal per order slot (20 slots)
- **Close existing positions?** — one signal per position slot (20 slots), with partial close fraction

All actions happen simultaneously in a single step — the model can open a new order, cancel some orders, and close some positions at the same time.

### Exchange Simulation

The simulated exchange processes each candle in this order:

1. **Liquidation check** — any position where price hits liquidation → force-close, margin lost
2. **Stop loss check** — any position where price crosses SL → close at SL price (taker fee)
3. **Take profit check** — any TP where price crosses TP → partial close at TP price (maker fee)
4. **Order fill check** — any limit order where price reaches trigger → open position
5. **Model actions** — cancel orders, close positions, place new orders

**Leverage** is automatically maximized based on the stop loss distance: the closer the SL to entry, the higher leverage is possible (because the max loss per unit is smaller). A safety buffer ensures the liquidation price stays beyond the SL.

**Fees** match Binance futures: maker fee (0.02%) for limit fills and TPs, taker fee (0.04%) for SL and manual closes, plus optional flat fee per trade.

### Reward Signal

The reward on each step is the **change in equity** (balance + unrealized PnL). This directly optimizes for growing the account. Episodes end when all candles are exhausted or the account is liquidated (equity ≤ 0).

---

## Model Configurations

### BTC Model (`btc_v1`)

- **Input**: BTCUSDT 1h candles (9 features per candle)
- **Target**: Trade BTC/USDT using only BTC price data
- **Use case**: Single-symbol model, simplest setup

### SOL Model (`sol_v1`)

- **Input**: BTCUSDT + SOLUSDT 1h candles (18 features per candle)
- **Target**: Trade SOL/USDT using both BTC and SOL price data
- **Use case**: Multi-symbol model — BTC often leads altcoin moves, so BTC data may improve SOL trading signals

### Creating Custom Configs

Add a new file in `src/trainer/models/` following the pattern:

```python
from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig

def make_eth_config() -> ModelConfig:
    return ModelConfig(
        name="eth_v1",
        symbols=["BTCUSDT", "ETHUSDT"],
        intervals=["1h"],
        columns=list(ALL_KLINE_COLUMNS),
        exchange=ExchangeConfig(),
        total_timesteps=2_000_000,  # train longer
    )
```

Then register it in `cli.py`'s `BUILTIN_CONFIGS` dict and run `uv run train create-model --config eth`.

---

## CLI Commands

| Command | Description |
|---|---|
| `train create-model --config <name>` | Register a built-in model config (`btc` or `sol`) in the database |
| `train start --model <name>` | Start training. Options: `--algo PPO\|SAC\|A2C`, `--timesteps N` |
| `train evaluate --model <name> --run <id>` | Run a trained model on the 20% holdout data |
| `train list` | Show all registered models with run counts and best PnL |
| `train status --run <id>` | Show detailed metrics for a specific training/evaluation run |

---

## Database Tables

| Table | Purpose |
|---|---|
| `model_configs` | Stores model configurations as JSON (name, symbols, exchange params, etc.) |
| `training_runs` | One row per train/evaluate run with status, metrics, and path to saved model |
| `pnl_snapshots` | Periodic equity snapshots during training for PnL curve visualization |

---

## Training Tips

- **Start small**: Use `--timesteps 100000` first to verify everything works before long runs.
- **PPO vs SAC**: PPO (default) is more stable for this action space. SAC can be better but may need tuning.
- **Data requirement**: You need at least `lookback_window + 1` candles (501 for default) of kline data in the DB for the configured symbols/intervals. More data = more training examples.
- **Evaluation**: Always check holdout performance (`train evaluate`) — training PnL can overfit.
- **Multiple runs**: Train the same model multiple times with different random seeds. Keep the best performers.

---

## File Structure

```
backend/src/trainer/
├── cli.py                 # CLI entrypoint
├── config.py              # ExchangeConfig + ModelConfig (fully commented)
├── db.py                  # Database operations
├── env/
│   ├── account.py         # Balance/margin/equity tracking
│   ├── data_feed.py       # Loads klines, builds observation windows
│   ├── exchange_sim.py    # Order/position/fill simulation
│   └── trading_env.py     # Gymnasium environment (integrates all above)
├── models/
│   ├── btc_config.py      # BTC model definition
│   └── sol_config.py      # SOL model definition
└── training/
    ├── trainer.py          # SB3 training loop + callbacks
    └── evaluator.py        # Holdout evaluation
```
