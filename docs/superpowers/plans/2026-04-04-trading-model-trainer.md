# Trading Model Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an RL-based trading model training system that trains models to trade crypto symbols by replaying historical kline data through a simulated exchange, using Stable-Baselines3 (PPO/SAC).

**Architecture:** A Gymnasium custom environment wraps an exchange simulator (orders, positions, fees, leverage, liquidation). A data feed reads klines from the existing PostgreSQL database and builds observation windows. SB3 algorithms train against this environment. Model configs, training runs, and PnL snapshots are stored in the database.

**Tech Stack:** Python 3.12+, Gymnasium, Stable-Baselines3, PyTorch, NumPy, pandas, psycopg (existing)

**Spec:** `docs/superpowers/specs/2026-04-04-trading-model-trainer-design.md`

---

## File Structure

```
backend/
├── pyproject.toml                          # MODIFY: add dependencies + train entrypoint
├── migrations/
│   └── 003_trainer_tables.sql              # CREATE: model_configs, training_runs, pnl_snapshots
├── src/
│   └── trainer/
│       ├── __init__.py                     # CREATE: empty
│       ├── cli.py                          # CREATE: argparse CLI (train command)
│       ├── config.py                       # CREATE: ExchangeConfig, ModelConfig dataclasses
│       ├── db.py                           # CREATE: DB operations for trainer tables
│       ├── env/
│       │   ├── __init__.py                 # CREATE: empty
│       │   ├── account.py                  # CREATE: balance/equity/margin tracking
│       │   ├── exchange_sim.py             # CREATE: order matching, positions, fees, leverage
│       │   ├── data_feed.py                # CREATE: klines query, multi-symbol alignment
│       │   └── trading_env.py              # CREATE: Gymnasium TradingEnv
│       ├── models/
│       │   ├── __init__.py                 # CREATE: empty
│       │   ├── btc_config.py               # CREATE: BTC model config
│       │   └── sol_config.py               # CREATE: SOL model config
│       └── training/
│           ├── __init__.py                 # CREATE: empty
│           ├── trainer.py                  # CREATE: SB3 training loop
│           └── evaluator.py                # CREATE: evaluation + metrics
└── tests/
    └── trainer/
        ├── __init__.py                     # CREATE: empty
        ├── test_account.py                 # CREATE
        ├── test_exchange_sim.py            # CREATE
        ├── test_data_feed.py               # CREATE
        └── test_trading_env.py             # CREATE
```

---

### Task 1: Project Setup & Dependencies

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/src/trainer/__init__.py`
- Create: `backend/src/trainer/env/__init__.py`
- Create: `backend/src/trainer/models/__init__.py`
- Create: `backend/src/trainer/training/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/trainer/__init__.py`

- [ ] **Step 1: Update pyproject.toml**

Add the new dependencies and the `train` CLI entrypoint. The project name stays `ingester` (it's the existing package). Both `ingester` and `trainer` are source packages under `src/`.

In `backend/pyproject.toml`, replace the full content with:

```toml
[project]
name = "ingester"
version = "0.1.0"
description = "Binance USDT-M futures kline ingester"
readme = "README.md"
authors = [
    { name = "Mardo Del Cid", email = "mardodcp@gmail.com" }
]
requires-python = ">=3.12"
dependencies = [
    "ccxt>=4.5.46",
    "fastapi>=0.135.3",
    "psycopg[binary]>=3.3.3",
    "python-dotenv>=1.2.2",
    "uvicorn[standard]>=0.43.0",
    "gymnasium>=1.0.0",
    "stable-baselines3>=2.4.0",
    "torch>=2.2.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "pytest>=8.0.0",
]

[project.scripts]
ingest = "ingester.cli:main"
train = "trainer.cli:main"

[build-system]
requires = ["uv_build>=0.10.4,<0.11.0"]
build-backend = "uv_build"
```

- [ ] **Step 2: Create package directory structure**

Create all `__init__.py` files (all empty):

```bash
mkdir -p backend/src/trainer/env backend/src/trainer/models backend/src/trainer/training
mkdir -p backend/tests/trainer
touch backend/src/trainer/__init__.py
touch backend/src/trainer/env/__init__.py
touch backend/src/trainer/models/__init__.py
touch backend/src/trainer/training/__init__.py
touch backend/tests/__init__.py
touch backend/tests/trainer/__init__.py
```

- [ ] **Step 3: Install dependencies**

```bash
cd backend && uv sync
```

Expected: resolves and installs gymnasium, stable-baselines3, torch, numpy, pandas, pytest.

- [ ] **Step 4: Verify package discovery**

```bash
cd backend && uv run python -c "import gymnasium; import stable_baselines3; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/trainer/ backend/tests/
git commit -m "feat(trainer): scaffold package structure and add ML dependencies"
```

---

### Task 2: Config Dataclasses

**Files:**
- Create: `backend/src/trainer/config.py`
- Create: `backend/src/trainer/models/btc_config.py`
- Create: `backend/src/trainer/models/sol_config.py`

- [ ] **Step 1: Create config.py with ExchangeConfig and ModelConfig**

Create `backend/src/trainer/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExchangeConfig:
    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.04
    flat_fee_usd: float = 0.0

    max_leverage: float = 125.0
    liquidation_buffer_pct: float = 0.5
    maintenance_margin_pct: float = 0.4

    max_open_orders: int = 20
    max_open_positions: int = 20
    min_order_size_usd: float = 10.0

    def to_dict(self) -> dict:
        return {
            "maker_fee_pct": self.maker_fee_pct,
            "taker_fee_pct": self.taker_fee_pct,
            "flat_fee_usd": self.flat_fee_usd,
            "max_leverage": self.max_leverage,
            "liquidation_buffer_pct": self.liquidation_buffer_pct,
            "maintenance_margin_pct": self.maintenance_margin_pct,
            "max_open_orders": self.max_open_orders,
            "max_open_positions": self.max_open_positions,
            "min_order_size_usd": self.min_order_size_usd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExchangeConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


ALL_KLINE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
]


@dataclass
class ModelConfig:
    name: str
    symbols: list[str]
    intervals: list[str]
    columns: list[str] = field(default_factory=lambda: list(ALL_KLINE_COLUMNS))
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)

    lookback_window: int = 500
    initial_balance: float = 10_000.0
    num_tp_levels: int = 3

    max_trigger_offset_pct: float = 5.0
    min_sl_pct: float = 0.1
    max_sl_pct: float = 10.0
    max_tp_pct: float = 20.0

    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4

    snapshot_interval: int = 100

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbols": self.symbols,
            "intervals": self.intervals,
            "columns": self.columns,
            "exchange": self.exchange.to_dict(),
            "lookback_window": self.lookback_window,
            "initial_balance": self.initial_balance,
            "num_tp_levels": self.num_tp_levels,
            "max_trigger_offset_pct": self.max_trigger_offset_pct,
            "min_sl_pct": self.min_sl_pct,
            "max_sl_pct": self.max_sl_pct,
            "max_tp_pct": self.max_tp_pct,
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "snapshot_interval": self.snapshot_interval,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        d = dict(d)
        if "exchange" in d and isinstance(d["exchange"], dict):
            d["exchange"] = ExchangeConfig.from_dict(d["exchange"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def num_features_per_candle(self) -> int:
        return len(self.symbols) * len(self.columns) * len(self.intervals)

    @property
    def action_size(self) -> int:
        base = 1 + 1 + 1 + 1 + self.num_tp_levels + self.num_tp_levels + 1
        return base + self.exchange.max_open_orders + self.exchange.max_open_positions
```

- [ ] **Step 2: Create BTC model config**

Create `backend/src/trainer/models/btc_config.py`:

```python
from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig


def make_btc_config() -> ModelConfig:
    return ModelConfig(
        name="btc_v1",
        symbols=["BTCUSDT"],
        intervals=["1h"],
        columns=list(ALL_KLINE_COLUMNS),
        exchange=ExchangeConfig(),
    )
```

- [ ] **Step 3: Create SOL model config**

Create `backend/src/trainer/models/sol_config.py`:

```python
from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig


def make_sol_config() -> ModelConfig:
    return ModelConfig(
        name="sol_v1",
        symbols=["BTCUSDT", "SOLUSDT"],
        intervals=["1h"],
        columns=list(ALL_KLINE_COLUMNS),
        exchange=ExchangeConfig(),
    )
```

- [ ] **Step 4: Verify configs load**

```bash
cd backend && uv run python -c "
from trainer.models.btc_config import make_btc_config
from trainer.models.sol_config import make_sol_config
btc = make_btc_config()
sol = make_sol_config()
print(f'BTC features/candle: {btc.num_features_per_candle}, action size: {btc.action_size}')
print(f'SOL features/candle: {sol.num_features_per_candle}, action size: {sol.action_size}')
import json; print(json.dumps(btc.to_dict(), indent=2))
"
```

Expected output:
```
BTC features/candle: 9, action size: 51
SOL features/candle: 18, action size: 51
{ ... full JSON dump of btc config ... }
```

- [ ] **Step 5: Commit**

```bash
git add backend/src/trainer/config.py backend/src/trainer/models/
git commit -m "feat(trainer): add ExchangeConfig, ModelConfig, BTC and SOL configs"
```

---

### Task 3: Database Migration & DB Operations

**Files:**
- Create: `backend/migrations/003_trainer_tables.sql`
- Create: `backend/src/trainer/db.py`

- [ ] **Step 1: Create migration SQL**

Create `backend/migrations/003_trainer_tables.sql`:

```sql
CREATE TABLE model_configs (
    id          SERIAL PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    config_json JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE training_runs (
    id               SERIAL PRIMARY KEY,
    model_config_id  INTEGER NOT NULL REFERENCES model_configs(id),
    run_type         TEXT NOT NULL,
    algorithm        TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'running',
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at     TIMESTAMPTZ,
    total_timesteps  INTEGER,
    final_balance    NUMERIC,
    final_equity     NUMERIC,
    total_pnl        NUMERIC,
    total_trades     INTEGER,
    win_rate         NUMERIC,
    max_drawdown     NUMERIC,
    sharpe_ratio     NUMERIC,
    model_path       TEXT,
    error            TEXT
);

CREATE INDEX training_runs_model_idx ON training_runs (model_config_id);
CREATE INDEX training_runs_status_idx ON training_runs (status);

CREATE TABLE pnl_snapshots (
    id               BIGSERIAL PRIMARY KEY,
    training_run_id  INTEGER NOT NULL REFERENCES training_runs(id),
    step             INTEGER NOT NULL,
    candle_time      BIGINT NOT NULL,
    balance          NUMERIC NOT NULL,
    equity           NUMERIC NOT NULL,
    unrealized_pnl   NUMERIC NOT NULL,
    open_position_count INTEGER NOT NULL,
    open_order_count    INTEGER NOT NULL
);

CREATE INDEX pnl_snapshots_run_step_idx ON pnl_snapshots (training_run_id, step);
```

- [ ] **Step 2: Create trainer db.py**

Create `backend/src/trainer/db.py`:

```python
from __future__ import annotations

import json

import psycopg

from ingester.db import connect
from trainer.config import ModelConfig


def save_model_config(config: ModelConfig) -> int:
    conn = connect()
    try:
        row = conn.execute(
            """
            INSERT INTO model_configs (name, config_json)
            VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET config_json = EXCLUDED.config_json
            RETURNING id
            """,
            (config.name, json.dumps(config.to_dict())),
        ).fetchone()
        conn.commit()
        return row[0]
    finally:
        conn.close()


def load_model_config(name: str) -> ModelConfig | None:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT config_json FROM model_configs WHERE name = %s", (name,)
        ).fetchone()
        if row is None:
            return None
        return ModelConfig.from_dict(row[0])
    finally:
        conn.close()


def get_model_config_id(name: str) -> int | None:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT id FROM model_configs WHERE name = %s", (name,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def list_model_configs() -> list[dict]:
    conn = connect()
    try:
        rows = conn.execute(
            """
            SELECT mc.name, mc.created_at,
                   count(tr.id) AS run_count,
                   max(tr.total_pnl) AS best_pnl
            FROM model_configs mc
            LEFT JOIN training_runs tr ON tr.model_config_id = mc.id
                AND tr.status = 'completed'
            GROUP BY mc.id
            ORDER BY mc.name
            """
        ).fetchall()
        return [
            {"name": r[0], "created_at": r[1], "run_count": r[2], "best_pnl": r[3]}
            for r in rows
        ]
    finally:
        conn.close()


def create_training_run(
    model_config_id: int, run_type: str, algorithm: str
) -> int:
    conn = connect()
    try:
        row = conn.execute(
            """
            INSERT INTO training_runs (model_config_id, run_type, algorithm)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (model_config_id, run_type, algorithm),
        ).fetchone()
        conn.commit()
        return row[0]
    finally:
        conn.close()


def complete_training_run(
    run_id: int,
    *,
    final_balance: float,
    final_equity: float,
    total_pnl: float,
    total_trades: int,
    win_rate: float,
    max_drawdown: float,
    sharpe_ratio: float,
    model_path: str,
) -> None:
    conn = connect()
    try:
        conn.execute(
            """
            UPDATE training_runs
            SET status = 'completed', completed_at = now(),
                final_balance = %s, final_equity = %s, total_pnl = %s,
                total_trades = %s, win_rate = %s, max_drawdown = %s,
                sharpe_ratio = %s, model_path = %s
            WHERE id = %s
            """,
            (final_balance, final_equity, total_pnl, total_trades,
             win_rate, max_drawdown, sharpe_ratio, model_path, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def fail_training_run(run_id: int, error: str) -> None:
    conn = connect()
    try:
        conn.execute(
            """
            UPDATE training_runs
            SET status = 'failed', completed_at = now(), error = %s
            WHERE id = %s
            """,
            (error, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_pnl_snapshots(
    conn: psycopg.Connection, snapshots: list[dict]
) -> None:
    if not snapshots:
        return
    conn.executemany(
        """
        INSERT INTO pnl_snapshots
            (training_run_id, step, candle_time, balance, equity,
             unrealized_pnl, open_position_count, open_order_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        [
            (s["training_run_id"], s["step"], s["candle_time"],
             s["balance"], s["equity"], s["unrealized_pnl"],
             s["open_position_count"], s["open_order_count"])
            for s in snapshots
        ],
    )
    conn.commit()


def get_training_run(run_id: int) -> dict | None:
    conn = connect()
    try:
        cur = conn.execute(
            """
            SELECT tr.*, mc.name AS model_name
            FROM training_runs tr
            JOIN model_configs mc ON mc.id = tr.model_config_id
            WHERE tr.id = %s
            """,
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [desc.name for desc in cur.description]
        return dict(zip(cols, row))
    finally:
        conn.close()
```

- [ ] **Step 3: Run migration**

```bash
cd backend && uv run ingest migrate
```

Expected: `[migrate] Applied: 003_trainer_tables.sql`

- [ ] **Step 4: Verify tables exist**

```bash
cd backend && uv run python -c "
from ingester.db import connect
conn = connect()
for table in ['model_configs', 'training_runs', 'pnl_snapshots']:
    row = conn.execute(\"SELECT count(*) FROM \" + table).fetchone()
    print(f'{table}: {row[0]} rows')
conn.close()
"
```

Expected:
```
model_configs: 0 rows
training_runs: 0 rows
pnl_snapshots: 0 rows
```

- [ ] **Step 5: Commit**

```bash
git add backend/migrations/003_trainer_tables.sql backend/src/trainer/db.py
git commit -m "feat(trainer): add database migration and DB operations"
```

---

### Task 4: Account Tracker

**Files:**
- Create: `backend/src/trainer/env/account.py`
- Create: `backend/tests/trainer/test_account.py`

- [ ] **Step 1: Write tests for Account**

Create `backend/tests/trainer/test_account.py`:

```python
import pytest

from trainer.env.account import Account


def test_initial_state():
    acc = Account(initial_balance=10_000.0)
    assert acc.balance == 10_000.0
    assert acc.margin_used == 0.0
    assert acc.available_balance == 10_000.0


def test_allocate_margin():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(1_000.0)
    assert acc.margin_used == 1_000.0
    assert acc.available_balance == 9_000.0


def test_allocate_margin_insufficient():
    acc = Account(initial_balance=10_000.0)
    with pytest.raises(ValueError, match="Insufficient"):
        acc.allocate_margin(11_000.0)


def test_release_margin():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(2_000.0)
    acc.release_margin(500.0)
    assert acc.margin_used == 1_500.0
    assert acc.available_balance == 8_500.0


def test_realize_pnl_profit():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(1_000.0)
    acc.realize_pnl(200.0)
    assert acc.balance == 10_200.0
    assert acc.margin_used == 1_000.0


def test_realize_pnl_loss():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(1_000.0)
    acc.realize_pnl(-300.0)
    assert acc.balance == 9_700.0


def test_apply_fee():
    acc = Account(initial_balance=10_000.0)
    acc.apply_fee(50.0)
    assert acc.balance == 9_950.0


def test_equity_with_unrealized():
    acc = Account(initial_balance=10_000.0)
    assert acc.equity(unrealized_pnl=500.0) == 10_500.0
    assert acc.equity(unrealized_pnl=-200.0) == 9_800.0


def test_reset():
    acc = Account(initial_balance=10_000.0)
    acc.allocate_margin(2_000.0)
    acc.realize_pnl(500.0)
    acc.reset()
    assert acc.balance == 10_000.0
    assert acc.margin_used == 0.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd backend && uv run pytest tests/trainer/test_account.py -v
```

Expected: all tests FAIL with `ModuleNotFoundError: No module named 'trainer.env.account'`

- [ ] **Step 3: Implement Account**

Create `backend/src/trainer/env/account.py`:

```python
from __future__ import annotations


class Account:
    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.margin_used = 0.0

    @property
    def available_balance(self) -> float:
        return self.balance - self.margin_used

    def equity(self, unrealized_pnl: float = 0.0) -> float:
        return self.balance + unrealized_pnl

    def allocate_margin(self, amount: float) -> None:
        if amount > self.available_balance:
            raise ValueError(
                f"Insufficient balance: need {amount:.2f}, "
                f"available {self.available_balance:.2f}"
            )
        self.margin_used += amount

    def release_margin(self, amount: float) -> None:
        self.margin_used = max(0.0, self.margin_used - amount)

    def realize_pnl(self, pnl: float) -> None:
        self.balance += pnl

    def apply_fee(self, fee: float) -> None:
        self.balance -= fee

    def reset(self) -> None:
        self.balance = self.initial_balance
        self.margin_used = 0.0
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd backend && uv run pytest tests/trainer/test_account.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/trainer/env/account.py backend/tests/trainer/test_account.py
git commit -m "feat(trainer): add Account tracker with tests"
```

---

### Task 5: Exchange Simulator

This is the largest component. It manages orders, positions, fills, leverage calculation, fees, and liquidation.

**Files:**
- Create: `backend/src/trainer/env/exchange_sim.py`
- Create: `backend/tests/trainer/test_exchange_sim.py`

- [ ] **Step 1: Write tests for exchange simulator**

Create `backend/tests/trainer/test_exchange_sim.py`:

```python
import pytest

from trainer.config import ExchangeConfig
from trainer.env.account import Account
from trainer.env.exchange_sim import ExchangeSim, Order, Position


@pytest.fixture
def exchange() -> ExchangeSim:
    return ExchangeSim(
        config=ExchangeConfig(
            maker_fee_pct=0.02,
            taker_fee_pct=0.04,
            flat_fee_usd=0.0,
            max_leverage=125.0,
            liquidation_buffer_pct=0.5,
            maintenance_margin_pct=0.4,
            max_open_orders=5,
            max_open_positions=5,
            min_order_size_usd=10.0,
        ),
        account=Account(initial_balance=10_000.0),
    )


class TestLeverageCalculation:
    def test_long_leverage_from_sl(self, exchange: ExchangeSim):
        leverage = exchange.compute_leverage(
            entry_price=50_000.0, sl_price=49_000.0, direction=1
        )
        assert leverage > 0
        liq = exchange.liquidation_price(50_000.0, leverage, 1)
        assert liq < 49_000.0

    def test_short_leverage_from_sl(self, exchange: ExchangeSim):
        leverage = exchange.compute_leverage(
            entry_price=50_000.0, sl_price=51_000.0, direction=-1
        )
        assert leverage > 0
        liq = exchange.liquidation_price(50_000.0, leverage, -1)
        assert liq > 51_000.0

    def test_leverage_clamped_to_max(self, exchange: ExchangeSim):
        leverage = exchange.compute_leverage(
            entry_price=50_000.0, sl_price=10_000.0, direction=1
        )
        assert leverage <= exchange.config.max_leverage

    def test_tight_sl_gives_lower_leverage(self, exchange: ExchangeSim):
        wide = exchange.compute_leverage(50_000.0, 45_000.0, 1)
        tight = exchange.compute_leverage(50_000.0, 49_500.0, 1)
        assert tight < wide


class TestPlaceOrder:
    def test_place_long_order(self, exchange: ExchangeSim):
        order = exchange.place_order(
            direction=1,
            trigger_price=50_000.0,
            sl_price=49_000.0,
            tp_prices=[51_000.0, 52_000.0],
            tp_size_pcts=[0.5, 0.5],
            margin=1_000.0,
        )
        assert order is not None
        assert order.direction == 1
        assert len(exchange.open_orders) == 1
        assert exchange.account.margin_used == 1_000.0

    def test_place_order_rejected_no_margin(self, exchange: ExchangeSim):
        order = exchange.place_order(
            direction=1,
            trigger_price=50_000.0,
            sl_price=49_000.0,
            tp_prices=[51_000.0],
            tp_size_pcts=[1.0],
            margin=20_000.0,
        )
        assert order is None

    def test_place_order_rejected_max_orders(self, exchange: ExchangeSim):
        for i in range(5):
            exchange.place_order(
                direction=1,
                trigger_price=50_000.0 + i,
                sl_price=49_000.0,
                tp_prices=[51_000.0],
                tp_size_pcts=[1.0],
                margin=100.0,
            )
        order = exchange.place_order(
            direction=1,
            trigger_price=50_010.0,
            sl_price=49_000.0,
            tp_prices=[51_000.0],
            tp_size_pcts=[1.0],
            margin=100.0,
        )
        assert order is None


class TestOrderFill:
    def test_long_order_fills_when_price_drops_to_trigger(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1,
            trigger_price=49_000.0,
            sl_price=48_000.0,
            tp_prices=[50_000.0],
            tp_size_pcts=[1.0],
            margin=1_000.0,
        )
        fills = exchange.process_candle(
            high=50_000.0, low=48_500.0, close=49_500.0
        )
        assert len(exchange.open_orders) == 0
        assert len(exchange.open_positions) == 1
        pos = exchange.open_positions[0]
        assert pos.direction == 1
        assert pos.entry_price == 49_000.0

    def test_short_order_fills_when_price_rises_to_trigger(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=-1,
            trigger_price=51_000.0,
            sl_price=52_000.0,
            tp_prices=[50_000.0],
            tp_size_pcts=[1.0],
            margin=1_000.0,
        )
        fills = exchange.process_candle(
            high=51_500.0, low=50_000.0, close=50_500.0
        )
        assert len(exchange.open_positions) == 1
        assert exchange.open_positions[0].direction == -1


class TestStopLoss:
    def test_long_sl_triggers(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        assert len(exchange.open_positions) == 1

        exchange.process_candle(high=50_000.0, low=48_500.0, close=48_800.0)
        assert len(exchange.open_positions) == 0


class TestTakeProfit:
    def test_partial_tp_triggers(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[51_000.0, 53_000.0], tp_size_pcts=[0.5, 0.5],
            margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        pos = exchange.open_positions[0]
        original_size = pos.size

        exchange.process_candle(high=51_500.0, low=50_500.0, close=51_200.0)
        assert len(exchange.open_positions) == 1
        assert exchange.open_positions[0].size == pytest.approx(
            original_size * 0.5, rel=1e-6
        )

    def test_all_tps_close_position(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[51_000.0, 52_000.0], tp_size_pcts=[0.5, 0.5],
            margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        exchange.process_candle(high=53_000.0, low=50_500.0, close=52_500.0)
        assert len(exchange.open_positions) == 0


class TestManualClose:
    def test_close_position_fully(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        assert len(exchange.open_positions) == 1

        exchange.close_position(0, fraction=1.0, current_price=50_200.0)
        assert len(exchange.open_positions) == 0

    def test_close_position_partially(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        original_size = exchange.open_positions[0].size

        exchange.close_position(0, fraction=0.3, current_price=50_200.0)
        assert len(exchange.open_positions) == 1
        assert exchange.open_positions[0].size == pytest.approx(
            original_size * 0.7, rel=1e-6
        )


class TestCancelOrder:
    def test_cancel_releases_margin(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        assert exchange.account.margin_used == 1_000.0
        exchange.cancel_order(0)
        assert len(exchange.open_orders) == 0
        assert exchange.account.margin_used == 0.0


class TestFees:
    def test_maker_fee_on_order_fill(self, exchange: ExchangeSim):
        initial_balance = exchange.account.balance
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        assert exchange.account.balance < initial_balance


class TestUnrealizedPnl:
    def test_total_unrealized_pnl(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        pnl = exchange.total_unrealized_pnl(current_price=50_500.0)
        assert pnl > 0

    def test_short_unrealized_pnl(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=-1, trigger_price=50_000.0, sl_price=51_000.0,
            tp_prices=[49_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_500.0, close=49_800.0)
        pnl = exchange.total_unrealized_pnl(current_price=49_500.0)
        assert pnl > 0


class TestReset:
    def test_reset_clears_state(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.reset()
        assert len(exchange.open_orders) == 0
        assert len(exchange.open_positions) == 0
        assert exchange.account.balance == 10_000.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd backend && uv run pytest tests/trainer/test_exchange_sim.py -v
```

Expected: all tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement exchange_sim.py**

Create `backend/src/trainer/env/exchange_sim.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from trainer.config import ExchangeConfig
from trainer.env.account import Account


@dataclass
class Order:
    id: int
    direction: int
    trigger_price: float
    sl_price: float
    tp_prices: list[float]
    tp_size_pcts: list[float]
    margin: float


@dataclass
class Position:
    id: int
    direction: int
    entry_price: float
    size: float
    leverage: float
    sl_price: float
    tp_prices: list[float]
    tp_size_pcts: list[float]
    margin: float
    liquidation_price: float

    def unrealized_pnl(self, current_price: float) -> float:
        return self.direction * (current_price - self.entry_price) * self.size


@dataclass
class FillEvent:
    event_type: str
    position_id: int | None = None
    order_id: int | None = None
    pnl: float = 0.0
    fee: float = 0.0


class ExchangeSim:
    def __init__(self, config: ExchangeConfig, account: Account) -> None:
        self.config = config
        self.account = account
        self.open_orders: list[Order] = []
        self.open_positions: list[Position] = []
        self._next_order_id = 0
        self._next_position_id = 0
        self.total_trades = 0
        self.winning_trades = 0

    def reset(self) -> None:
        self.open_orders.clear()
        self.open_positions.clear()
        self._next_order_id = 0
        self._next_position_id = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.account.reset()

    def compute_leverage(
        self, entry_price: float, sl_price: float, direction: int
    ) -> float:
        sl_dist_pct = abs(entry_price - sl_price) / entry_price
        if sl_dist_pct == 0:
            return 1.0
        mm = self.config.maintenance_margin_pct / 100.0
        buf = self.config.liquidation_buffer_pct / 100.0
        safe_dist = sl_dist_pct - buf
        if safe_dist <= 0:
            return 1.0
        leverage = 1.0 / (safe_dist + mm)
        return min(leverage, self.config.max_leverage)

    def liquidation_price(
        self, entry_price: float, leverage: float, direction: int
    ) -> float:
        mm = self.config.maintenance_margin_pct / 100.0
        if direction == 1:
            return entry_price * (1.0 - (1.0 / leverage) + mm)
        else:
            return entry_price * (1.0 + (1.0 / leverage) - mm)

    def _compute_fee(
        self, notional: float, fee_type: str
    ) -> float:
        if fee_type == "maker":
            pct = self.config.maker_fee_pct / 100.0
        else:
            pct = self.config.taker_fee_pct / 100.0
        return notional * pct + self.config.flat_fee_usd

    def place_order(
        self,
        direction: int,
        trigger_price: float,
        sl_price: float,
        tp_prices: list[float],
        tp_size_pcts: list[float],
        margin: float,
    ) -> Order | None:
        if len(self.open_orders) >= self.config.max_open_orders:
            return None
        if margin > self.account.available_balance:
            return None
        leverage = self.compute_leverage(trigger_price, sl_price, direction)
        notional = margin * leverage
        if notional < self.config.min_order_size_usd:
            return None

        self.account.allocate_margin(margin)
        order = Order(
            id=self._next_order_id,
            direction=direction,
            trigger_price=trigger_price,
            sl_price=sl_price,
            tp_prices=list(tp_prices),
            tp_size_pcts=list(tp_size_pcts),
            margin=margin,
        )
        self._next_order_id += 1
        self.open_orders.append(order)
        return order

    def cancel_order(self, index: int) -> None:
        if 0 <= index < len(self.open_orders):
            order = self.open_orders.pop(index)
            self.account.release_margin(order.margin)

    def _fill_order(self, order: Order) -> Position:
        leverage = self.compute_leverage(
            order.trigger_price, order.sl_price, order.direction
        )
        notional = order.margin * leverage
        size = notional / order.trigger_price
        liq_price = self.liquidation_price(
            order.trigger_price, leverage, order.direction
        )

        fee = self._compute_fee(notional, "maker")
        self.account.apply_fee(fee)

        position = Position(
            id=self._next_position_id,
            direction=order.direction,
            entry_price=order.trigger_price,
            size=size,
            leverage=leverage,
            sl_price=order.sl_price,
            tp_prices=list(order.tp_prices),
            tp_size_pcts=list(order.tp_size_pcts),
            margin=order.margin,
            liquidation_price=liq_price,
        )
        self._next_position_id += 1
        self.open_positions.append(position)
        return position

    def close_position(
        self, index: int, fraction: float, current_price: float
    ) -> float:
        if not (0 <= index < len(self.open_positions)):
            return 0.0
        pos = self.open_positions[index]
        fraction = min(max(fraction, 0.0), 1.0)
        close_size = pos.size * fraction
        pnl = pos.direction * (current_price - pos.entry_price) * close_size
        notional = close_size * current_price
        fee = self._compute_fee(notional, "taker")

        margin_released = pos.margin * fraction
        self.account.release_margin(margin_released)
        self.account.realize_pnl(pnl)
        self.account.apply_fee(fee)

        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        if fraction >= 1.0 - 1e-9:
            self.open_positions.pop(index)
        else:
            pos.size -= close_size
            pos.margin -= margin_released
        return pnl

    def _check_liquidations(self, high: float, low: float) -> list[FillEvent]:
        events: list[FillEvent] = []
        to_remove: list[int] = []
        for i, pos in enumerate(self.open_positions):
            liquidated = False
            if pos.direction == 1 and low <= pos.liquidation_price:
                liquidated = True
            elif pos.direction == -1 and high >= pos.liquidation_price:
                liquidated = True
            if liquidated:
                notional = pos.size * pos.liquidation_price
                fee = self._compute_fee(notional, "taker")
                self.account.apply_fee(fee)
                self.account.release_margin(pos.margin)
                self.account.realize_pnl(-pos.margin)
                self.total_trades += 1
                to_remove.append(i)
                events.append(FillEvent("liquidation", position_id=pos.id, pnl=-pos.margin, fee=fee))
        for i in reversed(to_remove):
            self.open_positions.pop(i)
        return events

    def _check_stop_losses(self, high: float, low: float) -> list[FillEvent]:
        events: list[FillEvent] = []
        to_remove: list[int] = []
        for i, pos in enumerate(self.open_positions):
            hit = False
            if pos.direction == 1 and low <= pos.sl_price:
                hit = True
            elif pos.direction == -1 and high >= pos.sl_price:
                hit = True
            if hit:
                pnl = pos.direction * (pos.sl_price - pos.entry_price) * pos.size
                notional = pos.size * pos.sl_price
                fee = self._compute_fee(notional, "taker")
                self.account.release_margin(pos.margin)
                self.account.realize_pnl(pnl)
                self.account.apply_fee(fee)
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                to_remove.append(i)
                events.append(FillEvent("sl", position_id=pos.id, pnl=pnl, fee=fee))
        for i in reversed(to_remove):
            self.open_positions.pop(i)
        return events

    def _check_take_profits(self, high: float, low: float) -> list[FillEvent]:
        events: list[FillEvent] = []
        to_remove: list[int] = []
        for i, pos in enumerate(self.open_positions):
            tps_hit: list[int] = []
            for j, tp_price in enumerate(pos.tp_prices):
                if pos.direction == 1 and high >= tp_price:
                    tps_hit.append(j)
                elif pos.direction == -1 and low <= tp_price:
                    tps_hit.append(j)
            for j in sorted(tps_hit, reverse=True):
                tp_price = pos.tp_prices[j]
                tp_frac = pos.tp_size_pcts[j]
                close_size = pos.size * tp_frac
                pnl = pos.direction * (tp_price - pos.entry_price) * close_size
                notional = close_size * tp_price
                fee = self._compute_fee(notional, "maker")
                margin_released = pos.margin * tp_frac
                self.account.release_margin(margin_released)
                self.account.realize_pnl(pnl)
                self.account.apply_fee(fee)
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                pos.size -= close_size
                pos.margin -= margin_released
                pos.tp_prices.pop(j)
                pos.tp_size_pcts.pop(j)
                if pos.tp_size_pcts:
                    total = sum(pos.tp_size_pcts)
                    if total > 0:
                        pos.tp_size_pcts = [p / total for p in pos.tp_size_pcts]
                events.append(FillEvent("tp", position_id=pos.id, pnl=pnl, fee=fee))
            if pos.size < 1e-12 or not pos.tp_prices:
                if pos.margin > 0:
                    self.account.release_margin(pos.margin)
                to_remove.append(i)
        for i in reversed(to_remove):
            self.open_positions.pop(i)
        return events

    def _check_order_fills(self, high: float, low: float) -> list[FillEvent]:
        events: list[FillEvent] = []
        to_remove: list[int] = []
        for i, order in enumerate(self.open_orders):
            filled = False
            if order.direction == 1 and low <= order.trigger_price:
                filled = True
            elif order.direction == -1 and high >= order.trigger_price:
                filled = True
            if filled and len(self.open_positions) < self.config.max_open_positions:
                pos = self._fill_order(order)
                to_remove.append(i)
                events.append(FillEvent("fill", position_id=pos.id, order_id=order.id))
        for i in reversed(to_remove):
            self.open_orders.pop(i)
        return events

    def process_candle(
        self, high: float, low: float, close: float
    ) -> list[FillEvent]:
        events: list[FillEvent] = []
        events.extend(self._check_liquidations(high, low))
        events.extend(self._check_stop_losses(high, low))
        events.extend(self._check_take_profits(high, low))
        events.extend(self._check_order_fills(high, low))
        return events

    def total_unrealized_pnl(self, current_price: float) -> float:
        return sum(p.unrealized_pnl(current_price) for p in self.open_positions)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd backend && uv run pytest tests/trainer/test_exchange_sim.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/trainer/env/exchange_sim.py backend/tests/trainer/test_exchange_sim.py
git commit -m "feat(trainer): add exchange simulator with order/position/fill logic"
```

---

### Task 6: Data Feed

**Files:**
- Create: `backend/src/trainer/env/data_feed.py`
- Create: `backend/tests/trainer/test_data_feed.py`

- [ ] **Step 1: Write tests for DataFeed**

Create `backend/tests/trainer/test_data_feed.py`:

```python
import numpy as np
import pytest

from trainer.env.data_feed import DataFeed


@pytest.fixture
def sample_data() -> dict:
    n = 600
    timestamps = list(range(1000, 1000 + n))
    base = np.random.default_rng(42)
    prices = 50000 + base.normal(0, 100, n).cumsum()
    return {
        "timestamps": np.array(timestamps, dtype=np.int64),
        "features": np.column_stack([
            prices,
            prices + base.uniform(10, 200, n),
            prices - base.uniform(10, 200, n),
            prices + base.normal(0, 50, n),
            base.uniform(100, 1000, n),
        ]),
        "columns": ["open", "high", "low", "close", "volume"],
    }


def test_feed_shape(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
    )
    assert feed.total_steps == 100
    obs = feed.get_observation(0)
    assert obs.shape == (500, 5)


def test_feed_normalization(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
    )
    obs = feed.get_observation(0)
    for col in range(obs.shape[1]):
        assert abs(obs[:, col].mean()) < 1.0
        assert obs[:, col].std() < 5.0


def test_feed_stepping(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
    )
    obs0 = feed.get_observation(0)
    obs1 = feed.get_observation(1)
    assert not np.array_equal(obs0, obs1)
    np.testing.assert_array_equal(obs0[1:, :], obs1[:-1, :])


def test_get_candle_prices(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
        price_columns={"open": 0, "high": 1, "low": 2, "close": 3},
    )
    prices = feed.get_candle_prices(0)
    assert "open" in prices
    assert "high" in prices
    assert "low" in prices
    assert "close" in prices


def test_get_timestamp(sample_data):
    feed = DataFeed(
        timestamps=sample_data["timestamps"],
        features=sample_data["features"],
        lookback=500,
    )
    ts = feed.get_timestamp(0)
    assert ts == sample_data["timestamps"][500]
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd backend && uv run pytest tests/trainer/test_data_feed.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement DataFeed**

Create `backend/src/trainer/env/data_feed.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg

from trainer.config import ModelConfig


class DataFeed:
    def __init__(
        self,
        timestamps: np.ndarray,
        features: np.ndarray,
        lookback: int = 500,
        price_columns: dict[str, int] | None = None,
    ) -> None:
        self.timestamps = timestamps
        self.raw_features = features.astype(np.float32)
        self.lookback = lookback
        self.price_columns = price_columns or {}

        self._mean = self.raw_features.mean(axis=0)
        self._std = self.raw_features.std(axis=0)
        self._std[self._std < 1e-8] = 1.0

    @property
    def total_steps(self) -> int:
        return len(self.timestamps) - self.lookback

    @property
    def num_features(self) -> int:
        return self.raw_features.shape[1]

    def get_observation(self, step: int) -> np.ndarray:
        start = step
        end = step + self.lookback
        window = self.raw_features[start:end]
        return ((window - self._mean) / self._std).astype(np.float32)

    def get_raw_observation(self, step: int) -> np.ndarray:
        start = step
        end = step + self.lookback
        return self.raw_features[start:end]

    def get_candle_prices(self, step: int) -> dict[str, float]:
        idx = step + self.lookback
        row = self.raw_features[idx] if idx < len(self.raw_features) else self.raw_features[-1]
        return {name: float(row[col]) for name, col in self.price_columns.items()}

    def get_current_raw(self, step: int) -> np.ndarray:
        idx = step + self.lookback - 1
        return self.raw_features[idx]

    def get_timestamp(self, step: int) -> int:
        return int(self.timestamps[step + self.lookback])


def load_data_feed(config: ModelConfig, conn: psycopg.Connection) -> DataFeed:
    primary_interval = config.intervals[0]

    dfs: list[pd.DataFrame] = []
    for symbol in config.symbols:
        rows = conn.execute(
            """
            SELECT open_time, {} FROM klines
            WHERE symbol = %s AND interval = %s
            ORDER BY open_time
            """.format(", ".join(config.columns)),
            (symbol, primary_interval),
        ).fetchall()

        col_names = ["open_time"] + [f"{symbol}_{c}" for c in config.columns]
        df = pd.DataFrame(rows, columns=col_names)
        df = df.set_index("open_time")
        dfs.append(df)

    if len(dfs) == 1:
        merged = dfs[0]
    else:
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.join(df, how="inner")

    merged = merged.sort_index()

    timestamps = merged.index.values.astype(np.int64)
    features = merged.values.astype(np.float32)

    primary_symbol = config.symbols[0]
    price_columns: dict[str, int] = {}
    for name in ["open", "high", "low", "close"]:
        col_name = f"{primary_symbol}_{name}"
        if col_name in merged.columns:
            price_columns[name] = list(merged.columns).index(col_name)

    return DataFeed(
        timestamps=timestamps,
        features=features,
        lookback=config.lookback_window,
        price_columns=price_columns,
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd backend && uv run pytest tests/trainer/test_data_feed.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/trainer/env/data_feed.py backend/tests/trainer/test_data_feed.py
git commit -m "feat(trainer): add DataFeed for loading and normalizing kline observations"
```

---

### Task 7: Trading Environment (Gymnasium)

**Files:**
- Create: `backend/src/trainer/env/trading_env.py`
- Create: `backend/tests/trainer/test_trading_env.py`

- [ ] **Step 1: Write tests for TradingEnv**

Create `backend/tests/trainer/test_trading_env.py`:

```python
import gymnasium as gym
import numpy as np
import pytest

from trainer.config import ExchangeConfig, ModelConfig
from trainer.env.data_feed import DataFeed
from trainer.env.trading_env import TradingEnv


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        name="test",
        symbols=["BTCUSDT"],
        intervals=["1h"],
        columns=["open", "high", "low", "close", "volume"],
        exchange=ExchangeConfig(max_open_orders=5, max_open_positions=5),
        lookback_window=50,
        num_tp_levels=3,
    )


@pytest.fixture
def feed() -> DataFeed:
    n = 200
    rng = np.random.default_rng(42)
    base_price = 50_000.0
    closes = base_price + rng.normal(0, 100, n).cumsum()
    highs = closes + rng.uniform(50, 300, n)
    lows = closes - rng.uniform(50, 300, n)
    opens = closes + rng.normal(0, 50, n)
    volume = rng.uniform(100, 1000, n)

    features = np.column_stack([opens, highs, lows, closes, volume])
    timestamps = np.arange(n, dtype=np.int64)

    return DataFeed(
        timestamps=timestamps,
        features=features,
        lookback=50,
        price_columns={"open": 0, "high": 1, "low": 2, "close": 3},
    )


@pytest.fixture
def env(config: ModelConfig, feed: DataFeed) -> TradingEnv:
    return TradingEnv(config=config, data_feed=feed)


def test_env_is_gymnasium_compliant(env: TradingEnv):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "market" in obs
    assert "account" in obs
    assert "orders" in obs
    assert "positions" in obs


def test_observation_shapes(env: TradingEnv, config: ModelConfig):
    obs, _ = env.reset()
    assert obs["market"].shape == (50, 5)
    assert obs["account"].shape == (5,)
    assert obs["orders"].shape == (5, 11)
    assert obs["positions"].shape == (5, 6)


def test_action_space_shape(env: TradingEnv):
    assert env.action_space.shape == (env.config.action_size,)


def test_do_nothing_action(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert isinstance(reward, float)


def test_step_advances_candle(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    env.step(action)
    assert env._current_step == 1


def test_episode_truncates_at_end(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    terminated = False
    truncated = False
    steps = 0
    while not terminated and not truncated:
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
    assert truncated or terminated
    assert steps == env.data_feed.total_steps


def test_open_order_action(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    action[0] = 0.9   # open_confidence > 0.5
    action[1] = 0.8   # direction = long
    action[2] = -0.1   # trigger slightly below current
    action[3] = 0.3   # SL distance
    action[4] = 0.3   # TP1 distance
    action[5] = 0.5   # TP2 distance
    action[6] = 0.7   # TP3 distance
    action[7] = 0.33  # TP1 size
    action[8] = 0.33  # TP2 size
    action[9] = 0.34  # TP3 size
    action[10] = 0.1  # 10% of balance
    obs, _, _, _, info = env.step(action)
    assert info.get("orders_placed", 0) >= 0


def test_reset_returns_fresh_state(env: TradingEnv):
    env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(10):
        env.step(action)
    obs, _ = env.reset()
    assert env._current_step == 0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd backend && uv run pytest tests/trainer/test_trading_env.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement TradingEnv**

Create `backend/src/trainer/env/trading_env.py`:

```python
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trainer.config import ModelConfig
from trainer.env.account import Account
from trainer.env.data_feed import DataFeed
from trainer.env.exchange_sim import ExchangeSim


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ModelConfig, data_feed: DataFeed) -> None:
        super().__init__()
        self.config = config
        self.data_feed = data_feed

        self.account = Account(initial_balance=config.initial_balance)
        self.exchange = ExchangeSim(config=config.exchange, account=self.account)

        exc = config.exchange
        num_features = data_feed.num_features
        lookback = config.lookback_window
        num_tp = config.num_tp_levels

        self.observation_space = spaces.Dict({
            "market": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback, num_features), dtype=np.float32,
            ),
            "account": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(5,), dtype=np.float32,
            ),
            "orders": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(exc.max_open_orders, 11), dtype=np.float32,
            ),
            "positions": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(exc.max_open_positions, 6), dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(config.action_size,), dtype=np.float32,
        )

        self._current_step = 0
        self._prev_equity = config.initial_balance
        self.pnl_history: list[dict] = []

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._current_step = 0
        self._prev_equity = self.config.initial_balance
        self.exchange.reset()
        self.pnl_history.clear()
        return self._build_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}

        prices = self.data_feed.get_candle_prices(self._current_step)
        high = prices.get("high", prices.get("close", 0.0))
        low = prices.get("low", prices.get("close", 0.0))
        close = prices.get("close", 0.0)

        if close <= 0:
            close = 1.0
            high = max(high, 1.0)
            low = max(low, 0.1)

        events = self.exchange.process_candle(high=high, low=low, close=close)
        info["fill_events"] = len(events)

        orders_placed = 0
        orders_cancelled = 0
        positions_closed = 0
        self._process_actions(action, close, info)
        orders_placed = info.get("orders_placed", 0)
        orders_cancelled = info.get("orders_cancelled", 0)
        positions_closed = info.get("positions_closed", 0)

        unrealized = self.exchange.total_unrealized_pnl(close)
        equity = self.account.equity(unrealized)
        reward = float(equity - self._prev_equity)
        self._prev_equity = equity

        self.pnl_history.append({
            "step": self._current_step,
            "candle_time": self.data_feed.get_timestamp(self._current_step),
            "balance": self.account.balance,
            "equity": equity,
            "unrealized_pnl": unrealized,
            "open_position_count": len(self.exchange.open_positions),
            "open_order_count": len(self.exchange.open_orders),
        })

        terminated = equity <= 0
        self._current_step += 1
        truncated = self._current_step >= self.data_feed.total_steps

        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    def _process_actions(
        self, action: np.ndarray, close: float, info: dict[str, Any]
    ) -> None:
        num_tp = self.config.num_tp_levels
        exc = self.config.exchange

        idx = 1 + 1 + 1 + 1 + num_tp + num_tp + 1
        cancel_start = idx
        cancel_end = cancel_start + exc.max_open_orders
        close_start = cancel_end
        close_end = close_start + exc.max_open_positions

        cancelled = 0
        cancel_signals = action[cancel_start:cancel_end]
        indices_to_cancel = [
            i for i in range(len(self.exchange.open_orders))
            if i < len(cancel_signals) and cancel_signals[i] > 0.0
        ]
        for i in sorted(indices_to_cancel, reverse=True):
            self.exchange.cancel_order(i)
            cancelled += 1
        info["orders_cancelled"] = cancelled

        closed = 0
        close_signals = action[close_start:close_end]
        for i in range(min(len(self.exchange.open_positions), len(close_signals))):
            frac = float(max(0.0, min(1.0, (close_signals[i] + 1.0) / 2.0)))
            if frac > 0.05:
                self.exchange.close_position(i, frac, close)
                closed += 1
        info["positions_closed"] = closed

        open_conf = (action[0] + 1.0) / 2.0
        placed = 0
        if open_conf > 0.5:
            direction = 1 if action[1] > 0.0 else -1

            offset_pct = action[2] * self.config.max_trigger_offset_pct / 100.0
            trigger_price = close * (1.0 + offset_pct)

            sl_raw = (action[3] + 1.0) / 2.0
            sl_dist_pct = (
                self.config.min_sl_pct
                + sl_raw * (self.config.max_sl_pct - self.config.min_sl_pct)
            ) / 100.0
            if direction == 1:
                sl_price = trigger_price * (1.0 - sl_dist_pct)
            else:
                sl_price = trigger_price * (1.0 + sl_dist_pct)

            tp_prices: list[float] = []
            tp_size_pcts: list[float] = []
            raw_tp_sizes: list[float] = []
            for j in range(num_tp):
                tp_raw = (action[4 + j] + 1.0) / 2.0
                tp_dist_pct = tp_raw * self.config.max_tp_pct / 100.0
                tp_dist_pct = max(tp_dist_pct, 0.001)
                if direction == 1:
                    tp_price = trigger_price * (1.0 + tp_dist_pct)
                else:
                    tp_price = trigger_price * (1.0 - tp_dist_pct)
                tp_prices.append(tp_price)
                raw_size = max((action[4 + num_tp + j] + 1.0) / 2.0, 0.01)
                raw_tp_sizes.append(raw_size)

            total = sum(raw_tp_sizes)
            tp_size_pcts = [s / total for s in raw_tp_sizes]

            size_raw = (action[4 + 2 * num_tp] + 1.0) / 2.0
            margin = size_raw * self.account.available_balance

            if margin >= self.config.exchange.min_order_size_usd:
                order = self.exchange.place_order(
                    direction=direction,
                    trigger_price=trigger_price,
                    sl_price=sl_price,
                    tp_prices=tp_prices,
                    tp_size_pcts=tp_size_pcts,
                    margin=margin,
                )
                if order is not None:
                    placed = 1
        info["orders_placed"] = placed

    def _build_observation(self) -> dict[str, np.ndarray]:
        step = min(self._current_step, self.data_feed.total_steps - 1)
        market = self.data_feed.get_observation(step)

        raw = self.data_feed.get_current_raw(step)
        close = float(raw[self.data_feed.price_columns.get("close", 3)])
        if close <= 0:
            close = 1.0

        init = self.config.initial_balance
        unrealized = self.exchange.total_unrealized_pnl(close)
        account_state = np.array([
            self.account.balance / init,
            self.account.equity(unrealized) / init,
            unrealized / init,
            self.account.margin_used / init,
            self.account.available_balance / init,
        ], dtype=np.float32)

        exc = self.config.exchange
        orders = np.zeros((exc.max_open_orders, 11), dtype=np.float32)
        for i, order in enumerate(self.exchange.open_orders[:exc.max_open_orders]):
            orders[i, 0] = 1.0
            orders[i, 1] = float(order.direction)
            orders[i, 2] = order.trigger_price / close
            orders[i, 3] = order.sl_price / close
            for j, tp in enumerate(order.tp_prices[:3]):
                orders[i, 4 + j] = tp / close
            for j, pct in enumerate(order.tp_size_pcts[:3]):
                orders[i, 7 + j] = pct
            orders[i, 10] = order.margin / init

        positions = np.zeros((exc.max_open_positions, 6), dtype=np.float32)
        for i, pos in enumerate(self.exchange.open_positions[:exc.max_open_positions]):
            positions[i, 0] = 1.0
            positions[i, 1] = float(pos.direction)
            positions[i, 2] = pos.entry_price / close
            positions[i, 3] = pos.size * pos.entry_price / init
            positions[i, 4] = pos.unrealized_pnl(close) / init
            positions[i, 5] = pos.leverage / exc.max_leverage

        return {
            "market": market,
            "account": account_state,
            "orders": orders,
            "positions": positions,
        }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd backend && uv run pytest tests/trainer/test_trading_env.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run all tests together**

```bash
cd backend && uv run pytest tests/trainer/ -v
```

Expected: all tests across all test files PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/trainer/env/trading_env.py backend/tests/trainer/test_trading_env.py
git commit -m "feat(trainer): add Gymnasium TradingEnv wrapping exchange sim + data feed"
```

---

### Task 8: Training Pipeline

**Files:**
- Create: `backend/src/trainer/training/trainer.py`

- [ ] **Step 1: Implement trainer.py**

Create `backend/src/trainer/training/trainer.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)

from ingester.db import connect
from trainer.config import ModelConfig
from trainer.db import (
    complete_training_run,
    create_training_run,
    fail_training_run,
    get_model_config_id,
    save_pnl_snapshots,
)
from trainer.env.data_feed import DataFeed, load_data_feed
from trainer.env.trading_env import TradingEnv

ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
}

MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "trained_models"


class PnlSnapshotCallback(BaseCallback):
    def __init__(
        self, env: TradingEnv, run_id: int, interval: int = 100, verbose: int = 0
    ) -> None:
        super().__init__(verbose)
        self.env = env
        self.run_id = run_id
        self.interval = interval
        self._buffer: list[dict] = []
        self._last_flushed = 0

    def _on_step(self) -> bool:
        if self.env.pnl_history and len(self.env.pnl_history) % self.interval == 0:
            new_entries = self.env.pnl_history[self._last_flushed:]
            for entry in new_entries:
                self._buffer.append({**entry, "training_run_id": self.run_id})
            self._last_flushed = len(self.env.pnl_history)

            if len(self._buffer) >= 500:
                self._flush()
        return True

    def _flush(self) -> None:
        if not self._buffer:
            return
        conn = connect()
        try:
            save_pnl_snapshots(conn, self._buffer)
            self._buffer.clear()
        finally:
            conn.close()

    def _on_training_end(self) -> None:
        new_entries = self.env.pnl_history[self._last_flushed:]
        for entry in new_entries:
            self._buffer.append({**entry, "training_run_id": self.run_id})
        self._flush()


def compute_metrics(env: TradingEnv) -> dict:
    if not env.pnl_history:
        return {
            "final_balance": env.account.balance,
            "final_equity": env.account.balance,
            "total_pnl": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    equities = [h["equity"] for h in env.pnl_history]
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    import numpy as np
    returns = np.diff(equities) / (np.array(equities[:-1]) + 1e-9)
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 1e-9:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24))

    return {
        "final_balance": env.account.balance,
        "final_equity": equities[-1] if equities else env.account.balance,
        "total_pnl": equities[-1] - env.config.initial_balance if equities else 0.0,
        "total_trades": env.exchange.total_trades,
        "win_rate": env.exchange.win_rate,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
    }


def train_model(
    config: ModelConfig,
    *,
    algo_override: str | None = None,
    timesteps_override: int | None = None,
) -> int:
    algorithm = algo_override or config.algorithm
    total_timesteps = timesteps_override or config.total_timesteps

    algo_cls = ALGO_MAP.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use one of: {list(ALGO_MAP)}")

    config_id = get_model_config_id(config.name)
    if config_id is None:
        raise ValueError(f"Model '{config.name}' not found in DB. Run create-model first.")

    run_id = create_training_run(config_id, "train", algorithm)
    print(f"Training run #{run_id} started: model={config.name} algo={algorithm} steps={total_timesteps}")

    try:
        conn = connect()
        try:
            data_feed = load_data_feed(config, conn)
        finally:
            conn.close()

        split_idx = int(data_feed.total_steps * 0.8)
        train_timestamps = data_feed.timestamps[: split_idx + config.lookback_window]
        train_features = data_feed.raw_features[: split_idx + config.lookback_window]

        train_feed = DataFeed(
            timestamps=train_timestamps,
            features=train_features,
            lookback=config.lookback_window,
            price_columns=data_feed.price_columns,
        )

        env = TradingEnv(config=config, data_feed=train_feed)

        model_dir = MODELS_DIR / config.name / str(run_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = CheckpointCallback(
            save_freq=100_000,
            save_path=str(model_dir),
            name_prefix="checkpoint",
        )
        pnl_cb = PnlSnapshotCallback(
            env=env, run_id=run_id, interval=config.snapshot_interval
        )

        model = algo_cls(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            verbose=1,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, pnl_cb],
        )

        model_path = str(model_dir / "model.zip")
        model.save(model_path)

        metrics = compute_metrics(env)
        complete_training_run(run_id, model_path=model_path, **metrics)

        print(f"Training run #{run_id} completed.")
        print(f"  Final equity: ${metrics['final_equity']:.2f}")
        print(f"  Total PnL:    ${metrics['total_pnl']:.2f}")
        print(f"  Win rate:     {metrics['win_rate']*100:.1f}%")
        print(f"  Max drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Model saved:  {model_path}")
        return run_id

    except Exception as e:
        fail_training_run(run_id, str(e))
        print(f"Training run #{run_id} FAILED: {e}")
        raise
```

- [ ] **Step 2: Verify imports resolve**

```bash
cd backend && uv run python -c "from trainer.training.trainer import train_model, ALGO_MAP; print('Algos:', list(ALGO_MAP))"
```

Expected: `Algos: ['PPO', 'SAC', 'A2C']`

- [ ] **Step 3: Commit**

```bash
git add backend/src/trainer/training/trainer.py
git commit -m "feat(trainer): add SB3 training pipeline with PnL snapshot callbacks"
```

---

### Task 9: Evaluator

**Files:**
- Create: `backend/src/trainer/training/evaluator.py`

- [ ] **Step 1: Implement evaluator.py**

Create `backend/src/trainer/training/evaluator.py`:

```python
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import A2C, PPO, SAC

from ingester.db import connect
from trainer.config import ModelConfig
from trainer.db import (
    complete_training_run,
    create_training_run,
    fail_training_run,
    get_model_config_id,
    save_pnl_snapshots,
)
from trainer.env.data_feed import DataFeed, load_data_feed
from trainer.env.trading_env import TradingEnv
from trainer.training.trainer import ALGO_MAP, MODELS_DIR, compute_metrics


def evaluate_model(
    config: ModelConfig,
    model_path: str,
    algorithm: str | None = None,
) -> int:
    algorithm = algorithm or config.algorithm

    algo_cls = ALGO_MAP.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    config_id = get_model_config_id(config.name)
    if config_id is None:
        raise ValueError(f"Model '{config.name}' not found in DB.")

    run_id = create_training_run(config_id, "evaluate", algorithm)
    print(f"Evaluation run #{run_id}: model={config.name} path={model_path}")

    try:
        conn = connect()
        try:
            data_feed = load_data_feed(config, conn)
        finally:
            conn.close()

        split_idx = int(data_feed.total_steps * 0.8)
        holdout_start = split_idx
        holdout_timestamps = data_feed.timestamps[holdout_start:]
        holdout_features = data_feed.raw_features[holdout_start:]

        holdout_feed = DataFeed(
            timestamps=holdout_timestamps,
            features=holdout_features,
            lookback=config.lookback_window,
            price_columns=data_feed.price_columns,
        )

        env = TradingEnv(config=config, data_feed=holdout_feed)
        model = algo_cls.load(model_path, env=env)

        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

        snapshots = []
        for entry in env.pnl_history[::config.snapshot_interval]:
            snapshots.append({**entry, "training_run_id": run_id})
        if snapshots:
            conn = connect()
            try:
                save_pnl_snapshots(conn, snapshots)
            finally:
                conn.close()

        metrics = compute_metrics(env)
        eval_model_path = str(MODELS_DIR / config.name / str(run_id) / "eval_reference.txt")
        Path(eval_model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(eval_model_path).write_text(f"Evaluated from: {model_path}\n")

        complete_training_run(run_id, model_path=model_path, **metrics)

        print(f"Evaluation run #{run_id} completed.")
        print(f"  Final equity: ${metrics['final_equity']:.2f}")
        print(f"  Total PnL:    ${metrics['total_pnl']:.2f}")
        print(f"  Win rate:     {metrics['win_rate']*100:.1f}%")
        print(f"  Max drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        return run_id

    except Exception as e:
        fail_training_run(run_id, str(e))
        print(f"Evaluation run #{run_id} FAILED: {e}")
        raise
```

- [ ] **Step 2: Verify imports resolve**

```bash
cd backend && uv run python -c "from trainer.training.evaluator import evaluate_model; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/trainer/training/evaluator.py
git commit -m "feat(trainer): add model evaluator for holdout data backtesting"
```

---

### Task 10: CLI

**Files:**
- Create: `backend/src/trainer/cli.py`

- [ ] **Step 1: Implement cli.py**

Create `backend/src/trainer/cli.py`:

```python
from __future__ import annotations

import argparse
import sys

from trainer.config import ModelConfig
from trainer.db import (
    get_training_run,
    list_model_configs,
    load_model_config,
    save_model_config,
)
from trainer.models.btc_config import make_btc_config
from trainer.models.sol_config import make_sol_config

BUILTIN_CONFIGS = {
    "btc": make_btc_config,
    "sol": make_sol_config,
}


def cmd_create_model(args: argparse.Namespace) -> None:
    name = args.config
    factory = BUILTIN_CONFIGS.get(name)
    if factory is None:
        print(f"Unknown config: {name}. Available: {list(BUILTIN_CONFIGS)}")
        sys.exit(1)
    config = factory()
    config_id = save_model_config(config)
    print(f"Model '{config.name}' registered (id={config_id}).")


def cmd_start(args: argparse.Namespace) -> None:
    from trainer.training.trainer import train_model

    config = load_model_config(args.model)
    if config is None:
        print(f"Model '{args.model}' not found. Run create-model first.")
        sys.exit(1)

    train_model(
        config,
        algo_override=args.algo,
        timesteps_override=args.timesteps,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    from trainer.training.evaluator import evaluate_model

    config = load_model_config(args.model)
    if config is None:
        print(f"Model '{args.model}' not found.")
        sys.exit(1)

    run = get_training_run(args.run)
    if run is None:
        print(f"Training run #{args.run} not found.")
        sys.exit(1)

    model_path = run.get("model_path")
    if not model_path:
        print(f"Run #{args.run} has no saved model.")
        sys.exit(1)

    evaluate_model(config, model_path, algorithm=run.get("algorithm"))


def cmd_list(_args: argparse.Namespace) -> None:
    models = list_model_configs()
    if not models:
        print("No models registered.")
        return
    print(f"{'Name':<20} {'Runs':>6} {'Best PnL':>12} {'Created'}")
    print("-" * 60)
    for m in models:
        pnl = f"${m['best_pnl']:.2f}" if m["best_pnl"] is not None else "—"
        created = m["created_at"].strftime("%Y-%m-%d %H:%M") if m["created_at"] else "—"
        print(f"{m['name']:<20} {m['run_count']:>6} {pnl:>12} {created}")


def cmd_status(args: argparse.Namespace) -> None:
    run = get_training_run(args.run)
    if run is None:
        print(f"Training run #{args.run} not found.")
        sys.exit(1)
    print(f"Run #{run['id']}:")
    print(f"  Model:        {run.get('model_name', '?')}")
    print(f"  Type:         {run['run_type']}")
    print(f"  Algorithm:    {run['algorithm']}")
    print(f"  Status:       {run['status']}")
    print(f"  Started:      {run['started_at']}")
    if run.get("completed_at"):
        print(f"  Completed:    {run['completed_at']}")
    if run.get("final_equity") is not None:
        print(f"  Final equity: ${float(run['final_equity']):.2f}")
        print(f"  Total PnL:    ${float(run['total_pnl']):.2f}")
        print(f"  Win rate:     {float(run['win_rate'])*100:.1f}%")
        print(f"  Max drawdown: {float(run['max_drawdown'])*100:.1f}%")
        print(f"  Sharpe ratio: {float(run['sharpe_ratio']):.2f}")
        print(f"  Total trades: {run['total_trades']}")
    if run.get("model_path"):
        print(f"  Model:        {run['model_path']}")
    if run.get("error"):
        print(f"  Error:        {run['error']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Trading model trainer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cm = sub.add_parser("create-model", help="Register a model config")
    cm.add_argument(
        "--config", required=True,
        help=f"Built-in config name: {list(BUILTIN_CONFIGS)}",
    )

    st = sub.add_parser("start", help="Start a training run")
    st.add_argument("--model", required=True, help="Model name (registered)")
    st.add_argument("--algo", default=None, help="Algorithm override (PPO, SAC, A2C)")
    st.add_argument("--timesteps", type=int, default=None, help="Total timesteps override")

    ev = sub.add_parser("evaluate", help="Evaluate a trained model on holdout data")
    ev.add_argument("--model", required=True, help="Model name")
    ev.add_argument("--run", type=int, required=True, help="Training run ID to evaluate")

    sub.add_parser("list", help="List all registered models")

    sr = sub.add_parser("status", help="Show details for a training run")
    sr.add_argument("--run", type=int, required=True, help="Run ID")

    return parser


_COMMANDS = {
    "create-model": cmd_create_model,
    "start": cmd_start,
    "evaluate": cmd_evaluate,
    "list": cmd_list,
    "status": cmd_status,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

```bash
cd backend && uv run train --help
```

Expected: shows help with subcommands: create-model, start, evaluate, list, status.

- [ ] **Step 3: Verify subcommand help**

```bash
cd backend && uv run train start --help
```

Expected: shows --model, --algo, --timesteps options.

- [ ] **Step 4: Commit**

```bash
git add backend/src/trainer/cli.py
git commit -m "feat(trainer): add CLI with create-model, start, evaluate, list, status commands"
```

---

### Task 11: Integration Smoke Test

Verifies the full pipeline works end-to-end by registering a model, confirming it writes to the DB, and running a short training session on synthetic data (no real DB klines needed for the config registration part; the training part requires klines data in the database).

- [ ] **Step 1: Test model registration**

```bash
cd backend && uv run train create-model --config btc
```

Expected: `Model 'btc_v1' registered (id=1).`

```bash
cd backend && uv run train create-model --config sol
```

Expected: `Model 'sol_v1' registered (id=2).`

- [ ] **Step 2: Test model listing**

```bash
cd backend && uv run train list
```

Expected:
```
Name                   Runs     Best PnL Created
------------------------------------------------------------
btc_v1                    0            — 2026-04-...
sol_v1                    0            — 2026-04-...
```

- [ ] **Step 3: Test short training run (requires BTCUSDT klines in DB)**

Only run this if klines data has been ingested. This trains for just 1000 timesteps as a smoke test:

```bash
cd backend && uv run train start --model btc_v1 --timesteps 1000
```

Expected: prints training progress, then summary with final equity, PnL, win rate, etc. Training run is recorded in `training_runs` table.

- [ ] **Step 4: Check run status**

```bash
cd backend && uv run train status --run 1
```

Expected: shows full details for the run.

- [ ] **Step 5: Run all unit tests one final time**

```bash
cd backend && uv run pytest tests/trainer/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Add trained_models to .gitignore**

Append to `backend/.gitignore` (create if it doesn't exist):

```
trained_models/
```

- [ ] **Step 7: Final commit**

```bash
git add backend/.gitignore
git commit -m "chore: add trained_models to gitignore"
```

---

## Summary

| Task | Component | Key Files |
|------|-----------|-----------|
| 1 | Project setup | `pyproject.toml`, package dirs |
| 2 | Config dataclasses | `config.py`, `btc_config.py`, `sol_config.py` |
| 3 | Database migration & DB ops | `003_trainer_tables.sql`, `db.py` |
| 4 | Account tracker | `account.py`, `test_account.py` |
| 5 | Exchange simulator | `exchange_sim.py`, `test_exchange_sim.py` |
| 6 | Data feed | `data_feed.py`, `test_data_feed.py` |
| 7 | Trading environment | `trading_env.py`, `test_trading_env.py` |
| 8 | Training pipeline | `trainer.py` |
| 9 | Evaluator | `evaluator.py` |
| 10 | CLI | `cli.py` |
| 11 | Integration smoke test | end-to-end verification |
