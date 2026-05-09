# Live Testing on BingX VST — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the framework that runs trained SB3 models against real-time BingX VST data, with parity to the trainer's eval, persistent state, supervised by systemd, and gated by a 0%-divergence replay test before any pick goes live.

**Architecture:** Extract observation building, action decoding, and the fill simulator from `trainer/env/` into pure modules consumed by both `TradingEnv` (training/eval) and a new `live/` package (production). Live state is stored in four new `live_*` Postgres tables. A `ReplayAdapter` shares the simulator with `ExchangeSim` so a `live_replay` script can reproduce any trainer eval bit-for-bit before that pick's systemd unit is enabled.

**Tech Stack:** Python 3.12+, gymnasium, stable-baselines3, ccxt (BingX swap unified `BTC/USDT:USDT`), psycopg3, Pydantic, PyYAML, pytest, systemd.

**Spec:** `docs/superpowers/specs/2026-05-09-live-testing-bingx-design.md`

---

## File map

Created in this plan:

```
backend/migrations/006_live_testing_tables.sql

backend/src/trainer/env/observation.py       # NEW — pure: build observation dict
backend/src/trainer/env/action_decoder.py    # NEW — pure: decode action vector → OrderIntent
backend/src/trainer/env/normalization.py     # NEW — fit/save/load mean+std for DataFeed

backend/src/live/__init__.py
backend/src/live/config.py
backend/src/live/db.py
backend/src/live/exchange/__init__.py
backend/src/live/exchange/base.py
backend/src/live/exchange/registry.py
backend/src/live/exchange/replay.py
backend/src/live/exchange/bingx.py
backend/src/live/feature_pipeline.py
backend/src/live/action_decoder.py
backend/src/live/model_runner.py
backend/src/live/runner.py
backend/src/live/reconciliation.py
backend/src/live/cli.py

backend/scripts/live_replay.py

backend/configs/live/live-s1.yaml
backend/configs/live/live-s2.yaml
backend/configs/live/live-s3.yaml

backend/tests/trainer/env/test_action_decoder.py
backend/tests/trainer/env/test_observation.py
backend/tests/trainer/env/test_apply_intent.py
backend/tests/trainer/env/test_refactor_parity.py
backend/tests/live/test_config.py
backend/tests/live/test_db.py
backend/tests/live/test_registry.py
backend/tests/live/test_replay_adapter.py
backend/tests/live/test_action_decoder_clamp.py
backend/tests/live/test_reconciliation.py

infra/systemd/tradan-live@.service
infra/scripts/live_runner_deploy.sh
```

Modified in this plan:

```
backend/pyproject.toml                       # add pyyaml, pydantic; register live-test, live-replay
backend/src/trainer/env/data_feed.py         # accept persisted normalization stats
backend/src/trainer/env/exchange_sim.py      # gain apply_intent()
backend/src/trainer/env/trading_env.py       # delegate to extracted modules
backend/src/trainer/training/trainer.py      # save normalization stats alongside model
```

---

## Phase A — Trainer/env extraction

The trainer's `_build_observation` and `_process_actions` move into pure modules. Both `TradingEnv.step` and `live/runner` will call the same code. Before any `live/` code is written, this phase must pass Gate A1 (before/after eval bit-identical regression).

### Task A.1: Investigate normalization state and write parity-bug fix design

**Files:**
- Read: `backend/src/trainer/env/data_feed.py`
- Read: `backend/src/trainer/training/trainer.py` (find where DataFeed is constructed)
- Read: `backend/src/trainer/training/evaluator.py` (find where DataFeed is constructed for eval)
- Create: `docs/superpowers/notes/normalization-parity.md` (research notes; deleted at end of phase)

The brainstorming uncovered that `DataFeed` computes `_mean`/`_std` from whatever feature array it is constructed with. If train and eval load different slices, their normalization differs. Live will differ again. We must persist train-time stats and load them at eval and live.

- [ ] **Step 1: Confirm the bug exists**

```bash
grep -n "DataFeed" backend/src/trainer/training/trainer.py backend/src/trainer/training/evaluator.py
grep -n "load_data_feed" backend/src/trainer/training/trainer.py backend/src/trainer/training/evaluator.py
```

Read each call site. Document in `docs/superpowers/notes/normalization-parity.md`:
- Which slice (train? holdout? full?) is passed to `DataFeed.__init__` in each context.
- Whether `_mean`/`_std` differ across train and eval today.

- [ ] **Step 2: Decide where stats live**

Stats persist as `mean.npy` and `std.npy` next to the SB3 model `.zip` (same directory the trainer already writes to). Write the decision into the notes file. Reasoning:
- Already-shipped trainer models did not save stats. We will fail loudly if stats files are missing — eval and live both refuse to run without them.
- Decision rejected: stash stats in `model_configs` row. Reason: stats are large (one float per feature × num_features ≈ 9), but we may add features later; filesystem stays cleaner.

- [ ] **Step 3: Commit notes**

```bash
git add docs/superpowers/notes/normalization-parity.md
git commit -m "docs: investigate trainer normalization parity bug"
```

### Task A.2: Add normalization persistence module

**Files:**
- Create: `backend/src/trainer/env/normalization.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/trainer/env/test_normalization.py`

- [ ] **Step 1: Add new test subdirs (existing layout already works)**

`backend/tests/` already exists with 46 passing tests under `tests/trainer/`. The project is installed via `uv sync`, so `from trainer....` imports resolve through the installed package — no conftest is needed for path setup. Only create the new subdirs:

```bash
mkdir -p backend/tests/trainer/env backend/tests/live
touch backend/tests/trainer/env/__init__.py backend/tests/live/__init__.py
```

(`tests/__init__.py` and `tests/trainer/__init__.py` already exist; do not overwrite them.) Skip creating `tests/conftest.py` — the existing test files (`tests/trainer/test_account.py` etc.) already work without one.

- [ ] **Step 2: Write the failing test**

`backend/tests/trainer/env/test_normalization.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trainer.env.normalization import (
    NormalizationStats,
    fit_stats,
    load_stats,
    save_stats,
)


def test_fit_stats_returns_per_feature_mean_std():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(1000, 7)).astype(np.float32)
    stats = fit_stats(features)
    assert stats.mean.shape == (7,)
    assert stats.std.shape == (7,)
    np.testing.assert_allclose(stats.mean, features.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(stats.std, features.std(axis=0), atol=1e-6)


def test_fit_stats_replaces_zero_std_with_one():
    features = np.ones((100, 3), dtype=np.float32)
    stats = fit_stats(features)
    np.testing.assert_array_equal(stats.std, np.ones(3, dtype=np.float32))


def test_save_and_load_roundtrip(tmp_path: Path):
    stats = NormalizationStats(
        mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        std=np.array([0.5, 1.0, 2.0], dtype=np.float32),
    )
    save_stats(stats, tmp_path / "model")
    loaded = load_stats(tmp_path / "model")
    np.testing.assert_array_equal(loaded.mean, stats.mean)
    np.testing.assert_array_equal(loaded.std, stats.std)


def test_load_stats_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_stats(tmp_path / "nope")
```

- [ ] **Step 3: Run test, verify failure**

```bash
cd backend && uv run pytest tests/trainer/env/test_normalization.py -v
```

Expected: `ModuleNotFoundError: No module named 'trainer.env.normalization'`.

- [ ] **Step 4: Implement the module**

`backend/src/trainer/env/normalization.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    """Per-feature mean and std used to normalize DataFeed observations.

    Computed once at training time from the training feature array, then
    reused at eval and live inference so the model sees the same input
    distribution it was trained on.
    """

    mean: np.ndarray  # shape (num_features,), dtype float32
    std: np.ndarray   # shape (num_features,), dtype float32, no zeros


def fit_stats(features: np.ndarray) -> NormalizationStats:
    """Compute per-feature mean and std; clamp zero std to 1.0."""
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return NormalizationStats(mean=mean, std=std)


def save_stats(stats: NormalizationStats, model_path_no_ext: Path) -> None:
    """Save mean.npy and std.npy next to a model file (without extension)."""
    base = Path(model_path_no_ext)
    base.parent.mkdir(parents=True, exist_ok=True)
    np.save(base.with_suffix(".mean.npy"), stats.mean)
    np.save(base.with_suffix(".std.npy"), stats.std)


def load_stats(model_path_no_ext: Path) -> NormalizationStats:
    """Load saved stats. Raises FileNotFoundError if either file missing."""
    base = Path(model_path_no_ext)
    mean_path = base.with_suffix(".mean.npy")
    std_path = base.with_suffix(".std.npy")
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {mean_path}, {std_path}"
        )
    return NormalizationStats(
        mean=np.load(mean_path).astype(np.float32),
        std=np.load(std_path).astype(np.float32),
    )
```

- [ ] **Step 5: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/trainer/env/test_normalization.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/tests backend/src/trainer/env/normalization.py
git commit -m "feat(trainer): add NormalizationStats persistence module"
```

### Task A.3: Wire DataFeed to accept persisted stats

**Files:**
- Modify: `backend/src/trainer/env/data_feed.py`
- Create: `backend/tests/trainer/env/test_data_feed.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/trainer/env/test_data_feed.py`:

```python
from __future__ import annotations

import numpy as np

from trainer.env.data_feed import DataFeed
from trainer.env.normalization import NormalizationStats


def test_data_feed_uses_provided_stats():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3)).astype(np.float32)
    timestamps = np.arange(200, dtype=np.int64)

    custom_stats = NormalizationStats(
        mean=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        std=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    feed = DataFeed(
        timestamps=timestamps,
        features=features,
        lookback=10,
        stats=custom_stats,
    )

    obs = feed.get_observation(0)
    expected = (features[:10] - custom_stats.mean) / custom_stats.std
    np.testing.assert_allclose(obs, expected.astype(np.float32), atol=1e-6)


def test_data_feed_fits_stats_when_none_provided():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(200, 3)).astype(np.float32)
    timestamps = np.arange(200, dtype=np.int64)

    feed = DataFeed(timestamps=timestamps, features=features, lookback=10)

    # Stats should match fit_stats() output.
    np.testing.assert_allclose(feed.stats.mean, features.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(feed.stats.std, features.std(axis=0), atol=1e-6)
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && uv run pytest tests/trainer/env/test_data_feed.py -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'stats'` and `AttributeError: ... has no attribute 'stats'`.

- [ ] **Step 3: Modify DataFeed**

Replace the `__init__` and add a `stats` attribute:

```python
# backend/src/trainer/env/data_feed.py — partial
from trainer.env.normalization import NormalizationStats, fit_stats


class DataFeed:
    def __init__(
        self,
        timestamps: np.ndarray,
        features: np.ndarray,
        lookback: int = 500,
        price_columns: dict[str, int] | None = None,
        stats: NormalizationStats | None = None,
    ) -> None:
        self.timestamps = timestamps
        self.raw_features = features.astype(np.float32)
        self.lookback = lookback
        self.price_columns = price_columns or {}

        if stats is None:
            stats = fit_stats(self.raw_features)
        self.stats = stats

    def get_observation(self, step: int) -> np.ndarray:
        start = step
        end = step + self.lookback
        window = self.raw_features[start:end]
        return ((window - self.stats.mean) / self.stats.std).astype(np.float32)
```

Remove the now-redundant `_mean`/`_std` attributes.

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/trainer/env/test_data_feed.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/trainer/env/data_feed.py backend/tests/trainer/env/test_data_feed.py
git commit -m "feat(trainer): DataFeed accepts pre-fit NormalizationStats"
```

### Task A.4: Save stats during training; load at eval and live

**Files:**
- Modify: `backend/src/trainer/training/trainer.py`
- Modify: `backend/src/trainer/training/evaluator.py`

- [ ] **Step 1: Read current save flow**

```bash
grep -n "save\|model_path\|\.zip" backend/src/trainer/training/trainer.py
```

Identify the line where the SB3 model is saved (`model.save(...)` or similar). The same path (without `.zip`) is what we pass to `save_stats`.

- [ ] **Step 2: Save stats next to the model**

In `trainer.py`, immediately after the `model.save(<path>)` call, add:

```python
from trainer.env.normalization import save_stats

# ... after model.save(model_path):
save_stats(data_feed.stats, model_path_no_ext)
```

Where `model_path_no_ext` is the `.zip`-less form of the save path. Adjust to match the existing variable.

- [ ] **Step 3: Load stats during eval**

In `evaluator.py`, find where `DataFeed` is constructed for eval. Replace with:

```python
from trainer.env.normalization import load_stats

stats = load_stats(model_path_no_ext)
data_feed = load_data_feed(config, conn, stats=stats)
```

- [ ] **Step 4: Update load_data_feed to forward stats**

In `data_feed.py`, modify `load_data_feed`:

```python
def load_data_feed(
    config: ModelConfig,
    conn: psycopg.Connection,
    stats: NormalizationStats | None = None,
) -> DataFeed:
    # ... existing code building timestamps/features/price_columns ...
    return DataFeed(
        timestamps=timestamps,
        features=features,
        lookback=config.lookback_window,
        price_columns=price_columns,
        stats=stats,
    )
```

- [ ] **Step 5: Backfill stats for existing models**

For Pick 1, Pick 2, Pick 3, the training already happened and no `.mean.npy`/`.std.npy` exists. Write a one-shot script that reconstructs the stats by re-running `load_data_feed` against the same train slice each model used.

Create `backend/scripts/backfill_normalization_stats.py`:

```python
"""Backfill mean.npy/std.npy for already-trained models.

Re-creates the DataFeed used at training time and persists its fit_stats output
next to the model file. Idempotent (skips models that already have stats).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ingester.db import connect
from trainer.config import ModelConfig
from trainer.env.data_feed import load_data_feed
from trainer.env.normalization import save_stats
# adapt the import below to the project's actual model loader
from trainer.training.evaluator import load_model_config_by_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_names", nargs="+")
    parser.add_argument("--models-dir", required=True)
    args = parser.parse_args()

    with connect() as conn:
        for name in args.model_names:
            cfg: ModelConfig = load_model_config_by_name(conn, name)
            base = Path(args.models_dir) / name
            mean_path = base.with_suffix(".mean.npy")
            if mean_path.exists():
                print(f"[skip] {name} (stats already present)")
                continue
            feed = load_data_feed(cfg, conn)
            save_stats(feed.stats, base)
            print(f"[ok] {name}")


if __name__ == "__main__":
    main()
```

If `load_model_config_by_name` does not exist with that name, find the equivalent function in `trainer/training/` that converts a `model_configs` row into `ModelConfig` and adjust the import.

- [ ] **Step 6: Run the backfill on the picks**

```bash
cd backend && uv run python scripts/backfill_normalization_stats.py \
    btc_4h_a2c_lb500_3em4_p2_s1 \
    btc_4h_a2c_lb100_3em4_p2_s0 \
    btc_4h_a2c_lb500_3em4_p2_s0 \
    --models-dir "$MODELS_DIR"
```

Expected: 3 lines `[ok] <name>` and the `.mean.npy`/`.std.npy` files now exist alongside each model `.zip`.

- [ ] **Step 7: Commit**

```bash
git add backend/src/trainer/training/trainer.py \
        backend/src/trainer/training/evaluator.py \
        backend/src/trainer/env/data_feed.py \
        backend/scripts/backfill_normalization_stats.py
git commit -m "feat(trainer): persist normalization stats alongside model"
```

### Task A.5: Define `OrderIntent` dataclass and extract `decode_action`

**Files:**
- Create: `backend/src/trainer/env/action_decoder.py`
- Create: `backend/tests/trainer/env/test_action_decoder.py`

The current `_process_actions` in `TradingEnv` decides what to do AND mutates `self.exchange`. Split: `decode_action` is pure; `apply_intent` (next task) does the mutation.

- [ ] **Step 1: Write the failing test**

`backend/tests/trainer/env/test_action_decoder.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from trainer.config import ExchangeConfig, ModelConfig
from trainer.env.action_decoder import (
    DecoderState,
    OpenIntent,
    OrderIntent,
    decode_action,
)


def _cfg() -> ModelConfig:
    return ModelConfig(
        name="t",
        symbols=["BTCUSDT"],
        intervals=["4h"],
        num_tp_levels=3,
        exchange=ExchangeConfig(max_open_orders=20, max_open_positions=20),
    )


def _state(close: float = 100.0, available: float = 1000.0) -> DecoderState:
    return DecoderState(
        close=close, available_balance=available,
        num_open_orders=0, num_open_positions=0,
    )


def _zero_action(cfg: ModelConfig) -> np.ndarray:
    size = (
        1 + 1 + 1 + 1
        + cfg.num_tp_levels + cfg.num_tp_levels + 1
        + cfg.exchange.max_open_orders + cfg.exchange.max_open_positions
    )
    # All -1 → no opens (open_conf = 0), no cancels, no closes (frac = 0)
    return np.full(size, -1.0, dtype=np.float32)


def test_decode_action_no_open_when_open_confidence_below_threshold():
    cfg = _cfg()
    action = _zero_action(cfg)
    intent = decode_action(action, _state(), cfg)
    assert intent.open is None
    assert intent.cancels == []
    assert intent.closes == []


def test_decode_action_emits_open_with_long_direction():
    cfg = _cfg()
    action = _zero_action(cfg)
    action[0] = 1.0      # open_conf = 1.0 > 0.5 → open
    action[1] = 1.0      # direction = +1 (long)
    action[2] = 0.0      # trigger offset = 0 → trigger == close
    action[3] = 0.5      # SL distance midway between min_sl_pct and max_sl_pct
    # Default TP distances/sizes (=-1 → all 0 → clamped to 0.001 dist)
    action[4 + 2 * cfg.num_tp_levels] = 1.0  # margin = available_balance

    intent = decode_action(action, _state(close=100.0, available=1000.0), cfg)
    assert intent.open is not None
    assert intent.open.direction == 1
    assert intent.open.trigger_price == pytest.approx(100.0)
    # SL midway: min_sl_pct=0.1, max_sl_pct=10.0 → midpoint 5.05 → 5.05% below 100
    assert intent.open.sl_price == pytest.approx(100.0 * (1 - 0.0505), rel=1e-3)
    assert intent.open.margin == pytest.approx(1000.0)


def test_decode_action_picks_indices_to_cancel():
    cfg = _cfg()
    action = _zero_action(cfg)
    cancel_start = 1 + 1 + 1 + 1 + cfg.num_tp_levels + cfg.num_tp_levels + 1
    action[cancel_start + 0] = 0.5     # cancel index 0
    action[cancel_start + 3] = 0.7     # cancel index 3

    intent = decode_action(action, _state(num_open_orders=5), cfg)
    assert intent.cancels == [0, 3]


def test_decode_action_emits_close_intents_above_threshold():
    cfg = _cfg()
    action = _zero_action(cfg)
    close_start = (
        1 + 1 + 1 + 1
        + cfg.num_tp_levels + cfg.num_tp_levels + 1
        + cfg.exchange.max_open_orders
    )
    # close index 1 with frac > 0.05: action ∈ [-1,1] → frac=(a+1)/2; need (a+1)/2 > 0.05 → a > -0.9
    action[close_start + 1] = 0.0   # → frac=0.5

    intent = decode_action(action, _state(num_open_positions=3), cfg)
    assert len(intent.closes) == 1
    assert intent.closes[0].position_index == 1
    assert intent.closes[0].fraction == pytest.approx(0.5)


def test_decode_action_skips_open_below_min_order_size():
    cfg = _cfg()
    action = _zero_action(cfg)
    action[0] = 1.0
    action[1] = 1.0
    action[4 + 2 * cfg.num_tp_levels] = -0.99   # margin ≈ 0.005 * available
    intent = decode_action(action, _state(close=100.0, available=1000.0), cfg)
    # margin = 0.005 * 1000 = 5; min_order_size_usd = 10 → no open
    assert intent.open is None
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && uv run pytest tests/trainer/env/test_action_decoder.py -v
```

Expected: `ModuleNotFoundError: No module named 'trainer.env.action_decoder'`.

- [ ] **Step 3: Implement the module**

`backend/src/trainer/env/action_decoder.py`:

```python
"""Pure action decoder shared between TradingEnv and live runner.

Translates the model's float vector (shape (action_size,), values in [-1, 1])
into a structured OrderIntent. No I/O, no mutation, no exchange access.

Used by:
- trainer.env.trading_env.TradingEnv.step (training/eval)
- live.action_decoder.to_exchange_intent (production)

Both code paths must produce identical intents from identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trainer.config import ModelConfig


@dataclass(frozen=True)
class DecoderState:
    """Snapshot of state needed to decode an action."""
    close: float                # current close price
    available_balance: float    # for margin sizing
    num_open_orders: int        # to truncate cancel signals
    num_open_positions: int     # to truncate close signals


@dataclass(frozen=True)
class OpenIntent:
    direction: int                  # +1 long, -1 short
    trigger_price: float
    sl_price: float
    tp_prices: list[float]
    tp_size_pcts: list[float]       # sums to 1.0
    margin: float                   # in account currency (USD)


@dataclass(frozen=True)
class CloseIntent:
    position_index: int
    fraction: float                 # 0 < fraction <= 1


@dataclass(frozen=True)
class OrderIntent:
    open: OpenIntent | None
    cancels: list[int]              # indices into open_orders
    closes: list[CloseIntent]


def _action_size(cfg: ModelConfig) -> int:
    n_tp = cfg.num_tp_levels
    exc = cfg.exchange
    return 1 + 1 + 1 + 1 + n_tp + n_tp + 1 + exc.max_open_orders + exc.max_open_positions


def decode_action(
    action: np.ndarray,
    state: DecoderState,
    cfg: ModelConfig,
) -> OrderIntent:
    """Translate a 51-float action vector into a structured OrderIntent.

    Layout (matches the original TradingEnv._process_actions):
      [0]                        : open confidence (-1..1; >0 → open)
      [1]                        : direction (+/-)
      [2]                        : trigger price offset
      [3]                        : SL distance
      [4 .. 4+n_tp-1]            : TP distances
      [4+n_tp .. 4+2*n_tp-1]     : TP size weights
      [4+2*n_tp]                 : margin size
      [next max_open_orders]     : cancel signals
      [next max_open_positions]  : close fractions
    """
    expected = _action_size(cfg)
    if action.shape[0] != expected:
        raise ValueError(
            f"action shape {action.shape}, expected ({expected},)"
        )

    n_tp = cfg.num_tp_levels
    exc = cfg.exchange
    cancel_start = 1 + 1 + 1 + 1 + n_tp + n_tp + 1
    cancel_end = cancel_start + exc.max_open_orders
    close_start = cancel_end
    close_end = close_start + exc.max_open_positions

    cancels = [
        i for i in range(state.num_open_orders)
        if i < exc.max_open_orders and action[cancel_start + i] > 0.0
    ]

    closes: list[CloseIntent] = []
    for i in range(min(state.num_open_positions, exc.max_open_positions)):
        frac = float(max(0.0, min(1.0, (action[close_start + i] + 1.0) / 2.0)))
        if frac > 0.05:
            closes.append(CloseIntent(position_index=i, fraction=frac))

    open_conf = (action[0] + 1.0) / 2.0
    open_intent: OpenIntent | None = None
    if open_conf > 0.5:
        direction = 1 if action[1] > 0.0 else -1
        offset_pct = float(action[2]) * cfg.max_trigger_offset_pct / 100.0
        trigger_price = state.close * (1.0 + offset_pct)

        sl_raw = (action[3] + 1.0) / 2.0
        sl_dist_pct = (
            cfg.min_sl_pct + sl_raw * (cfg.max_sl_pct - cfg.min_sl_pct)
        ) / 100.0
        if direction == 1:
            sl_price = trigger_price * (1.0 - sl_dist_pct)
        else:
            sl_price = trigger_price * (1.0 + sl_dist_pct)

        tp_prices: list[float] = []
        raw_tp_sizes: list[float] = []
        for j in range(n_tp):
            tp_raw = (action[4 + j] + 1.0) / 2.0
            tp_dist_pct = max(tp_raw * cfg.max_tp_pct / 100.0, 0.001)
            if direction == 1:
                tp_price = trigger_price * (1.0 + tp_dist_pct)
            else:
                tp_price = trigger_price * (1.0 - tp_dist_pct)
            tp_prices.append(tp_price)
            raw_tp_sizes.append(max((action[4 + n_tp + j] + 1.0) / 2.0, 0.01))

        total = sum(raw_tp_sizes)
        tp_size_pcts = [s / total for s in raw_tp_sizes]

        size_raw = (action[4 + 2 * n_tp] + 1.0) / 2.0
        margin = float(size_raw * state.available_balance)

        if margin >= cfg.exchange.min_order_size_usd:
            open_intent = OpenIntent(
                direction=direction,
                trigger_price=trigger_price,
                sl_price=sl_price,
                tp_prices=tp_prices,
                tp_size_pcts=tp_size_pcts,
                margin=margin,
            )

    return OrderIntent(open=open_intent, cancels=cancels, closes=closes)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/trainer/env/test_action_decoder.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/trainer/env/action_decoder.py \
        backend/tests/trainer/env/test_action_decoder.py
git commit -m "feat(trainer): extract pure decode_action + OrderIntent"
```

### Task A.6: Add `ExchangeSim.apply_intent`

**Files:**
- Modify: `backend/src/trainer/env/exchange_sim.py`
- Create: `backend/tests/trainer/env/test_apply_intent.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/trainer/env/test_apply_intent.py`:

```python
from __future__ import annotations

import pytest

from trainer.config import ExchangeConfig
from trainer.env.account import Account
from trainer.env.action_decoder import CloseIntent, OpenIntent, OrderIntent
from trainer.env.exchange_sim import ExchangeSim


def _sim() -> ExchangeSim:
    return ExchangeSim(config=ExchangeConfig(), account=Account(initial_balance=10_000.0))


def test_apply_intent_opens_order_when_intent_has_open():
    sim = _sim()
    intent = OrderIntent(
        open=OpenIntent(
            direction=1, trigger_price=100.0, sl_price=99.0,
            tp_prices=[101.0, 102.0, 103.0],
            tp_size_pcts=[1/3, 1/3, 1/3], margin=500.0,
        ),
        cancels=[], closes=[],
    )
    info = sim.apply_intent(intent, current_price=100.0)
    assert info["orders_placed"] == 1
    assert len(sim.open_orders) == 1


def test_apply_intent_cancels_in_descending_order():
    sim = _sim()
    # Place 3 orders so we have something to cancel
    for _ in range(3):
        sim.place_order(
            direction=1, trigger_price=100.0, sl_price=99.0,
            tp_prices=[101.0, 102.0, 103.0],
            tp_size_pcts=[1/3, 1/3, 1/3], margin=500.0,
        )
    intent = OrderIntent(open=None, cancels=[0, 2], closes=[])
    info = sim.apply_intent(intent, current_price=100.0)
    assert info["orders_cancelled"] == 2
    assert len(sim.open_orders) == 1


def test_apply_intent_returns_zero_counts_for_empty_intent():
    sim = _sim()
    info = sim.apply_intent(OrderIntent(open=None, cancels=[], closes=[]), current_price=100.0)
    assert info == {"orders_placed": 0, "orders_cancelled": 0, "positions_closed": 0}
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && uv run pytest tests/trainer/env/test_apply_intent.py -v
```

Expected: `AttributeError: 'ExchangeSim' object has no attribute 'apply_intent'`.

- [ ] **Step 3: Implement `apply_intent`**

Append to `backend/src/trainer/env/exchange_sim.py`:

```python
# at top of the file:
from trainer.env.action_decoder import OrderIntent

# inside class ExchangeSim:
def apply_intent(self, intent: "OrderIntent", current_price: float) -> dict:
    """Apply a decoded OrderIntent: cancels first, then closes, then open.

    Mirrors the original ordering in TradingEnv._process_actions so the
    refactor is bit-identical.
    """
    cancelled = 0
    for i in sorted(intent.cancels, reverse=True):
        if 0 <= i < len(self.open_orders):
            self.cancel_order(i)
            cancelled += 1

    closed = 0
    # Close in ascending order (matches original behavior).
    for ci in intent.closes:
        if 0 <= ci.position_index < len(self.open_positions):
            self.close_position(ci.position_index, ci.fraction, current_price)
            closed += 1

    placed = 0
    if intent.open is not None:
        op = intent.open
        order = self.place_order(
            direction=op.direction, trigger_price=op.trigger_price,
            sl_price=op.sl_price, tp_prices=op.tp_prices,
            tp_size_pcts=op.tp_size_pcts, margin=op.margin,
        )
        if order is not None:
            placed = 1

    return {
        "orders_placed": placed,
        "orders_cancelled": cancelled,
        "positions_closed": closed,
    }
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/trainer/env/test_apply_intent.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/trainer/env/exchange_sim.py \
        backend/tests/trainer/env/test_apply_intent.py
git commit -m "feat(trainer): ExchangeSim.apply_intent for shared intent dispatch"
```

### Task A.7: Extract `build_observation` to `trainer/env/observation.py`

**Files:**
- Create: `backend/src/trainer/env/observation.py`
- Create: `backend/tests/trainer/env/test_observation.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/trainer/env/test_observation.py`:

```python
from __future__ import annotations

import numpy as np

from trainer.config import ExchangeConfig
from trainer.env.observation import (
    ObservationConfig,
    ObservationInputs,
    build_observation,
)
from trainer.env.exchange_sim import Order, Position


def _cfg() -> ObservationConfig:
    return ObservationConfig(
        lookback=50,
        num_features=7,
        max_open_orders=20,
        max_open_positions=20,
        max_leverage=125.0,
        initial_balance=10_000.0,
    )


def test_build_observation_shape_and_keys():
    cfg = _cfg()
    market = np.zeros((50, 7), dtype=np.float32)
    inputs = ObservationInputs(
        market=market,
        balance=10_000.0, equity=10_000.0, unrealized_pnl=0.0,
        margin_used=0.0, available_balance=10_000.0,
        open_orders=[], open_positions=[], close=100.0,
    )
    obs = build_observation(inputs, cfg)
    assert set(obs.keys()) == {"market", "account", "orders", "positions"}
    assert obs["market"].shape == (50, 7)
    assert obs["account"].shape == (5,)
    assert obs["orders"].shape == (20, 11)
    assert obs["positions"].shape == (20, 6)
    assert obs["market"].dtype == np.float32
    assert obs["account"].dtype == np.float32


def test_build_observation_account_state_normalized_by_initial_balance():
    cfg = _cfg()
    inputs = ObservationInputs(
        market=np.zeros((50, 7), dtype=np.float32),
        balance=8_000.0, equity=12_000.0, unrealized_pnl=4_000.0,
        margin_used=2_000.0, available_balance=6_000.0,
        open_orders=[], open_positions=[], close=100.0,
    )
    obs = build_observation(inputs, cfg)
    np.testing.assert_allclose(
        obs["account"],
        np.array([0.8, 1.2, 0.4, 0.2, 0.6], dtype=np.float32),
    )


def test_build_observation_encodes_open_order_row():
    cfg = _cfg()
    order = Order(
        id=0, direction=1, trigger_price=100.0, sl_price=98.0,
        tp_prices=[105.0, 110.0, 115.0],
        tp_size_pcts=[0.5, 0.3, 0.2], margin=500.0,
    )
    inputs = ObservationInputs(
        market=np.zeros((50, 7), dtype=np.float32),
        balance=10_000.0, equity=10_000.0, unrealized_pnl=0.0,
        margin_used=500.0, available_balance=9_500.0,
        open_orders=[order], open_positions=[], close=100.0,
    )
    obs = build_observation(inputs, cfg)
    row = obs["orders"][0]
    assert row[0] == 1.0
    assert row[1] == 1.0                        # direction
    assert row[2] == 1.0                        # trigger / close = 1.0
    assert row[3] == 0.98                       # sl / close
    np.testing.assert_allclose(row[4:7], [1.05, 1.10, 1.15], atol=1e-6)
    np.testing.assert_allclose(row[7:10], [0.5, 0.3, 0.2], atol=1e-6)
    assert row[10] == 0.05                      # margin / initial_balance
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && uv run pytest tests/trainer/env/test_observation.py -v
```

Expected: `ModuleNotFoundError: No module named 'trainer.env.observation'`.

- [ ] **Step 3: Implement the module**

`backend/src/trainer/env/observation.py`:

```python
"""Pure observation builder shared between TradingEnv and live runner.

Returns the same Dict observation that TradingEnv.observation_space describes.
No I/O, no DB, no exchange access.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trainer.env.exchange_sim import Order, Position


@dataclass(frozen=True)
class ObservationConfig:
    lookback: int
    num_features: int
    max_open_orders: int
    max_open_positions: int
    max_leverage: float
    initial_balance: float


@dataclass
class ObservationInputs:
    """Snapshot of state at observation time. Caller is responsible for
    normalizing `market` (i.e. applying mean/std)."""
    market: np.ndarray              # shape (lookback, num_features), already normalized
    balance: float
    equity: float
    unrealized_pnl: float
    margin_used: float
    available_balance: float
    open_orders: list[Order]
    open_positions: list[Position]
    close: float                    # current close (for ratio normalization)


def build_observation(
    inputs: ObservationInputs,
    cfg: ObservationConfig,
) -> dict[str, np.ndarray]:
    init = cfg.initial_balance
    close = inputs.close if inputs.close > 0 else 1.0

    account_state = np.array([
        inputs.balance / init,
        inputs.equity / init,
        inputs.unrealized_pnl / init,
        inputs.margin_used / init,
        inputs.available_balance / init,
    ], dtype=np.float32)

    orders = np.zeros((cfg.max_open_orders, 11), dtype=np.float32)
    for i, order in enumerate(inputs.open_orders[:cfg.max_open_orders]):
        orders[i, 0] = 1.0
        orders[i, 1] = float(order.direction)
        orders[i, 2] = order.trigger_price / close
        orders[i, 3] = order.sl_price / close
        for j, tp in enumerate(order.tp_prices[:3]):
            orders[i, 4 + j] = tp / close
        for j, pct in enumerate(order.tp_size_pcts[:3]):
            orders[i, 7 + j] = pct
        orders[i, 10] = order.margin / init

    positions = np.zeros((cfg.max_open_positions, 6), dtype=np.float32)
    for i, pos in enumerate(inputs.open_positions[:cfg.max_open_positions]):
        positions[i, 0] = 1.0
        positions[i, 1] = float(pos.direction)
        positions[i, 2] = pos.entry_price / close
        positions[i, 3] = pos.size * pos.entry_price / init
        positions[i, 4] = pos.unrealized_pnl(close) / init
        positions[i, 5] = pos.leverage / cfg.max_leverage

    return {
        "market": inputs.market.astype(np.float32),
        "account": account_state,
        "orders": orders,
        "positions": positions,
    }
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/trainer/env/test_observation.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/trainer/env/observation.py \
        backend/tests/trainer/env/test_observation.py
git commit -m "feat(trainer): extract pure build_observation"
```

### Task A.8: Refactor `TradingEnv` to delegate

**Files:**
- Modify: `backend/src/trainer/env/trading_env.py`

- [ ] **Step 1: Replace `_process_actions` with decode + apply_intent**

In `trading_env.py`, replace the entire `_process_actions` method body:

```python
def _process_actions(self, action: np.ndarray, close: float, info: dict) -> None:
    from trainer.env.action_decoder import DecoderState, decode_action
    state = DecoderState(
        close=close,
        available_balance=self.account.available_balance,
        num_open_orders=len(self.exchange.open_orders),
        num_open_positions=len(self.exchange.open_positions),
    )
    intent = decode_action(action, state, self.config)
    info.update(self.exchange.apply_intent(intent, current_price=close))
```

- [ ] **Step 2: Replace `_build_observation` with delegate**

```python
def _build_observation(self) -> dict[str, np.ndarray]:
    from trainer.env.observation import (
        ObservationConfig, ObservationInputs, build_observation,
    )
    step = min(self._current_step, self.data_feed.total_steps - 1)
    market = self.data_feed.get_observation(step)
    raw = self.data_feed.get_current_raw(step)
    close = float(raw[self.data_feed.price_columns.get("close", 3)])
    if close <= 0:
        close = 1.0

    unrealized = self.exchange.total_unrealized_pnl(close)
    inputs = ObservationInputs(
        market=market,
        balance=self.account.balance,
        equity=self.account.equity(unrealized),
        unrealized_pnl=unrealized,
        margin_used=self.account.margin_used,
        available_balance=self.account.available_balance,
        open_orders=self.exchange.open_orders,
        open_positions=self.exchange.open_positions,
        close=close,
    )
    cfg = ObservationConfig(
        lookback=self.config.lookback_window,
        num_features=self.data_feed.num_features,
        max_open_orders=self.config.exchange.max_open_orders,
        max_open_positions=self.config.exchange.max_open_positions,
        max_leverage=self.config.exchange.max_leverage,
        initial_balance=self.config.initial_balance,
    )
    return build_observation(inputs, cfg)
```

- [ ] **Step 3: Run all existing trainer-related tests to detect regressions**

```bash
cd backend && uv run pytest tests/trainer -v
```

Expected: all green. If anything fails, investigate before proceeding — the refactor must be behaviorally identical.

- [ ] **Step 4: Commit**

```bash
git add backend/src/trainer/env/trading_env.py
git commit -m "refactor(trainer): TradingEnv delegates to extracted modules"
```

### Task A.9: Gate A1 — before/after eval bit-identical regression test

**Files:**
- Create: `backend/tests/trainer/env/test_refactor_parity.py`

The plan above made the refactor TDD-clean; this gate certifies that running a saved model through the refactored env reproduces stored eval results byte-for-byte. This stays in the repo as a regression test.

- [ ] **Step 1: Pick frozen model + window**

Use Pick 2 (`btc_4h_a2c_lb100_3em4_p2_s0`) — small lookback, low DD, fast to eval. Note its expected terminal equity from `training_runs` table:

```bash
psql "$DATABASE_URL" -c "
SELECT mc.name, tr.total_pnl, tr.final_balance, tr.total_trades, tr.sharpe_ratio
FROM training_runs tr
JOIN model_configs mc ON mc.id = tr.model_config_id
WHERE mc.name = 'btc_4h_a2c_lb100_3em4_p2_s0' AND tr.run_type = 'evaluate';
"
```

Record the `final_balance` value. Capture as `EXPECTED_FINAL_BALANCE` in the test.

- [ ] **Step 2: Write the regression test**

`backend/tests/trainer/env/test_refactor_parity.py`:

```python
"""Regression: a saved model evaluates to the same terminal equity it did
when its row was first inserted into training_runs.

Skips if the model file or DB connection isn't available.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


PICK = "btc_4h_a2c_lb100_3em4_p2_s0"
# Fill in from the SELECT above. Tolerance 1e-6 — the refactor should be
# numerically identical, not approximately equal.
EXPECTED_FINAL_BALANCE: float = float(os.environ.get("PICK2_EXPECTED_FINAL_BALANCE", "0.0"))


@pytest.mark.skipif(
    EXPECTED_FINAL_BALANCE == 0.0
    or "DATABASE_URL" not in os.environ
    or "MODELS_DIR" not in os.environ,
    reason="Requires DATABASE_URL, MODELS_DIR, and PICK2_EXPECTED_FINAL_BALANCE",
)
def test_refactor_eval_terminal_equity_identical():
    from ingester.db import connect
    from trainer.env.data_feed import load_data_feed
    from trainer.env.normalization import load_stats
    from trainer.env.trading_env import TradingEnv
    from trainer.training.evaluator import (
        load_model_config_by_name,
        load_sb3_model,
        run_eval,
    )

    with connect() as conn:
        cfg = load_model_config_by_name(conn, PICK)
        models_dir = Path(os.environ["MODELS_DIR"])
        stats = load_stats(models_dir / PICK)
        feed = load_data_feed(cfg, conn, stats=stats)
        env = TradingEnv(cfg, feed)
        model = load_sb3_model(models_dir / PICK)
        result = run_eval(model, env)

    assert result.final_balance == pytest.approx(EXPECTED_FINAL_BALANCE, abs=1e-6)
```

The exact import names (`load_model_config_by_name`, `load_sb3_model`, `run_eval`) may differ from the trainer's conventions. Use the actual function names from `evaluator.py`.

- [ ] **Step 3: Run the parity test**

```bash
cd backend && PICK2_EXPECTED_FINAL_BALANCE=<value-from-step-1> \
  uv run pytest tests/trainer/env/test_refactor_parity.py -v
```

Expected: PASS, with terminal equity matching to within 1e-6.

If it fails: do NOT proceed. The refactor changed behavior somewhere. Re-read the diff vs the original `_process_actions` and `_build_observation`, find the divergence, fix, re-run. This gate is non-negotiable.

- [ ] **Step 4: Run the same parity test against Picks 1 and 3**

Repeat Step 1 to get expected final balances for Pick 1 and Pick 3. Add two more parametrized test cases (or run the test three times with env var swaps). All three must pass with abs tolerance 1e-6.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/trainer/env/test_refactor_parity.py
git commit -m "test(trainer): regression gate for refactor parity"
```

- [ ] **Step 6: Delete temporary research notes**

```bash
git rm docs/superpowers/notes/normalization-parity.md
git commit -m "chore: remove transitional research notes"
```

**Phase A complete.** `live/` work begins next, building on top of stable extracted modules.

---

## Phase B — Live infrastructure scaffolding (no exchange yet)

### Task B.1: Add deps and register console scripts

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add dependencies**

In `backend/pyproject.toml`, add to `dependencies = [...]`:

```toml
"pyyaml>=6.0",
"pydantic>=2.7",
```

`ccxt>=4.5.46` is already present. Pydantic was not — add it.

- [ ] **Step 2: Register console scripts**

In `[project.scripts]`, add:

```toml
live-test = "live.cli:main"
live-replay = "live.cli:replay_main"
```

- [ ] **Step 3: Sync**

```bash
cd backend && uv sync
```

Expected: clean install, lockfile updated.

- [ ] **Step 4: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "build(live): add pyyaml, pydantic; register live-test and live-replay"
```

### Task B.2: Pydantic config schemas

**Files:**
- Create: `backend/src/live/__init__.py`
- Create: `backend/src/live/config.py`
- Create: `backend/tests/live/test_config.py`

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p backend/src/live/exchange
touch backend/src/live/__init__.py backend/src/live/exchange/__init__.py
```

- [ ] **Step 2: Write the failing test**

`backend/tests/live/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from live.config import LiveConfig


VALID = """
exchange:
  name: bingx
  mode: demo
  api_key_env: BINGX_VST_S1_API_KEY
  api_secret_env: BINGX_VST_S1_API_SECRET
market:
  symbol: BTC/USDT:USDT
  interval: 4h
model:
  name: btc_4h_a2c_lb500_3em4_p2_s1
risk:
  starting_equity_quote: 10000
  max_drawdown_pct: 0.20
  max_position_size_pct: 0.50
  max_leverage: 3
  kill_switch_env: TRADAN_KILL_SWITCH_S1
logging:
  pnl_snapshot_interval_minutes: 60
"""


def test_valid_config_loads(tmp_path: Path):
    p = tmp_path / "live-s1.yaml"
    p.write_text(VALID)
    cfg = LiveConfig.from_yaml(p)
    assert cfg.exchange.name == "bingx"
    assert cfg.exchange.mode == "demo"
    assert cfg.market.symbol == "BTC/USDT:USDT"
    assert cfg.risk.max_drawdown_pct == 0.20


def test_invalid_mode_rejected(tmp_path: Path):
    bad = VALID.replace("mode: demo", "mode: production")
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        LiveConfig.from_yaml(p)


def test_invalid_drawdown_rejected(tmp_path: Path):
    bad = VALID.replace("max_drawdown_pct: 0.20", "max_drawdown_pct: 1.5")
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        LiveConfig.from_yaml(p)
```

- [ ] **Step 3: Run test, verify failure**

```bash
cd backend && uv run pytest tests/live/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'live.config'`.

- [ ] **Step 4: Implement `live/config.py`**

```python
"""Pydantic schemas for live runner YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ExchangeCfg(BaseModel):
    name: Literal["bingx"]
    mode: Literal["demo", "paper", "live"]
    api_key_env: str
    api_secret_env: str


class MarketCfg(BaseModel):
    symbol: str
    interval: str

    @field_validator("interval")
    @classmethod
    def _interval_known(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
        if v not in allowed:
            raise ValueError(f"interval {v!r} not in {sorted(allowed)}")
        return v


class ModelCfg(BaseModel):
    name: str


class RiskCfg(BaseModel):
    starting_equity_quote: float = Field(gt=0)
    max_drawdown_pct: float = Field(ge=0.0, le=1.0)
    max_position_size_pct: float = Field(gt=0.0, le=1.0)
    max_leverage: float = Field(gt=0)
    kill_switch_env: str


class LoggingCfg(BaseModel):
    pnl_snapshot_interval_minutes: int = Field(ge=1)


class LiveConfig(BaseModel):
    exchange: ExchangeCfg
    market: MarketCfg
    model: ModelCfg
    risk: RiskCfg
    logging: LoggingCfg

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LiveConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
```

- [ ] **Step 5: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/live/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/src/live/__init__.py \
        backend/src/live/exchange/__init__.py \
        backend/src/live/config.py \
        backend/tests/live/test_config.py
git commit -m "feat(live): Pydantic schemas for live runner config"
```

### Task B.3: Migration `006_live_testing_tables.sql`

**Files:**
- Create: `backend/migrations/006_live_testing_tables.sql`

- [ ] **Step 1: Write the migration**

```sql
-- backend/migrations/006_live_testing_tables.sql

CREATE TABLE live_runs (
    id              SERIAL PRIMARY KEY,
    model_config_id INT  NOT NULL REFERENCES model_configs(id),
    exchange        TEXT NOT NULL,
    mode            TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    interval        TEXT NOT NULL,
    starting_equity NUMERIC NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    stopped_at      TIMESTAMPTZ,
    stop_reason     TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    kill_requested  BOOLEAN NOT NULL DEFAULT FALSE,
    config_yaml     TEXT NOT NULL,
    git_sha         TEXT NOT NULL
);

CREATE UNIQUE INDEX live_runs_one_running_per_model
    ON live_runs(model_config_id, exchange) WHERE status = 'running';

CREATE TABLE live_actions (
    id              SERIAL PRIMARY KEY,
    live_run_id     INT  NOT NULL REFERENCES live_runs(id),
    event_type      TEXT NOT NULL DEFAULT 'inference',
    candle_close    TIMESTAMPTZ,
    raw_action      JSONB,
    decoded_intent  JSONB,
    account_state   JSONB NOT NULL,
    inference_ms    INT,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE live_orders (
    id                SERIAL PRIMARY KEY,
    live_run_id       INT  NOT NULL REFERENCES live_runs(id),
    live_action_id    INT  REFERENCES live_actions(id),
    exchange_order_id TEXT NOT NULL,
    side              TEXT NOT NULL,
    type              TEXT NOT NULL,
    price             NUMERIC,
    amount            NUMERIC NOT NULL,
    status            TEXT NOT NULL,
    fill_price        NUMERIC,
    fill_amount       NUMERIC,
    pnl               NUMERIC,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE live_pnl_snapshots (
    id              SERIAL PRIMARY KEY,
    live_run_id     INT NOT NULL REFERENCES live_runs(id),
    taken_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    equity          NUMERIC NOT NULL,
    realized_pnl    NUMERIC NOT NULL,
    unrealized_pnl  NUMERIC NOT NULL,
    open_positions  INT NOT NULL,
    open_orders     INT NOT NULL
);

CREATE INDEX live_actions_run_time ON live_actions(live_run_id, candle_close DESC);
CREATE INDEX live_orders_run_status ON live_orders(live_run_id, status);
CREATE INDEX live_pnl_run_time     ON live_pnl_snapshots(live_run_id, taken_at DESC);
```

- [ ] **Step 2: Apply on dev DB**

```bash
cd backend && uv run ingest migrate
```

Expected: `[migrate] Applied: 006_live_testing_tables.sql`.

- [ ] **Step 3: Verify tables**

```bash
psql "$DATABASE_URL" -c "\dt live_*"
```

Expected: four rows — `live_actions`, `live_orders`, `live_pnl_snapshots`, `live_runs`.

```bash
psql "$DATABASE_URL" -c "\di live_runs_one_running_per_model"
```

Expected: the partial unique index exists.

- [ ] **Step 4: Commit**

```bash
git add backend/migrations/006_live_testing_tables.sql
git commit -m "feat(db): add live_runs, live_actions, live_orders, live_pnl_snapshots"
```

### Task B.4: `ExchangeAdapter` ABC + DTOs

**Files:**
- Create: `backend/src/live/exchange/base.py`

- [ ] **Step 1: Implement DTOs and ABC**

`backend/src/live/exchange/base.py`:

```python
"""Abstract ExchangeAdapter and exchange-agnostic DTOs.

Concrete adapters (BingX, Replay, future Binance/Bybit) implement this
interface. The runner does not import any concrete adapter directly — it
goes through live.exchange.registry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Kline:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Balance:
    total: float            # equity-equivalent (USDT)
    available: float        # free margin
    used: float             # margin in use


@dataclass(frozen=True)
class Position:
    id: str                 # exchange position id (or symbol-side composite)
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    size: float             # base units
    leverage: float
    unrealized_pnl: float
    margin: float
    liquidation_price: float | None


@dataclass(frozen=True)
class Order:
    id: str
    symbol: str
    side: Literal["buy", "sell"]
    type: Literal["limit", "market", "stop", "take_profit"]
    price: float | None
    amount: float
    status: Literal["open", "filled", "cancelled", "rejected"]
    fill_price: float | None = None
    fill_amount: float | None = None


@dataclass(frozen=True)
class OrderRequest:
    side: Literal["buy", "sell"]
    type: Literal["limit", "market", "stop", "take_profit"]
    amount: float           # base units
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


class ExchangeAdapter(ABC):
    @abstractmethod
    def fetch_klines(
        self, symbol: str, interval: str, limit: int
    ) -> list[Kline]: ...

    @abstractmethod
    def fetch_balance(self) -> Balance: ...

    @abstractmethod
    def fetch_positions(self, symbol: str) -> list[Position]: ...

    @abstractmethod
    def fetch_open_orders(self, symbol: str) -> list[Order]: ...

    @abstractmethod
    def place_order(self, symbol: str, request: OrderRequest) -> Order: ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> None: ...

    @abstractmethod
    def close_position(self, symbol: str, position_id: str, fraction: float) -> Order: ...

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: float) -> None: ...
```

- [ ] **Step 2: Commit**

```bash
git add backend/src/live/exchange/base.py
git commit -m "feat(live): ExchangeAdapter ABC and DTOs"
```

### Task B.5: Adapter registry

**Files:**
- Create: `backend/src/live/exchange/registry.py`
- Create: `backend/tests/live/test_registry.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/live/test_registry.py`:

```python
from __future__ import annotations

import pytest

from live.exchange.base import ExchangeAdapter
from live.exchange.registry import get_adapter_class


def test_known_adapter_returned():
    cls = get_adapter_class("bingx")
    assert issubclass(cls, ExchangeAdapter)


def test_replay_adapter_returned():
    cls = get_adapter_class("replay")
    assert issubclass(cls, ExchangeAdapter)


def test_unknown_adapter_raises():
    with pytest.raises(KeyError):
        get_adapter_class("nope")
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && uv run pytest tests/live/test_registry.py -v
```

Expected: `ModuleNotFoundError: No module named 'live.exchange.registry'`.

- [ ] **Step 3: Implement registry**

`backend/src/live/exchange/registry.py`:

```python
"""Lazy registry mapping exchange name → adapter class."""
from __future__ import annotations

import importlib

from live.exchange.base import ExchangeAdapter


_REGISTRY: dict[str, str] = {
    "bingx": "live.exchange.bingx.BingXAdapter",
    "replay": "live.exchange.replay.ReplayAdapter",
}


def get_adapter_class(name: str) -> type[ExchangeAdapter]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown exchange adapter: {name!r}")
    dotted = _REGISTRY[name]
    module_path, _, cls_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)
```

The bingx and replay modules don't exist yet — Task B.5's test will fail until B.4-B.6 are also done. Either skip the bingx/replay subtests for now, or use placeholder stubs. Cleanest path: defer running this test until Task C.1 (replay) and Task D.1 (bingx) land.

- [ ] **Step 4: Add stubs so import works**

`backend/src/live/exchange/replay.py`:

```python
"""Stub — implementation in Phase C."""
from __future__ import annotations

from live.exchange.base import ExchangeAdapter


class ReplayAdapter(ExchangeAdapter):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("ReplayAdapter is implemented in Phase C")

    # Abstract methods stubbed to allow import; concrete impls in Phase C.
    def fetch_klines(self, symbol, interval, limit): raise NotImplementedError
    def fetch_balance(self): raise NotImplementedError
    def fetch_positions(self, symbol): raise NotImplementedError
    def fetch_open_orders(self, symbol): raise NotImplementedError
    def place_order(self, symbol, request): raise NotImplementedError
    def cancel_order(self, symbol, order_id): raise NotImplementedError
    def close_position(self, symbol, position_id, fraction): raise NotImplementedError
    def set_leverage(self, symbol, leverage): raise NotImplementedError
```

`backend/src/live/exchange/bingx.py`: identical structure with `class BingXAdapter(ExchangeAdapter)`.

- [ ] **Step 5: Run registry tests, verify pass**

```bash
cd backend && uv run pytest tests/live/test_registry.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/src/live/exchange/registry.py \
        backend/src/live/exchange/replay.py \
        backend/src/live/exchange/bingx.py \
        backend/tests/live/test_registry.py
git commit -m "feat(live): exchange adapter registry with stubs"
```

### Task B.6: DB write helpers

**Files:**
- Create: `backend/src/live/db.py`
- Create: `backend/tests/live/test_db.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/live/test_db.py`:

```python
"""Integration tests for live DB writes. Each test runs in a transaction
that is rolled back at teardown so the DB stays clean.

Requires DATABASE_URL pointing at a DB with migrations 001-006 applied.
A `model_configs` row will be created and torn down with the transaction.
"""
from __future__ import annotations

import json
import os
import pytest

from ingester.db import connect


pytestmark = pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="needs DATABASE_URL",
)


@pytest.fixture
def conn_tx():
    conn = connect()
    conn.autocommit = False
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.fixture
def model_config_id(conn_tx):
    row = conn_tx.execute(
        """
        INSERT INTO model_configs (name, config)
        VALUES ('test_live', '{}'::jsonb)
        RETURNING id
        """
    ).fetchone()
    return row[0]


def test_start_run_inserts_row(conn_tx, model_config_id):
    from live.db import start_run

    run_id = start_run(
        conn_tx,
        model_config_id=model_config_id,
        exchange="bingx", mode="demo",
        symbol="BTC/USDT:USDT", interval="4h",
        starting_equity=10_000.0,
        config_yaml="exchange: {name: bingx}\n",
        git_sha="deadbeef",
    )
    row = conn_tx.execute(
        "SELECT status, mode, starting_equity FROM live_runs WHERE id = %s",
        (run_id,),
    ).fetchone()
    assert row[0] == "running"
    assert row[1] == "demo"
    assert float(row[2]) == 10_000.0


def test_start_run_blocks_second_running_for_same_model(conn_tx, model_config_id):
    from live.db import start_run

    start_run(
        conn_tx, model_config_id=model_config_id,
        exchange="bingx", mode="demo", symbol="BTC/USDT:USDT", interval="4h",
        starting_equity=10_000.0, config_yaml="x", git_sha="aaa",
    )
    with pytest.raises(Exception):
        start_run(
            conn_tx, model_config_id=model_config_id,
            exchange="bingx", mode="demo", symbol="BTC/USDT:USDT", interval="4h",
            starting_equity=10_000.0, config_yaml="y", git_sha="bbb",
        )


def test_log_action_writes_json_payload(conn_tx, model_config_id):
    from live.db import start_run, log_action

    run_id = start_run(
        conn_tx, model_config_id=model_config_id,
        exchange="bingx", mode="demo", symbol="BTC/USDT:USDT", interval="4h",
        starting_equity=10_000.0, config_yaml="x", git_sha="aaa",
    )
    action_id = log_action(
        conn_tx, live_run_id=run_id, event_type="inference",
        candle_close="2026-05-09T00:00:00Z",
        raw_action=[0.1, -0.2, 0.3],
        decoded_intent={"open": None, "cancels": [], "closes": []},
        account_state={"equity": 10_000.0},
        inference_ms=12,
    )
    row = conn_tx.execute(
        "SELECT raw_action, decoded_intent FROM live_actions WHERE id = %s",
        (action_id,),
    ).fetchone()
    assert row[0] == [0.1, -0.2, 0.3]
    assert row[1] == {"open": None, "cancels": [], "closes": []}


def test_request_stop_sets_kill_flag(conn_tx, model_config_id):
    from live.db import start_run, request_stop

    run_id = start_run(
        conn_tx, model_config_id=model_config_id,
        exchange="bingx", mode="demo", symbol="BTC/USDT:USDT", interval="4h",
        starting_equity=10_000.0, config_yaml="x", git_sha="aaa",
    )
    request_stop(conn_tx, run_id)
    row = conn_tx.execute(
        "SELECT kill_requested FROM live_runs WHERE id = %s", (run_id,),
    ).fetchone()
    assert row[0] is True
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && uv run pytest tests/live/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'live.db'`.

- [ ] **Step 3: Implement `live/db.py`**

```python
"""DB writes for live_runs, live_actions, live_orders, live_pnl_snapshots.

Functions take a psycopg connection so callers can decide transaction
boundaries (e.g., LiveRunner uses one connection for the whole run).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Json


def start_run(
    conn: psycopg.Connection,
    *,
    model_config_id: int,
    exchange: str,
    mode: str,
    symbol: str,
    interval: str,
    starting_equity: float,
    config_yaml: str,
    git_sha: str,
) -> int:
    row = conn.execute(
        """
        INSERT INTO live_runs (
            model_config_id, exchange, mode, symbol, interval,
            starting_equity, config_yaml, git_sha
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (model_config_id, exchange, mode, symbol, interval,
         starting_equity, config_yaml, git_sha),
    ).fetchone()
    return row[0]


def find_running_run(
    conn: psycopg.Connection,
    *,
    model_config_id: int,
    exchange: str,
) -> int | None:
    row = conn.execute(
        """
        SELECT id FROM live_runs
        WHERE model_config_id = %s AND exchange = %s AND status = 'running'
        """,
        (model_config_id, exchange),
    ).fetchone()
    return row[0] if row else None


def stop_run(
    conn: psycopg.Connection,
    run_id: int,
    *,
    reason: str,
) -> None:
    conn.execute(
        """
        UPDATE live_runs
        SET status = 'stopped', stopped_at = now(), stop_reason = %s
        WHERE id = %s
        """,
        (reason, run_id),
    )


def request_stop(conn: psycopg.Connection, run_id: int) -> None:
    conn.execute(
        "UPDATE live_runs SET kill_requested = TRUE WHERE id = %s",
        (run_id,),
    )


def is_kill_requested(conn: psycopg.Connection, run_id: int) -> bool:
    row = conn.execute(
        "SELECT kill_requested FROM live_runs WHERE id = %s", (run_id,),
    ).fetchone()
    return bool(row and row[0])


def log_action(
    conn: psycopg.Connection,
    *,
    live_run_id: int,
    event_type: str,
    candle_close: datetime | str | None = None,
    raw_action: list[float] | None = None,
    decoded_intent: dict | None = None,
    account_state: dict,
    inference_ms: int | None = None,
    notes: str | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO live_actions (
            live_run_id, event_type, candle_close, raw_action,
            decoded_intent, account_state, inference_ms, notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            live_run_id, event_type, candle_close,
            Json(raw_action) if raw_action is not None else None,
            Json(decoded_intent) if decoded_intent is not None else None,
            Json(account_state),
            inference_ms, notes,
        ),
    ).fetchone()
    return row[0]


def log_order(
    conn: psycopg.Connection,
    *,
    live_run_id: int,
    live_action_id: int | None,
    exchange_order_id: str,
    side: str,
    type: str,
    price: float | None,
    amount: float,
    status: str,
    fill_price: float | None = None,
    fill_amount: float | None = None,
    pnl: float | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO live_orders (
            live_run_id, live_action_id, exchange_order_id,
            side, type, price, amount, status,
            fill_price, fill_amount, pnl
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (live_run_id, live_action_id, exchange_order_id, side, type,
         price, amount, status, fill_price, fill_amount, pnl),
    ).fetchone()
    return row[0]


def log_pnl_snapshot(
    conn: psycopg.Connection,
    *,
    live_run_id: int,
    equity: float,
    realized_pnl: float,
    unrealized_pnl: float,
    open_positions: int,
    open_orders: int,
) -> int:
    row = conn.execute(
        """
        INSERT INTO live_pnl_snapshots (
            live_run_id, equity, realized_pnl, unrealized_pnl,
            open_positions, open_orders
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (live_run_id, equity, realized_pnl, unrealized_pnl,
         open_positions, open_orders),
    ).fetchone()
    return row[0]
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/live/test_db.py -v
```

Expected: 4 passed (or skipped if `DATABASE_URL` not set — set it for dev DB).

- [ ] **Step 5: Commit**

```bash
git add backend/src/live/db.py backend/tests/live/test_db.py
git commit -m "feat(live): DB write helpers for live_* tables"
```

**Phase B complete.** Adapter ABC, registry stubs, config schemas, and DB layer are in place.

---

## Phase C — Replay adapter and gate

The most important phase for correctness. The replay gate proves that the live code path produces the same equity curve as the trainer's eval, *before* any live exchange touches.

### Task C.1: ReplayAdapter

**Files:**
- Modify: `backend/src/live/exchange/replay.py`
- Create: `backend/tests/live/test_replay_adapter.py`

The ReplayAdapter wraps `ExchangeSim` so the same fill simulator the trainer used drives the live code path.

- [ ] **Step 1: Write the failing test**

`backend/tests/live/test_replay_adapter.py`:

```python
from __future__ import annotations

import numpy as np

from live.exchange.base import OrderRequest
from live.exchange.replay import ReplayAdapter


def test_fetch_klines_returns_recent_window():
    timestamps = np.arange(0, 200, dtype=np.int64) * 60_000
    features = np.column_stack([
        np.linspace(100, 120, 200),     # open
        np.linspace(101, 121, 200),     # high
        np.linspace(99, 119, 200),      # low
        np.linspace(100, 120, 200),     # close
        np.full(200, 1000.0),           # volume
    ]).astype(np.float32)
    price_columns = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}

    adapter = ReplayAdapter.from_arrays(
        timestamps=timestamps,
        features=features,
        price_columns=price_columns,
        symbol="BTC/USDT:USDT",
        interval="4h",
        starting_balance=10_000.0,
    )
    klines = adapter.fetch_klines("BTC/USDT:USDT", "4h", limit=50)
    assert len(klines) == 50
    assert klines[-1].open_time_ms == int(timestamps[-1])
    assert klines[0].close < klines[-1].close


def test_advance_step_processes_candle():
    timestamps = np.arange(0, 50, dtype=np.int64)
    features = np.tile(
        np.array([100.0, 101.0, 99.0, 100.0, 1.0]), (50, 1),
    ).astype(np.float32)
    price_columns = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}

    adapter = ReplayAdapter.from_arrays(
        timestamps=timestamps, features=features, price_columns=price_columns,
        symbol="BTC/USDT:USDT", interval="4h", starting_balance=10_000.0,
    )
    initial_balance = adapter.fetch_balance()
    assert initial_balance.available == 10_000.0
    adapter.advance()
    after = adapter.fetch_balance()
    # Nothing happened (no orders); balance should still be 10_000.
    assert after.available == 10_000.0
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && uv run pytest tests/live/test_replay_adapter.py -v
```

Expected: `NotImplementedError: ReplayAdapter is implemented in Phase C`.

- [ ] **Step 3: Implement ReplayAdapter**

Replace `backend/src/live/exchange/replay.py`:

```python
"""Replay adapter — drives the live code path with historical klines.

Wraps ExchangeSim so fills, fees, leverage, and liquidations match the
trainer's eval exactly. The cursor advances one candle per `advance()` call.

Used only by scripts/live_replay.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from live.exchange.base import (
    Balance,
    ExchangeAdapter,
    Kline,
    Order,
    OrderRequest,
    Position,
)
from trainer.config import ExchangeConfig
from trainer.env.account import Account
from trainer.env.exchange_sim import ExchangeSim


@dataclass
class _ReplayState:
    timestamps: np.ndarray
    features: np.ndarray
    price_columns: dict[str, int]
    interval: str
    symbol: str
    cursor: int                # index into features (current candle)
    sim: ExchangeSim


class ReplayAdapter(ExchangeAdapter):
    """Constructed via from_arrays(). Do not instantiate directly."""

    def __init__(self, state: _ReplayState):
        self._state = state

    @classmethod
    def from_arrays(
        cls,
        *,
        timestamps: np.ndarray,
        features: np.ndarray,
        price_columns: dict[str, int],
        symbol: str,
        interval: str,
        starting_balance: float,
        exchange_config: ExchangeConfig | None = None,
    ) -> "ReplayAdapter":
        cfg = exchange_config or ExchangeConfig()
        sim = ExchangeSim(config=cfg, account=Account(initial_balance=starting_balance))
        state = _ReplayState(
            timestamps=timestamps, features=features,
            price_columns=price_columns, interval=interval, symbol=symbol,
            cursor=0, sim=sim,
        )
        return cls(state)

    @property
    def cursor(self) -> int:
        return self._state.cursor

    @property
    def sim(self) -> ExchangeSim:
        return self._state.sim

    def advance(self) -> None:
        st = self._state
        idx = st.cursor
        if idx >= len(st.features):
            return
        row = st.features[idx]
        high = float(row[st.price_columns["high"]])
        low = float(row[st.price_columns["low"]])
        close = float(row[st.price_columns["close"]])
        st.sim.process_candle(high=high, low=low, close=close)
        st.cursor += 1

    # -- ExchangeAdapter interface ------------------------------------------

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[Kline]:
        st = self._state
        end = max(0, st.cursor)
        start = max(0, end - limit)
        out: list[Kline] = []
        for i in range(start, end):
            row = st.features[i]
            out.append(Kline(
                open_time_ms=int(st.timestamps[i]),
                open=float(row[st.price_columns["open"]]),
                high=float(row[st.price_columns["high"]]),
                low=float(row[st.price_columns["low"]]),
                close=float(row[st.price_columns["close"]]),
                volume=float(row[st.price_columns["volume"]]),
            ))
        return out

    def fetch_balance(self) -> Balance:
        sim = self._state.sim
        unrealized = sim.total_unrealized_pnl(self._current_close())
        return Balance(
            total=sim.account.equity(unrealized),
            available=sim.account.available_balance,
            used=sim.account.margin_used,
        )

    def fetch_positions(self, symbol: str) -> list[Position]:
        out: list[Position] = []
        close = self._current_close()
        for p in self._state.sim.open_positions:
            out.append(Position(
                id=str(p.id), symbol=symbol,
                side="long" if p.direction == 1 else "short",
                entry_price=p.entry_price, size=p.size,
                leverage=p.leverage, unrealized_pnl=p.unrealized_pnl(close),
                margin=p.margin, liquidation_price=p.liquidation_price,
            ))
        return out

    def fetch_open_orders(self, symbol: str) -> list[Order]:
        out: list[Order] = []
        for o in self._state.sim.open_orders:
            out.append(Order(
                id=str(o.id), symbol=symbol,
                side="buy" if o.direction == 1 else "sell",
                type="limit", price=o.trigger_price, amount=o.margin,
                status="open",
            ))
        return out

    def place_order(self, symbol: str, request: OrderRequest) -> Order:
        # ReplayAdapter doesn't accept raw OrderRequest from the runner —
        # it uses the trainer's apply_intent path. This method exists to
        # satisfy the ABC and is unused in the replay flow.
        raise NotImplementedError(
            "ReplayAdapter does not implement place_order; runner uses apply_intent"
        )

    def cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError

    def close_position(self, symbol: str, position_id: str, fraction: float) -> Order:
        raise NotImplementedError

    def set_leverage(self, symbol: str, leverage: float) -> None:
        # No-op — leverage is computed per-order by ExchangeSim.
        return

    # -- internal helpers ---------------------------------------------------

    def _current_close(self) -> float:
        st = self._state
        idx = max(0, st.cursor - 1)
        if idx >= len(st.features):
            idx = len(st.features) - 1
        return float(st.features[idx][st.price_columns["close"]])
```

The "ReplayAdapter doesn't implement place_order" decision is intentional: in replay mode, the runner is allowed to call `self.exchange.sim.apply_intent(...)` directly, because both code paths are testing the simulator. This keeps the runner generic but lets the replay gate exercise the real fill logic. (Production runs use BingXAdapter and go through `place_order`.)

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/live/test_replay_adapter.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/live/exchange/replay.py \
        backend/tests/live/test_replay_adapter.py
git commit -m "feat(live): ReplayAdapter wrapping ExchangeSim"
```

### Task C.2: feature_pipeline (live thin wrapper)

**Files:**
- Create: `backend/src/live/feature_pipeline.py`

- [ ] **Step 1: Implement**

```python
"""Convert exchange-DTO klines into the trainer's normalized observation.

Thin wrapper:
  ccxt-style klines → numpy feature array
  account / positions / orders DTOs → ObservationInputs
  → trainer.env.observation.build_observation
"""
from __future__ import annotations

import numpy as np

from live.exchange.base import Balance, Kline, Order, Position
from trainer.env.normalization import NormalizationStats
from trainer.env.observation import (
    ObservationConfig,
    ObservationInputs,
    build_observation,
)
from trainer.env.exchange_sim import Order as SimOrder, Position as SimPosition


def klines_to_features(
    klines: list[Kline],
    columns: list[str],
) -> np.ndarray:
    """Project Kline DTOs into a (N, len(columns)) float32 array.

    `columns` is the same list the trainer used (e.g. ['open','high','low','close','volume']).
    """
    name_to_attr = {
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume",
    }
    rows: list[list[float]] = []
    for k in klines:
        rows.append([getattr(k, name_to_attr[c]) for c in columns])
    return np.array(rows, dtype=np.float32)


def normalize(features: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((features - stats.mean) / stats.std).astype(np.float32)


def build_live_observation(
    *,
    klines: list[Kline],
    columns: list[str],
    balance: Balance,
    positions: list[Position],
    open_orders: list[Order],
    stats: NormalizationStats,
    obs_cfg: ObservationConfig,
) -> dict[str, np.ndarray]:
    raw = klines_to_features(klines, columns)
    if raw.shape[0] != obs_cfg.lookback:
        raise ValueError(
            f"got {raw.shape[0]} klines, need {obs_cfg.lookback}"
        )
    market = normalize(raw, stats)
    close = float(klines[-1].close) if close_nonzero(klines) else 1.0

    inputs = ObservationInputs(
        market=market,
        balance=balance.available + balance.used,
        equity=balance.total,
        unrealized_pnl=balance.total - (balance.available + balance.used),
        margin_used=balance.used, available_balance=balance.available,
        open_orders=[_to_sim_order(o) for o in open_orders],
        open_positions=[_to_sim_position(p) for p in positions],
        close=close,
    )
    return build_observation(inputs, obs_cfg)


def close_nonzero(klines: list[Kline]) -> bool:
    return bool(klines) and klines[-1].close > 0


def _to_sim_order(o: Order) -> SimOrder:
    """Best-effort projection: live Order DTO has fewer fields than SimOrder.
    We use it only for the order observation rows where price/sl_price/tp slots
    are needed; SL/TP details live on the position once filled, so for orders
    we provide trigger_price, sl=trigger (no SL on the limit itself), and zero TPs.
    """
    return SimOrder(
        id=int(o.id) if o.id.isdigit() else 0,
        direction=1 if o.side == "buy" else -1,
        trigger_price=o.price or 0.0,
        sl_price=o.price or 0.0,
        tp_prices=[],
        tp_size_pcts=[],
        margin=o.amount,
    )


def _to_sim_position(p: Position) -> SimPosition:
    return SimPosition(
        id=int(p.id) if p.id.isdigit() else 0,
        direction=1 if p.side == "long" else -1,
        entry_price=p.entry_price, size=p.size, leverage=p.leverage,
        sl_price=0.0, tp_prices=[], tp_size_pcts=[],
        margin=p.margin,
        liquidation_price=p.liquidation_price or 0.0,
    )
```

- [ ] **Step 2: Commit**

```bash
git add backend/src/live/feature_pipeline.py
git commit -m "feat(live): feature_pipeline thin wrapper to trainer build_observation"
```

### Task C.3: live action decoder with risk clamping

**Files:**
- Create: `backend/src/live/action_decoder.py`
- Create: `backend/tests/live/test_action_decoder_clamp.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/live/test_action_decoder_clamp.py`:

```python
from __future__ import annotations

from live.action_decoder import RiskClampConfig, clamp_intent
from trainer.env.action_decoder import OpenIntent, OrderIntent


def test_open_margin_clamped_to_max_position_pct():
    intent = OrderIntent(
        open=OpenIntent(
            direction=1, trigger_price=100.0, sl_price=99.0,
            tp_prices=[101.0, 102.0, 103.0],
            tp_size_pcts=[1/3, 1/3, 1/3], margin=8_000.0,
        ),
        cancels=[], closes=[],
    )
    cfg = RiskClampConfig(
        equity=10_000.0, max_position_size_pct=0.50, max_leverage=3.0,
    )
    clamped = clamp_intent(intent, cfg)
    assert clamped.open is not None
    assert clamped.open.margin == 5_000.0


def test_no_open_unaffected():
    intent = OrderIntent(open=None, cancels=[1], closes=[])
    cfg = RiskClampConfig(
        equity=10_000.0, max_position_size_pct=0.50, max_leverage=3.0,
    )
    assert clamp_intent(intent, cfg) == intent
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && uv run pytest tests/live/test_action_decoder_clamp.py -v
```

Expected: `ModuleNotFoundError: No module named 'live.action_decoder'`.

- [ ] **Step 3: Implement**

`backend/src/live/action_decoder.py`:

```python
"""Live action decoder: trainer pure decoder + per-run risk clamps."""
from __future__ import annotations

from dataclasses import dataclass, replace

from trainer.env.action_decoder import OrderIntent


@dataclass(frozen=True)
class RiskClampConfig:
    equity: float                 # current account equity
    max_position_size_pct: float  # max margin as fraction of equity
    max_leverage: float           # cap on per-order leverage


def clamp_intent(intent: OrderIntent, cfg: RiskClampConfig) -> OrderIntent:
    if intent.open is None:
        return intent
    cap = cfg.equity * cfg.max_position_size_pct
    margin = min(intent.open.margin, cap)
    new_open = replace(intent.open, margin=margin)
    return replace(intent, open=new_open)
```

Leverage clamp is enforced inside `ExchangeSim.compute_leverage` (already capped at `max_leverage` from `ExchangeConfig`). The live `RiskClampConfig.max_leverage` is informational here; the runner sets the exchange's `max_leverage` config when constructing `ExchangeSim`-equivalent state, and calls `adapter.set_leverage(...)` on real exchanges.

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/live/test_action_decoder_clamp.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/live/action_decoder.py \
        backend/tests/live/test_action_decoder_clamp.py
git commit -m "feat(live): action decoder with risk clamping"
```

### Task C.4: model_runner

**Files:**
- Create: `backend/src/live/model_runner.py`

- [ ] **Step 1: Implement**

```python
"""Load a saved SB3 model and run inference."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, PPO, SAC


_ALGO_REGISTRY = {"A2C": A2C, "PPO": PPO, "SAC": SAC}


@dataclass(frozen=True)
class InferenceResult:
    action: np.ndarray
    inference_ms: int


class ModelRunner:
    def __init__(self, model_path: Path, algorithm: str):
        cls = _ALGO_REGISTRY.get(algorithm.upper())
        if cls is None:
            raise ValueError(f"unknown algorithm: {algorithm}")
        self._model = cls.load(str(model_path))

    def predict(self, obs: dict) -> InferenceResult:
        t0 = time.perf_counter_ns()
        action, _ = self._model.predict(obs, deterministic=True)
        elapsed_ns = time.perf_counter_ns() - t0
        return InferenceResult(
            action=np.asarray(action, dtype=np.float32),
            inference_ms=int(elapsed_ns // 1_000_000),
        )
```

- [ ] **Step 2: Commit**

```bash
git add backend/src/live/model_runner.py
git commit -m "feat(live): ModelRunner loads SB3 model and predicts"
```

### Task C.5: LiveRunner skeleton (replay-only path, no reconciliation yet)

**Files:**
- Create: `backend/src/live/runner.py`

The runner has the full state machine in spec; this task implements only what the replay gate needs (no DB, no reconciliation, no kill switch). Phase F adds those.

- [ ] **Step 1: Implement**

```python
"""LiveRunner — orchestrates feature pipeline → model → action decoder
→ adapter dispatch.

This file ships a skeleton with three entry points:
  - run_replay(...)  : drives a ReplayAdapter to terminal step
  - run_live(...)    : the production loop (filled in Phase F)
  - shutdown(...)    : graceful close (filled in Phase F)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from live.action_decoder import RiskClampConfig, clamp_intent
from live.exchange.replay import ReplayAdapter
from live.feature_pipeline import build_live_observation
from live.model_runner import ModelRunner
from trainer.config import ModelConfig
from trainer.env.action_decoder import DecoderState, decode_action
from trainer.env.normalization import NormalizationStats
from trainer.env.observation import ObservationConfig


@dataclass
class ReplayResult:
    final_equity: float
    total_steps: int


def run_replay(
    *,
    adapter: ReplayAdapter,
    model_runner: ModelRunner,
    model_config: ModelConfig,
    stats: NormalizationStats,
    starting_equity: float,
    max_position_size_pct: float = 1.0,
    max_leverage: float = 125.0,
) -> ReplayResult:
    """Drive the replay adapter forward one candle at a time.

    Identical control flow to TradingEnv.step:
      1. process candle (already done by adapter.advance for prior step)
      2. observe
      3. predict
      4. decode + clamp
      5. apply intent (via ReplayAdapter.sim.apply_intent for parity)
      6. advance adapter cursor
    """
    obs_cfg = ObservationConfig(
        lookback=model_config.lookback_window,
        num_features=len(model_config.columns),
        max_open_orders=model_config.exchange.max_open_orders,
        max_open_positions=model_config.exchange.max_open_positions,
        max_leverage=model_config.exchange.max_leverage,
        initial_balance=model_config.initial_balance,
    )

    # Prime the adapter so it has at least `lookback` klines available.
    while adapter.cursor < model_config.lookback_window:
        adapter.advance()

    steps = 0
    while adapter.cursor < len(adapter._state.features):
        klines = adapter.fetch_klines(
            adapter._state.symbol,
            adapter._state.interval,
            limit=model_config.lookback_window,
        )
        balance = adapter.fetch_balance()
        positions = adapter.fetch_positions(adapter._state.symbol)
        open_orders = adapter.fetch_open_orders(adapter._state.symbol)

        obs = build_live_observation(
            klines=klines, columns=model_config.columns,
            balance=balance, positions=positions, open_orders=open_orders,
            stats=stats, obs_cfg=obs_cfg,
        )
        result = model_runner.predict(obs)

        state = DecoderState(
            close=float(klines[-1].close),
            available_balance=balance.available,
            num_open_orders=len(open_orders),
            num_open_positions=len(positions),
        )
        intent = decode_action(result.action, state, model_config)
        intent = clamp_intent(intent, RiskClampConfig(
            equity=balance.total,
            max_position_size_pct=max_position_size_pct,
            max_leverage=max_leverage,
        ))

        # Replay path: bypass the adapter's place_order and call sim.apply_intent
        # directly. This is the same code TradingEnv.step uses.
        adapter.sim.apply_intent(intent, current_price=state.close)
        adapter.advance()
        steps += 1

    final_balance = adapter.fetch_balance()
    return ReplayResult(final_equity=final_balance.total, total_steps=steps)
```

- [ ] **Step 2: Commit**

```bash
git add backend/src/live/runner.py
git commit -m "feat(live): LiveRunner replay path (production path comes in Phase F)"
```

### Task C.6: live_replay script

**Files:**
- Create: `backend/scripts/live_replay.py`

- [ ] **Step 1: Implement**

```python
"""Replay gate: run the live code path against historical klines and assert
terminal equity matches the trainer's stored eval result.

Usage:
  uv run live-replay \
    --model btc_4h_a2c_lb500_3em4_p2_s1 \
    --start 2024-12-01 --end 2026-04-30 \
    --tolerance-pct 0.5
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ingester.db import connect
from live.exchange.replay import ReplayAdapter
from live.model_runner import ModelRunner
from live.runner import run_replay
from trainer.config import ModelConfig
from trainer.env.normalization import load_stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--tolerance-pct", type=float, default=0.5,
                   help="abs(live - eval) / starting_equity * 100 must be ≤ this")
    p.add_argument("--models-dir", default=None,
                   help="overrides MODELS_DIR env")
    return p.parse_args()


def load_model_config(conn, name: str) -> tuple[int, ModelConfig]:
    """Read model_configs row by name and convert to ModelConfig."""
    row = conn.execute(
        "SELECT id, config FROM model_configs WHERE name = %s", (name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"model {name!r} not found in model_configs")
    cfg = ModelConfig.from_dict({"name": name, **row[1]})
    return row[0], cfg


def expected_terminal_equity(conn, model_config_id: int) -> float:
    """Pull final balance from the most-recent eval run for this model."""
    row = conn.execute(
        """
        SELECT final_balance
        FROM training_runs
        WHERE model_config_id = %s AND run_type = 'evaluate'
              AND status = 'completed'
        ORDER BY id DESC
        LIMIT 1
        """,
        (model_config_id,),
    ).fetchone()
    if row is None:
        raise ValueError("no completed evaluate run for this model")
    return float(row[0])


def load_kline_window(conn, symbol: str, interval: str,
                     start: datetime, end: datetime,
                     columns: list[str]):
    rows = conn.execute(
        f"""
        SELECT open_time, {", ".join(columns)} FROM klines
        WHERE symbol = %s AND interval = %s
              AND open_time >= %s AND open_time < %s
        ORDER BY open_time
        """,
        (symbol, interval, start, end),
    ).fetchall()
    if not rows:
        raise ValueError("empty kline window")
    df = pd.DataFrame(rows, columns=["open_time"] + columns)
    timestamps = df["open_time"].values.astype(np.int64)
    features = df[columns].values.astype(np.float32)
    price_columns = {c: i for i, c in enumerate(columns)}
    return timestamps, features, price_columns


def main() -> int:
    args = parse_args()
    models_dir = Path(args.models_dir or "/var/lib/tradan/models")

    with connect() as conn:
        model_id, cfg = load_model_config(conn, args.model)
        expected = expected_terminal_equity(conn, model_id)

        symbol_db = cfg.symbols[0]   # DB-side symbol (e.g. BTCUSDT)
        ts, feats, price_cols = load_kline_window(
            conn, symbol=symbol_db, interval=cfg.intervals[0],
            start=datetime.fromisoformat(args.start),
            end=datetime.fromisoformat(args.end),
            columns=cfg.columns,
        )

    stats = load_stats(models_dir / args.model)
    adapter = ReplayAdapter.from_arrays(
        timestamps=ts, features=feats, price_columns=price_cols,
        symbol="BTC/USDT:USDT", interval=cfg.intervals[0],
        starting_balance=cfg.initial_balance,
        exchange_config=cfg.exchange,
    )
    model_runner = ModelRunner(
        model_path=models_dir / f"{args.model}.zip",
        algorithm=cfg.algorithm,
    )

    result = run_replay(
        adapter=adapter, model_runner=model_runner,
        model_config=cfg, stats=stats,
        starting_equity=cfg.initial_balance,
    )

    diff = abs(result.final_equity - expected)
    diff_pct = (diff / cfg.initial_balance) * 100.0
    passed = diff_pct <= args.tolerance_pct

    print(f"model:           {args.model}")
    print(f"steps:           {result.total_steps}")
    print(f"expected equity: {expected:.4f}")
    print(f"live equity:     {result.final_equity:.4f}")
    print(f"abs diff:        {diff:.6f}  ({diff_pct:.4f}% of starting equity)")
    print(f"tolerance:       {args.tolerance_pct:.4f}%")
    print(f"result:          {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def replay_main():
    """Entry point for the `live-replay` console script."""
    sys.exit(main())


if __name__ == "__main__":
    replay_main()
```

This module provides `main()` and `replay_main()`. Update the `live/cli.py` console-script entry to re-export `replay_main` (next task).

- [ ] **Step 2: Commit**

```bash
git add backend/scripts/live_replay.py
git commit -m "feat(live): live_replay script for the correctness gate"
```

### Task C.7: Add `live/cli.py` skeleton with `replay_main` proxy

**Files:**
- Create: `backend/src/live/cli.py`

The full CLI lands in Phase F; for now we need the entry point so `uv run live-replay` resolves.

- [ ] **Step 1: Implement skeleton**

`backend/src/live/cli.py`:

```python
"""Live runner CLI. Full implementation in Phase F."""
from __future__ import annotations

import sys


def main() -> int:
    print("live-test: not yet implemented (Phase F)", file=sys.stderr)
    return 2


def replay_main() -> int:
    # Defer import so users without psycopg/SB3 still see --help on live-test.
    from scripts.live_replay import replay_main as _replay
    return _replay()


if __name__ == "__main__":
    sys.exit(main())
```

`backend/scripts/live_replay.py` lives in `backend/scripts/`, which isn't on the importable path by default. Either:
- Move `live_replay.py` into `backend/src/live/` (e.g. `live/replay_cli.py`), or
- Add a small `__init__.py` discovery shim.

Cleaner: move the script's body into `backend/src/live/replay_cli.py` and have `backend/scripts/live_replay.py` be a one-line shim that imports and runs it. Do that here:

- Create `backend/src/live/replay_cli.py` containing the body of `scripts/live_replay.py`.
- Replace `backend/scripts/live_replay.py` with `from live.replay_cli import replay_main; replay_main()`.
- Update `cli.py`'s `replay_main` to `from live.replay_cli import replay_main as _replay`.

- [ ] **Step 2: Commit**

```bash
git add backend/src/live/cli.py backend/src/live/replay_cli.py backend/scripts/live_replay.py
git commit -m "feat(live): CLI skeleton + replay_main entry point"
```

### Task C.8: Gate C1 — replay passes for all three picks

This is the critical correctness gate. It must pass with **0% divergence** for each pick. Any non-zero divergence is a bug.

- [ ] **Step 1: Run replay for Pick 1**

```bash
cd backend && uv run live-replay \
  --model btc_4h_a2c_lb500_3em4_p2_s1 \
  --start 2024-12-01 --end 2026-04-30 \
  --tolerance-pct 0.5
```

Expected: `result: PASS` and `abs diff: 0.000000`.

If non-zero divergence: investigate. Most likely causes:
- Normalization stats mismatch (eval ran with one DataFeed, replay ran with another) → re-run `backfill_normalization_stats.py`.
- Feature column ordering differs between training and replay.
- Klines pulled from DB don't cover the same window the trainer's eval used.

Fix the root cause; do not adjust tolerance. Re-run.

- [ ] **Step 2: Run replay for Pick 2**

```bash
cd backend && uv run live-replay \
  --model btc_4h_a2c_lb100_3em4_p2_s0 \
  --start 2024-12-01 --end 2026-04-30 \
  --tolerance-pct 0.5
```

Expected: `result: PASS` and `abs diff: 0.000000`.

- [ ] **Step 3: Run replay for Pick 3**

```bash
cd backend && uv run live-replay \
  --model btc_4h_a2c_lb500_3em4_p2_s0 \
  --start 2024-12-01 --end 2026-04-30 \
  --tolerance-pct 0.5
```

Expected: `result: PASS` and `abs diff: 0.000000`.

- [ ] **Step 4: Document in commit log**

```bash
git commit --allow-empty -m "test(live): gate C1 passes for picks 1, 2, 3 (0% divergence)"
```

**Phase C complete.** The live code path produces bit-identical equity curves to the trainer's eval. We can now safely point this code at a real exchange.

---

## Phase D — BingX adapter (read-only)

### Task D.1: BingXAdapter read methods

**Files:**
- Modify: `backend/src/live/exchange/bingx.py`

- [ ] **Step 1: Implement**

Replace the stub:

```python
"""BingX VST adapter via ccxt.

VST = Virtual Simulation Trading: real prices, fake balance, real API surface.
"""
from __future__ import annotations

import os

import ccxt

from live.exchange.base import (
    Balance, ExchangeAdapter, Kline, Order, OrderRequest, Position,
)


class BingXAdapter(ExchangeAdapter):
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        mode: str,                  # "demo" | "live"
    ):
        if mode not in ("demo", "live"):
            raise ValueError(f"unsupported mode: {mode}")
        self._mode = mode
        self._client = ccxt.bingx({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if mode == "demo":
            # ccxt unified flag for demo trading.
            self._client.set_sandbox_mode(True)

    @classmethod
    def from_env(cls, *, api_key_env: str, api_secret_env: str, mode: str) -> "BingXAdapter":
        key = os.environ[api_key_env]
        secret = os.environ[api_secret_env]
        return cls(api_key=key, api_secret=secret, mode=mode)

    # -- read methods -------------------------------------------------------

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[Kline]:
        rows = self._client.fetch_ohlcv(symbol, interval, limit=limit)
        return [
            Kline(
                open_time_ms=int(r[0]),
                open=float(r[1]), high=float(r[2]),
                low=float(r[3]), close=float(r[4]),
                volume=float(r[5]),
            )
            for r in rows
        ]

    def fetch_balance(self) -> Balance:
        bal = self._client.fetch_balance()
        usdt = bal.get("USDT", {})
        total = float(usdt.get("total", 0.0))
        free = float(usdt.get("free", 0.0))
        used = float(usdt.get("used", 0.0))
        return Balance(total=total, available=free, used=used)

    def fetch_positions(self, symbol: str) -> list[Position]:
        rows = self._client.fetch_positions([symbol])
        out: list[Position] = []
        for r in rows:
            contracts = float(r.get("contracts") or 0.0)
            if contracts == 0:
                continue
            side = "long" if r.get("side") == "long" else "short"
            out.append(Position(
                id=str(r.get("id") or f"{symbol}-{side}"),
                symbol=symbol, side=side,
                entry_price=float(r.get("entryPrice") or 0.0),
                size=contracts,
                leverage=float(r.get("leverage") or 1.0),
                unrealized_pnl=float(r.get("unrealizedPnl") or 0.0),
                margin=float(r.get("initialMargin") or 0.0),
                liquidation_price=
                    float(r["liquidationPrice"]) if r.get("liquidationPrice") else None,
            ))
        return out

    def fetch_open_orders(self, symbol: str) -> list[Order]:
        rows = self._client.fetch_open_orders(symbol)
        out: list[Order] = []
        for r in rows:
            out.append(Order(
                id=str(r["id"]),
                symbol=symbol,
                side="buy" if r["side"] == "buy" else "sell",
                type=_map_order_type(r["type"]),
                price=float(r.get("price") or 0.0) or None,
                amount=float(r["amount"]),
                status="open",
            ))
        return out

    # write methods (stubs, Phase E) ----------------------------------------

    def place_order(self, symbol: str, request: OrderRequest) -> Order:
        raise NotImplementedError("Phase E")

    def cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError("Phase E")

    def close_position(self, symbol: str, position_id: str, fraction: float) -> Order:
        raise NotImplementedError("Phase E")

    def set_leverage(self, symbol: str, leverage: float) -> None:
        raise NotImplementedError("Phase E")


def _map_order_type(ccxt_type: str) -> str:
    mapping = {
        "limit": "limit", "market": "market",
        "stop_market": "stop", "stop": "stop",
        "take_profit_market": "take_profit", "take_profit": "take_profit",
    }
    return mapping.get(ccxt_type, "limit")
```

The exact ccxt `set_sandbox_mode` call may differ for BingX — verify by running the smoke test in Step 2 below. If `set_sandbox_mode` is not honored, swap to manually setting `self._client.urls["api"] = self._client.urls["test"]` per ccxt's pattern for the exchange.

- [ ] **Step 2: Gate D1 — smoke test**

You will need a BingX VST account (Pick 1's account) with API keys generated. Set the env vars:

```bash
export BINGX_VST_S1_API_KEY=...
export BINGX_VST_S1_API_SECRET=...
```

Run a one-shot Python script:

```bash
cd backend && uv run python -c "
from live.exchange.bingx import BingXAdapter
a = BingXAdapter.from_env(
    api_key_env='BINGX_VST_S1_API_KEY',
    api_secret_env='BINGX_VST_S1_API_SECRET',
    mode='demo',
)
klines = a.fetch_klines('BTC/USDT:USDT', '4h', limit=5)
for k in klines: print(k)
print('balance:', a.fetch_balance())
print('positions:', a.fetch_positions('BTC/USDT:USDT'))
print('open orders:', a.fetch_open_orders('BTC/USDT:USDT'))
"
```

Expected:
- 5 Kline rows with non-zero close prices.
- Balance with `total > 0` (VST default starting balance).
- Empty position and order lists (assuming a fresh VST account).

If errors: read the ccxt traceback. Common issues:
- `AuthenticationError`: keys are wrong or VST flag not set.
- `BadSymbol`: BingX wants `BTC-USDT` instead of `BTC/USDT:USDT` — check ccxt's `markets` dict.

- [ ] **Step 3: Commit**

```bash
git add backend/src/live/exchange/bingx.py
git commit -m "feat(live): BingX adapter read methods (fetch klines/balance/positions/orders)"
```

**Phase D complete.** We can read live BingX VST state.

---

## Phase E — BingX adapter (writes)

### Task E.1: BingXAdapter write methods

**Files:**
- Modify: `backend/src/live/exchange/bingx.py`

- [ ] **Step 1: Implement**

Replace the four `NotImplementedError` stubs in `BingXAdapter`:

```python
def place_order(self, symbol: str, request: OrderRequest) -> Order:
    params = {}
    if request.stop_loss is not None:
        params["stopLoss"] = {"type": "STOP_MARKET", "stopPrice": request.stop_loss}
    if request.take_profit is not None:
        params["takeProfit"] = {"type": "TAKE_PROFIT_MARKET", "stopPrice": request.take_profit}

    if request.type == "market":
        r = self._client.create_market_order(
            symbol, request.side, request.amount, params=params,
        )
    elif request.type == "limit":
        r = self._client.create_limit_order(
            symbol, request.side, request.amount, request.price, params=params,
        )
    else:
        raise ValueError(f"unsupported order type for placement: {request.type}")

    return Order(
        id=str(r["id"]),
        symbol=symbol, side=request.side, type=request.type,
        price=float(r.get("price") or request.price or 0.0) or None,
        amount=float(r["amount"]),
        status=_map_status(r.get("status")),
    )


def cancel_order(self, symbol: str, order_id: str) -> None:
    self._client.cancel_order(order_id, symbol)


def close_position(self, symbol: str, position_id: str, fraction: float) -> Order:
    positions = self.fetch_positions(symbol)
    target = next((p for p in positions if p.id == position_id), None)
    if target is None:
        raise ValueError(f"position {position_id} not found")
    qty = target.size * fraction
    side: str = "sell" if target.side == "long" else "buy"
    r = self._client.create_market_order(
        symbol, side, qty, params={"reduceOnly": True},
    )
    return Order(
        id=str(r["id"]), symbol=symbol, side=side, type="market",
        price=None, amount=qty, status=_map_status(r.get("status")),
    )


def set_leverage(self, symbol: str, leverage: float) -> None:
    self._client.set_leverage(int(leverage), symbol)
```

Add `_map_status` helper at module level:

```python
def _map_status(s: str | None) -> str:
    s = (s or "").lower()
    if s in ("open", "pending"): return "open"
    if s in ("closed", "filled"): return "filled"
    if s == "canceled" or s == "cancelled": return "cancelled"
    return "rejected"
```

- [ ] **Step 2: Gate E1 — smoke tests against VST**

Three scripts, each should run independently. Watch the BingX VST web dashboard while running them.

**(a) Place limit far below market, then cancel:**

```bash
cd backend && uv run python -c "
from live.exchange.base import OrderRequest
from live.exchange.bingx import BingXAdapter
a = BingXAdapter.from_env(
    api_key_env='BINGX_VST_S1_API_KEY',
    api_secret_env='BINGX_VST_S1_API_SECRET',
    mode='demo',
)
a.set_leverage('BTC/USDT:USDT', 3)
order = a.place_order('BTC/USDT:USDT', OrderRequest(
    side='buy', type='limit', amount=0.001, price=10000.0,
))
print('placed:', order)
a.cancel_order('BTC/USDT:USDT', order.id)
print('cancelled')
"
```

Expected: `placed:` line shows status='open'; the order appears on the BingX dashboard; `cancelled` prints; the order disappears from the dashboard.

**(b) Place market with SL/TP attached, then close:**

```bash
cd backend && uv run python -c "
from live.exchange.base import OrderRequest
from live.exchange.bingx import BingXAdapter
a = BingXAdapter.from_env(
    api_key_env='BINGX_VST_S1_API_KEY',
    api_secret_env='BINGX_VST_S1_API_SECRET',
    mode='demo',
)
mark_price = a.fetch_klines('BTC/USDT:USDT', '4h', limit=1)[-1].close
order = a.place_order('BTC/USDT:USDT', OrderRequest(
    side='buy', type='market', amount=0.001,
    stop_loss=mark_price * 0.99,
    take_profit=mark_price * 1.01,
))
print('placed:', order)
positions = a.fetch_positions('BTC/USDT:USDT')
print('positions:', positions)
closing = a.close_position('BTC/USDT:USDT', positions[0].id, fraction=1.0)
print('closed:', closing)
"
```

Expected: position appears with the SL+TP visible on the dashboard; close call returns a market order that flattens the position.

If anything misbehaves: check ccxt's BingX response shapes; `create_market_order`'s `params={"stopLoss": ..., "takeProfit": ...}` may differ in key naming for BingX specifically — consult ccxt's BingX `describe()` output.

- [ ] **Step 3: Commit**

```bash
git add backend/src/live/exchange/bingx.py
git commit -m "feat(live): BingX adapter write methods + leverage"
```

**Phase E complete.** BingXAdapter is fully functional.

---

## Phase F — Reconciliation + full LiveRunner

### Task F.1: reconciliation module

**Files:**
- Create: `backend/src/live/reconciliation.py`
- Create: `backend/tests/live/test_reconciliation.py`

Reconciliation diffs the exchange's current state against the runner's last logged state and decides whether resume is safe.

- [ ] **Step 1: Write the failing test**

`backend/tests/live/test_reconciliation.py`:

```python
from __future__ import annotations

import pytest

from live.exchange.base import Balance, Order, Position
from live.reconciliation import reconcile, ReconciliationOutcome


def _state(positions=(), open_orders=(), balance_total=10_000.0):
    return {
        "balance": {"total": balance_total, "available": balance_total, "used": 0.0},
        "positions": [_pos_dict(p) for p in positions],
        "open_orders": [_order_dict(o) for o in open_orders],
    }


def _pos(id="P1", side="long", size=0.01) -> Position:
    return Position(
        id=id, symbol="BTC/USDT:USDT", side=side,
        entry_price=100.0, size=size, leverage=3.0,
        unrealized_pnl=0.0, margin=300.0, liquidation_price=80.0,
    )


def _pos_dict(p):
    return {"id": p.id, "side": p.side, "size": p.size}


def _order(id="O1") -> Order:
    return Order(
        id=id, symbol="BTC/USDT:USDT", side="buy", type="limit",
        price=100.0, amount=0.01, status="open",
    )


def _order_dict(o):
    return {"id": o.id, "side": o.side, "amount": o.amount}


def test_resume_clean_when_states_match():
    last_logged = _state(positions=[_pos()], open_orders=[_order()])
    outcome = reconcile(
        last_logged_account_state=last_logged,
        exchange_balance=Balance(10_000.0, 10_000.0, 0.0),
        exchange_positions=[_pos()],
        exchange_orders=[_order()],
    )
    assert outcome.action == "resume"
    assert "matched" in outcome.diff_notes.lower() or outcome.diff_notes == ""


def test_refuse_when_unknown_position_at_exchange():
    last_logged = _state(positions=[_pos(id="P1")], open_orders=[])
    outcome = reconcile(
        last_logged_account_state=last_logged,
        exchange_balance=Balance(10_000.0, 10_000.0, 0.0),
        exchange_positions=[_pos(id="P1"), _pos(id="P-UNKNOWN")],
        exchange_orders=[],
    )
    assert outcome.action == "refuse"
    assert "unknown" in outcome.diff_notes.lower()


def test_refuse_when_unknown_order_at_exchange():
    last_logged = _state(positions=[], open_orders=[_order(id="O1")])
    outcome = reconcile(
        last_logged_account_state=last_logged,
        exchange_balance=Balance(10_000.0, 10_000.0, 0.0),
        exchange_positions=[],
        exchange_orders=[_order(id="O1"), _order(id="O-NEW")],
    )
    assert outcome.action == "refuse"
    assert "unknown" in outcome.diff_notes.lower()


def test_resume_when_logged_position_no_longer_at_exchange():
    """A position we logged but the exchange doesn't have means it was
    closed (SL/TP/liquidation) while we were down. That's expected; resume."""
    last_logged = _state(positions=[_pos(id="P1")], open_orders=[])
    outcome = reconcile(
        last_logged_account_state=last_logged,
        exchange_balance=Balance(9_500.0, 9_500.0, 0.0),
        exchange_positions=[],
        exchange_orders=[],
    )
    assert outcome.action == "resume"
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && uv run pytest tests/live/test_reconciliation.py -v
```

Expected: `ModuleNotFoundError: No module named 'live.reconciliation'`.

- [ ] **Step 3: Implement**

`backend/src/live/reconciliation.py`:

```python
"""Reconcile exchange state against the runner's last logged state.

Rule (per design spec):
- If the exchange has positions/orders the runner did not log → refuse to
  resume. The runner does not know how to handle state it didn't create.
- Logged positions/orders that no longer exist at the exchange are fine —
  they were closed/filled/cancelled while we were down.
- Balance differences are informational; we record them but do not block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from live.exchange.base import Balance, Order, Position


@dataclass(frozen=True)
class ReconciliationOutcome:
    action: str            # "resume" | "refuse"
    diff_notes: str        # human-readable summary


def reconcile(
    *,
    last_logged_account_state: dict[str, Any],
    exchange_balance: Balance,
    exchange_positions: list[Position],
    exchange_orders: list[Order],
) -> ReconciliationOutcome:
    logged_position_ids = {
        p["id"] for p in (last_logged_account_state.get("positions") or [])
    }
    logged_order_ids = {
        o["id"] for o in (last_logged_account_state.get("open_orders") or [])
    }

    unknown_positions = [
        p for p in exchange_positions if p.id not in logged_position_ids
    ]
    unknown_orders = [
        o for o in exchange_orders if o.id not in logged_order_ids
    ]

    if unknown_positions or unknown_orders:
        notes = "Refusing resume: unknown exchange state. "
        if unknown_positions:
            notes += f"Unknown positions: {[p.id for p in unknown_positions]}. "
        if unknown_orders:
            notes += f"Unknown orders: {[o.id for o in unknown_orders]}."
        return ReconciliationOutcome(action="refuse", diff_notes=notes.strip())

    closed_positions = (
        logged_position_ids
        - {p.id for p in exchange_positions}
    )
    cancelled_or_filled_orders = (
        logged_order_ids
        - {o.id for o in exchange_orders}
    )
    notes = []
    if closed_positions:
        notes.append(f"closed positions: {sorted(closed_positions)}")
    if cancelled_or_filled_orders:
        notes.append(f"closed orders: {sorted(cancelled_or_filled_orders)}")
    if not notes:
        notes.append("matched cleanly")

    return ReconciliationOutcome(action="resume", diff_notes="; ".join(notes))
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && uv run pytest tests/live/test_reconciliation.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/live/reconciliation.py \
        backend/tests/live/test_reconciliation.py
git commit -m "feat(live): reconciliation module"
```

### Task F.2: Wire production loop into LiveRunner

**Files:**
- Modify: `backend/src/live/runner.py`
- Modify: `backend/src/live/cli.py`

This is the largest single piece of code in the plan. It wires the state machine described in the spec.

- [ ] **Step 1: Add production-mode entry**

Append to `backend/src/live/runner.py`:

```python
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import psycopg
from psycopg.types.json import Json

from ingester.db import connect
from live.config import LiveConfig
from live.db import (
    find_running_run,
    is_kill_requested,
    log_action,
    log_order,
    log_pnl_snapshot,
    start_run,
    stop_run,
)
from live.exchange.base import Balance, ExchangeAdapter, OrderRequest
from live.exchange.registry import get_adapter_class
from live.reconciliation import reconcile


_POLL_SECONDS = 30


@dataclass
class LiveContext:
    cfg: LiveConfig
    adapter: ExchangeAdapter
    model_runner: ModelRunner
    model_config: ModelConfig
    stats: NormalizationStats
    obs_cfg: ObservationConfig
    conn: psycopg.Connection
    run_id: int
    dry_run: bool
    interval_ms: int            # candle close-to-close, ms
    last_processed_close: int   # ms timestamp
    last_pnl_snapshot_at: float # monotonic seconds
    consecutive_errors: int


def run_live(*, config_path: str, dry_run: bool = False) -> int:
    cfg = LiveConfig.from_yaml(config_path)

    # Build adapter
    adapter_cls = get_adapter_class(cfg.exchange.name)
    adapter = adapter_cls.from_env(
        api_key_env=cfg.exchange.api_key_env,
        api_secret_env=cfg.exchange.api_secret_env,
        mode=cfg.exchange.mode,
    )

    # Resolve model
    conn = connect()
    model_id, model_cfg = _load_model_cfg(conn, cfg.model.name)
    models_dir = Path(os.environ.get("MODELS_DIR", "/var/lib/tradan/models"))
    stats = load_stats(models_dir / cfg.model.name)
    model_runner = ModelRunner(
        model_path=models_dir / f"{cfg.model.name}.zip",
        algorithm=model_cfg.algorithm,
    )
    obs_cfg = ObservationConfig(
        lookback=model_cfg.lookback_window,
        num_features=len(model_cfg.columns),
        max_open_orders=model_cfg.exchange.max_open_orders,
        max_open_positions=model_cfg.exchange.max_open_positions,
        max_leverage=model_cfg.exchange.max_leverage,
        initial_balance=model_cfg.initial_balance,
    )

    # Find or create live_runs row
    existing = find_running_run(
        conn, model_config_id=model_id, exchange=cfg.exchange.name,
    )
    if existing is not None:
        run_id = existing
        _do_reconciliation(conn, adapter, run_id, cfg.market.symbol)
    else:
        with conn.transaction():
            run_id = start_run(
                conn, model_config_id=model_id,
                exchange=cfg.exchange.name, mode=cfg.exchange.mode,
                symbol=cfg.market.symbol, interval=cfg.market.interval,
                starting_equity=cfg.risk.starting_equity_quote,
                config_yaml=Path(config_path).read_text(),
                git_sha=_git_sha(),
            )

    # Set leverage on the exchange (no-op for replay)
    try:
        adapter.set_leverage(cfg.market.symbol, cfg.risk.max_leverage)
    except NotImplementedError:
        pass

    ctx = LiveContext(
        cfg=cfg, adapter=adapter, model_runner=model_runner,
        model_config=model_cfg, stats=stats, obs_cfg=obs_cfg,
        conn=conn, run_id=run_id, dry_run=dry_run,
        interval_ms=_interval_to_ms(cfg.market.interval),
        last_processed_close=0,
        last_pnl_snapshot_at=time.monotonic(),
        consecutive_errors=0,
    )

    # SIGINT/SIGTERM → graceful stop
    _install_signal_handlers(ctx)

    return _loop(ctx)


def _loop(ctx: LiveContext) -> int:
    try:
        while True:
            # 1. kill checks
            if is_kill_requested(ctx.conn, ctx.run_id):
                _shutdown(ctx, reason="kill_switch")
                return 0
            if os.environ.get(ctx.cfg.risk.kill_switch_env, "").lower() == "true":
                _shutdown(ctx, reason="kill_switch")
                return 0

            # 2. fetch latest klines, detect new candle
            try:
                klines = ctx.adapter.fetch_klines(
                    ctx.cfg.market.symbol, ctx.cfg.market.interval,
                    limit=ctx.model_config.lookback_window,
                )
                ctx.consecutive_errors = 0
            except Exception as e:
                ctx.consecutive_errors += 1
                log_action(
                    ctx.conn, live_run_id=ctx.run_id, event_type="error",
                    account_state={"error": str(e)},
                    notes=f"fetch_klines failed: {e!r}",
                )
                if ctx.consecutive_errors >= 3:
                    _shutdown(ctx, reason="error")
                    return 1
                time.sleep(_POLL_SECONDS)
                continue

            newest_close = klines[-1].open_time_ms + ctx.interval_ms
            if newest_close > ctx.last_processed_close:
                _on_new_candle(ctx, klines)
                ctx.last_processed_close = newest_close

            # 3. periodic pnl snapshot + drawdown check
            elapsed = time.monotonic() - ctx.last_pnl_snapshot_at
            if elapsed >= ctx.cfg.logging.pnl_snapshot_interval_minutes * 60:
                _take_snapshot(ctx)
                ctx.last_pnl_snapshot_at = time.monotonic()
                bal = ctx.adapter.fetch_balance()
                threshold = ctx.cfg.risk.starting_equity_quote * (
                    1.0 - ctx.cfg.risk.max_drawdown_pct
                )
                if bal.total < threshold:
                    _shutdown(ctx, reason="drawdown")
                    return 0

            time.sleep(_POLL_SECONDS)
    except _GracefulExit as e:
        _shutdown(ctx, reason=e.reason)
        return 0
    except Exception as e:
        _shutdown(ctx, reason="error")
        log_action(
            ctx.conn, live_run_id=ctx.run_id, event_type="error",
            account_state={"error": str(e)}, notes=repr(e),
        )
        return 1


def _on_new_candle(ctx: LiveContext, klines: list[Kline]) -> None:
    bal = ctx.adapter.fetch_balance()
    positions = ctx.adapter.fetch_positions(ctx.cfg.market.symbol)
    open_orders = ctx.adapter.fetch_open_orders(ctx.cfg.market.symbol)

    obs = build_live_observation(
        klines=klines, columns=ctx.model_config.columns,
        balance=bal, positions=positions, open_orders=open_orders,
        stats=ctx.stats, obs_cfg=ctx.obs_cfg,
    )
    pred = ctx.model_runner.predict(obs)

    candle_close_ts = datetime.fromtimestamp(
        (klines[-1].open_time_ms + ctx.interval_ms) / 1000.0, tz=timezone.utc,
    )
    account_state = _account_state_dict(bal, positions, open_orders)

    action_id = log_action(
        ctx.conn, live_run_id=ctx.run_id, event_type="inference",
        candle_close=candle_close_ts,
        raw_action=pred.action.tolist(),
        account_state=account_state, inference_ms=pred.inference_ms,
    )

    state = DecoderState(
        close=float(klines[-1].close),
        available_balance=bal.available,
        num_open_orders=len(open_orders),
        num_open_positions=len(positions),
    )
    intent = decode_action(pred.action, state, ctx.model_config)
    intent = clamp_intent(intent, RiskClampConfig(
        equity=bal.total,
        max_position_size_pct=ctx.cfg.risk.max_position_size_pct,
        max_leverage=ctx.cfg.risk.max_leverage,
    ))

    ctx.conn.execute(
        "UPDATE live_actions SET decoded_intent = %s WHERE id = %s",
        (Json(_intent_dict(intent)), action_id),
    )

    if not ctx.dry_run:
        _execute_intent(ctx, intent, action_id, positions, open_orders)


def _execute_intent(ctx, intent, action_id, positions, open_orders) -> None:
    # Cancel
    for i in sorted(intent.cancels, reverse=True):
        if i < len(open_orders):
            target = open_orders[i]
            ctx.adapter.cancel_order(ctx.cfg.market.symbol, target.id)
            log_order(
                ctx.conn, live_run_id=ctx.run_id, live_action_id=action_id,
                exchange_order_id=target.id, side=target.side, type=target.type,
                price=target.price, amount=target.amount, status="cancelled",
            )

    # Close
    for ci in intent.closes:
        if ci.position_index < len(positions):
            target = positions[ci.position_index]
            order = ctx.adapter.close_position(
                ctx.cfg.market.symbol, target.id, ci.fraction,
            )
            log_order(
                ctx.conn, live_run_id=ctx.run_id, live_action_id=action_id,
                exchange_order_id=order.id, side=order.side, type=order.type,
                price=order.price, amount=order.amount, status=order.status,
            )

    # Open
    if intent.open is not None:
        op = intent.open
        # Translate margin → contract size at current price.
        amount = (op.margin * _approx_leverage(intent.open, ctx)) / op.trigger_price
        side = "buy" if op.direction == 1 else "sell"
        order = ctx.adapter.place_order(
            ctx.cfg.market.symbol,
            OrderRequest(
                side=side, type="limit", amount=amount, price=op.trigger_price,
                stop_loss=op.sl_price, take_profit=op.tp_prices[0] if op.tp_prices else None,
            ),
        )
        log_order(
            ctx.conn, live_run_id=ctx.run_id, live_action_id=action_id,
            exchange_order_id=order.id, side=order.side, type=order.type,
            price=order.price, amount=order.amount, status=order.status,
        )


def _take_snapshot(ctx: LiveContext) -> None:
    bal = ctx.adapter.fetch_balance()
    positions = ctx.adapter.fetch_positions(ctx.cfg.market.symbol)
    open_orders = ctx.adapter.fetch_open_orders(ctx.cfg.market.symbol)
    realized = bal.total - sum(p.unrealized_pnl for p in positions) - ctx.cfg.risk.starting_equity_quote
    log_pnl_snapshot(
        ctx.conn, live_run_id=ctx.run_id,
        equity=bal.total, realized_pnl=realized,
        unrealized_pnl=sum(p.unrealized_pnl for p in positions),
        open_positions=len(positions), open_orders=len(open_orders),
    )


def _shutdown(ctx: LiveContext, *, reason: str) -> None:
    """Graceful close: cancel orders, flatten positions, finalize run row."""
    try:
        for o in ctx.adapter.fetch_open_orders(ctx.cfg.market.symbol):
            try:
                ctx.adapter.cancel_order(ctx.cfg.market.symbol, o.id)
            except Exception as e:
                log_action(ctx.conn, live_run_id=ctx.run_id, event_type="error",
                           account_state={}, notes=f"cancel during shutdown failed: {e!r}")
        for p in ctx.adapter.fetch_positions(ctx.cfg.market.symbol):
            try:
                ctx.adapter.close_position(ctx.cfg.market.symbol, p.id, fraction=1.0)
            except Exception as e:
                log_action(ctx.conn, live_run_id=ctx.run_id, event_type="error",
                           account_state={}, notes=f"close during shutdown failed: {e!r}")
    finally:
        _take_snapshot(ctx)
        with ctx.conn.transaction():
            stop_run(ctx.conn, ctx.run_id, reason=reason)


def _do_reconciliation(conn, adapter, run_id, symbol):
    last = conn.execute(
        """
        SELECT account_state FROM live_actions
        WHERE live_run_id = %s
        ORDER BY created_at DESC LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    last_state = last[0] if last else {}
    bal = adapter.fetch_balance()
    positions = adapter.fetch_positions(symbol)
    open_orders = adapter.fetch_open_orders(symbol)
    outcome = reconcile(
        last_logged_account_state=last_state,
        exchange_balance=bal,
        exchange_positions=positions,
        exchange_orders=open_orders,
    )
    log_action(
        conn, live_run_id=run_id, event_type="reconciliation",
        account_state=_account_state_dict(bal, positions, open_orders),
        notes=outcome.diff_notes,
    )
    if outcome.action == "refuse":
        with conn.transaction():
            stop_run(conn, run_id, reason="reconciliation_failed")
        sys.exit(2)


# -- helpers --

class _GracefulExit(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def _install_signal_handlers(ctx: LiveContext) -> None:
    def handler(_signum, _frame):
        raise _GracefulExit(reason="manual")
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
        ).decode().strip()
    except Exception:
        return "unknown"


def _interval_to_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000, "5m": 5 * 60_000, "15m": 15 * 60_000,
        "30m": 30 * 60_000, "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000, "1d": 24 * 60 * 60_000,
    }
    return mapping[interval]


def _account_state_dict(bal, positions, orders) -> dict:
    return {
        "balance": {"total": bal.total, "available": bal.available, "used": bal.used},
        "positions": [{"id": p.id, "side": p.side, "size": p.size,
                       "entry_price": p.entry_price, "leverage": p.leverage} for p in positions],
        "open_orders": [{"id": o.id, "side": o.side, "type": o.type,
                         "price": o.price, "amount": o.amount} for o in orders],
    }


def _intent_dict(intent) -> dict:
    return {
        "open": (intent.open and {
            "direction": intent.open.direction,
            "trigger_price": intent.open.trigger_price,
            "sl_price": intent.open.sl_price,
            "tp_prices": intent.open.tp_prices,
            "tp_size_pcts": intent.open.tp_size_pcts,
            "margin": intent.open.margin,
        }),
        "cancels": list(intent.cancels),
        "closes": [{"position_index": c.position_index, "fraction": c.fraction}
                   for c in intent.closes],
    }


def _approx_leverage(open_intent, ctx) -> float:
    """Approximate leverage from SL distance — keeps the live order amount
    consistent with what TradingEnv.compute_leverage produces."""
    sl_dist = abs(open_intent.trigger_price - open_intent.sl_price) / open_intent.trigger_price
    if sl_dist == 0:
        return 1.0
    mm = ctx.model_config.exchange.maintenance_margin_pct / 100.0
    buf = ctx.model_config.exchange.liquidation_buffer_pct / 100.0
    lev = 1.0 / (sl_dist + buf + mm)
    return min(lev, ctx.cfg.risk.max_leverage)


def _load_model_cfg(conn, name: str):
    row = conn.execute(
        "SELECT id, config FROM model_configs WHERE name = %s", (name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"model {name!r} not found in model_configs")
    return row[0], ModelConfig.from_dict({"name": name, **row[1]})
```

`from psycopg.types.json import Json` import added at the top.

- [ ] **Step 2: Implement CLI**

Replace `backend/src/live/cli.py`:

```python
"""Live runner CLI: live-test."""
from __future__ import annotations

import argparse
import sys

from ingester.db import connect


def _cmd_run(args) -> int:
    from live.runner import run_live
    return run_live(config_path=args.config, dry_run=args.dry_run)


def _cmd_status(args) -> int:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT lr.id, mc.name, lr.exchange, lr.mode, lr.started_at,
                   lr.starting_equity, lr.kill_requested,
                   COALESCE(
                       (SELECT equity FROM live_pnl_snapshots
                        WHERE live_run_id = lr.id
                        ORDER BY taken_at DESC LIMIT 1),
                       lr.starting_equity
                   ) AS current_equity
            FROM live_runs lr
            JOIN model_configs mc ON mc.id = lr.model_config_id
            WHERE lr.status = 'running'
            ORDER BY lr.started_at
            """,
        ).fetchall()
    if not rows:
        print("No active runs.")
        return 0
    print(f"{'id':>4}  {'model':<40}  {'exch':<8}  {'mode':<6}  "
          f"{'start_equity':>14}  {'cur_equity':>14}  {'kill?':>5}")
    for r in rows:
        run_id, name, exc, mode, started, start_eq, kill, cur_eq = r
        print(f"{run_id:>4}  {name:<40}  {exc:<8}  {mode:<6}  "
              f"{float(start_eq):>14.2f}  {float(cur_eq):>14.2f}  {str(kill):>5}")
    return 0


def _cmd_stop(args) -> int:
    from live.db import request_stop
    with connect() as conn:
        conn.autocommit = True
        request_stop(conn, args.run_id)
    print(f"Stop requested for run {args.run_id}.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="live-test")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Start (or resume) a live run.")
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=_cmd_run)

    sub.add_parser("status", help="List active runs.").set_defaults(func=_cmd_status)

    stop = sub.add_parser("stop", help="Request graceful stop of a run.")
    stop.add_argument("run_id", type=int)
    stop.set_defaults(func=_cmd_stop)

    args, unknown = p.parse_known_args()
    if args.cmd is None:
        # Default: act like `run` if --config is in argv.
        if "--config" in sys.argv:
            run_args = run.parse_args(sys.argv[1:])
            return _cmd_run(run_args)
        p.print_help()
        return 1
    return args.func(args)


def replay_main() -> int:
    from live.replay_cli import replay_main as _replay
    return _replay()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Confirm imports compile**

```bash
cd backend && uv run python -c "from live.runner import run_live; from live.cli import main; print('ok')"
```

Expected: `ok`. If anything fails, fix imports before proceeding.

- [ ] **Step 4: Commit**

```bash
git add backend/src/live/runner.py backend/src/live/cli.py
git commit -m "feat(live): production LiveRunner loop + CLI subcommands"
```

### Task F.3: Gate F1 — 24h dry-run against BingX VST

- [ ] **Step 1: Create configs**

Create `backend/configs/live/live-s1.yaml`:

```yaml
exchange:
  name: bingx
  mode: demo
  api_key_env: BINGX_VST_S1_API_KEY
  api_secret_env: BINGX_VST_S1_API_SECRET
market:
  symbol: BTC/USDT:USDT
  interval: 4h
model:
  name: btc_4h_a2c_lb500_3em4_p2_s1
risk:
  starting_equity_quote: 10000
  max_drawdown_pct: 0.20
  max_position_size_pct: 0.50
  max_leverage: 3
  kill_switch_env: TRADAN_KILL_SWITCH_S1
logging:
  pnl_snapshot_interval_minutes: 60
```

Create `live-s2.yaml` (Pick 2: `btc_4h_a2c_lb100_3em4_p2_s0`, `BINGX_VST_S2_*`, `TRADAN_KILL_SWITCH_S2`).

Create `live-s3.yaml` (Pick 3: `btc_4h_a2c_lb500_3em4_p2_s0`, `BINGX_VST_S3_*`, `TRADAN_KILL_SWITCH_S3`).

- [ ] **Step 2: Run dry-run for Pick 1**

```bash
cd backend && BINGX_VST_S1_API_KEY=... BINGX_VST_S1_API_SECRET=... \
  MODELS_DIR=/var/lib/tradan/models \
  uv run live-test run --config configs/live/live-s1.yaml --dry-run
```

Expected: process starts, every 30s polls klines, on candle close writes a `live_actions` row with `decoded_intent`, never places an actual order.

- [ ] **Step 3: Verify in DB**

In another terminal:

```bash
psql "$DATABASE_URL" -c "
SELECT id, mode, status, started_at FROM live_runs ORDER BY id DESC LIMIT 5;
"
psql "$DATABASE_URL" -c "
SELECT count(*), max(created_at) FROM live_actions;
"
```

Expected: a `running` row in mode `demo`; live_actions count grows after each 4h candle close.

- [ ] **Step 4: Confirm `--status` and `--stop`**

```bash
uv run live-test status
```

Expected: a row showing the running run with `current_equity` ≈ starting equity.

```bash
uv run live-test stop <run_id>
```

Expected: the runner notices kill_requested within 30s, flattens, exits cleanly. `live-test status` shows no active runs.

- [ ] **Step 5: Commit configs**

```bash
git add backend/configs/live/live-s1.yaml \
        backend/configs/live/live-s2.yaml \
        backend/configs/live/live-s3.yaml
git commit -m "feat(live): per-pick configs (live-s1/s2/s3)"
```

**Phase F complete.** The runner works end-to-end against BingX VST in dry-run mode.

---

## Phase G — Systemd + first live (Pick 1)

### Task G.1: systemd template + deploy script

**Files:**
- Create: `infra/systemd/tradan-live@.service`
- Create: `infra/scripts/live_runner_deploy.sh`

- [ ] **Step 1: Create unit file**

`infra/systemd/tradan-live@.service`:

```ini
[Unit]
Description=Tradan live runner (%i)
After=network-online.target

[Service]
Type=simple
Environment=PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EnvironmentFile=/etc/tradan/live-%i.env
WorkingDirectory=/opt/tradan/backend
ExecStart=/root/.local/bin/uv run live-test run --config configs/live/live-%i.yaml
Restart=always
RestartSec=10
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Create deploy script**

`infra/scripts/live_runner_deploy.sh`:

```bash
#!/usr/bin/env bash
# Install systemd unit template and create env files (templates) for s1/s2/s3.
# Idempotent: existing env files are not overwritten.
set -euo pipefail

UNIT_SRC="$(dirname "$0")/../systemd/tradan-live@.service"
UNIT_DST="/etc/systemd/system/tradan-live@.service"
ENV_DIR="/etc/tradan"

install -m 0644 "$UNIT_SRC" "$UNIT_DST"
install -d -m 0755 "$ENV_DIR"

for instance in s1 s2 s3; do
    env_file="$ENV_DIR/live-${instance}.env"
    if [[ ! -f "$env_file" ]]; then
        cat > "$env_file" <<EOF
# Tradan live runner env for instance ${instance}
BINGX_VST_${instance^^}_API_KEY=
BINGX_VST_${instance^^}_API_SECRET=
TRADAN_KILL_SWITCH_${instance^^}=false
DATABASE_URL=
MODELS_DIR=/var/lib/tradan/models
EOF
        chmod 0600 "$env_file"
        echo "[install] created template ${env_file} (fill in secrets manually)"
    else
        echo "[skip]    ${env_file} exists, leaving as-is"
    fi
done

systemctl daemon-reload
echo "[ok] systemd unit installed; enable individual instances with:"
echo "     systemctl enable --now tradan-live@s1.service"
```

- [ ] **Step 3: Deploy on tradan**

```bash
ssh tradan 'cd /opt/tradan && git pull && bash infra/scripts/live_runner_deploy.sh'
```

Expected: unit installed, three env file templates created (or skipped). On first run, the templates need editing.

- [ ] **Step 4: Provision Pick 1's env file**

```bash
ssh tradan 'cat > /etc/tradan/live-s1.env <<EOF
BINGX_VST_S1_API_KEY=<your-pick-1-api-key>
BINGX_VST_S1_API_SECRET=<your-pick-1-secret>
TRADAN_KILL_SWITCH_S1=false
DATABASE_URL=<dsn>
MODELS_DIR=/var/lib/tradan/models
EOF
chmod 600 /etc/tradan/live-s1.env'
```

- [ ] **Step 5: Commit**

```bash
git add infra/systemd/tradan-live@.service infra/scripts/live_runner_deploy.sh
git commit -m "feat(infra): systemd template + deploy script for live runners"
```

### Task G.2: Gate G1 — Pick 1 live (mode demo, NOT --dry-run) for 24h

- [ ] **Step 1: Enable and start the service**

```bash
ssh tradan 'systemctl enable --now tradan-live@s1.service && \
            systemctl status tradan-live@s1.service'
```

Expected: `active (running)`.

- [ ] **Step 2: Tail logs for the first hour**

```bash
ssh tradan 'journalctl -u tradan-live@s1 -f -n 100'
```

Watch for:
- "Reconciliation passed" (since this is a fresh account, the first start creates a new live_run; reconciliation is skipped).
- Polling logs every 30s.
- On the next 4h candle close, an inference event.

- [ ] **Step 3: Check DB at the 4h mark**

```bash
psql "$DATABASE_URL" -c "
SELECT id, status, mode, started_at FROM live_runs WHERE status = 'running';
SELECT count(*) FROM live_actions WHERE event_type = 'inference';
SELECT * FROM live_orders ORDER BY id DESC LIMIT 5;
"
```

Expected: a running row, ≥1 inference action, possibly orders if the model decided to trade.

- [ ] **Step 4: Cross-check BingX dashboard**

Log into the BingX VST dashboard for Pick 1's account. The orders/positions visible there must match `live_orders`/`live_pnl_snapshots`.

- [ ] **Step 5: Test resume-on-restart**

```bash
ssh tradan 'systemctl restart tradan-live@s1.service'
ssh tradan 'journalctl -u tradan-live@s1 -n 50'
```

Expected:
- The runner re-attaches to the existing `live_runs` row (not creating a new one).
- A `reconciliation` event is logged in `live_actions`.
- The loop continues from the next candle close.

Verify in DB:

```bash
psql "$DATABASE_URL" -c "
SELECT event_type, count(*) FROM live_actions
WHERE live_run_id = <pick-1-run-id>
GROUP BY event_type;
"
```

Expected: `reconciliation` count ≥ 1.

- [ ] **Step 6: Document Gate G1 pass**

```bash
git commit --allow-empty -m "test(live): gate G1 — pick 1 live + restart resumes cleanly"
```

### Task G.3: Gate G2 — kill-switch test

- [ ] **Step 1: Trigger kill switch via env file**

```bash
ssh tradan "sed -i 's/TRADAN_KILL_SWITCH_S1=false/TRADAN_KILL_SWITCH_S1=true/' /etc/tradan/live-s1.env"
ssh tradan 'systemctl restart tradan-live@s1.service'
```

(Restart picks up the env-file change. If kill switch is detected at startup it must shut down within ≤60s.)

- [ ] **Step 2: Watch shutdown logs**

```bash
ssh tradan 'journalctl -u tradan-live@s1 -n 50'
```

Expected: graceful shutdown sequence — cancel orders, close positions, final pnl snapshot, stop run.

- [ ] **Step 3: Verify in DB**

```bash
psql "$DATABASE_URL" -c "
SELECT status, stop_reason FROM live_runs WHERE id = <pick-1-run-id>;
"
```

Expected: `status='stopped'`, `stop_reason='kill_switch'`.

- [ ] **Step 4: Reset env and resume**

```bash
ssh tradan "sed -i 's/TRADAN_KILL_SWITCH_S1=true/TRADAN_KILL_SWITCH_S1=false/' /etc/tradan/live-s1.env"
ssh tradan 'systemctl restart tradan-live@s1.service'
```

Expected: a NEW `live_runs` row is created (since the previous one is `stopped`). Reconciliation is skipped.

- [ ] **Step 5: Document Gate G2 pass**

```bash
git commit --allow-empty -m "test(live): gate G2 — kill switch flattens cleanly"
```

**Phase G complete for Pick 1.**

---

## Phase H — Picks 2 & 3 + 4-week observation

### Task H.1: Provision Pick 2 and Pick 3

- [ ] **Step 1: Provision Pick 2 env file**

```bash
ssh tradan 'cat > /etc/tradan/live-s2.env <<EOF
BINGX_VST_S2_API_KEY=<your-pick-2-api-key>
BINGX_VST_S2_API_SECRET=<your-pick-2-secret>
TRADAN_KILL_SWITCH_S2=false
DATABASE_URL=<dsn>
MODELS_DIR=/var/lib/tradan/models
EOF
chmod 600 /etc/tradan/live-s2.env'
```

- [ ] **Step 2: Re-run replay gate for Pick 2 to confirm 0% divergence**

```bash
cd backend && uv run live-replay \
  --model btc_4h_a2c_lb100_3em4_p2_s0 \
  --start 2024-12-01 --end 2026-04-30 \
  --tolerance-pct 0.5
```

Expected: PASS, abs diff 0.000000.

- [ ] **Step 3: Enable Pick 2**

```bash
ssh tradan 'systemctl enable --now tradan-live@s2.service'
```

Re-run Gate G1 + Gate G2 against Pick 2 (same checklist as Task G.2 + G.3, with `s2` substituted).

- [ ] **Step 4: Repeat for Pick 3**

Provision `live-s3.env`, run replay gate for `btc_4h_a2c_lb500_3em4_p2_s0`, enable `tradan-live@s3.service`, run G1+G2.

### Task H.2: 4-week observation runbook

- [ ] **Step 1: Create weekly checks query file**

`backend/scripts/live_weekly_check.sql`:

```sql
-- Run every Monday for a quick health snapshot of all live runs.
\echo 'Active runs:'
SELECT lr.id, mc.name, lr.exchange, lr.starting_equity,
       COALESCE((SELECT equity FROM live_pnl_snapshots
                 WHERE live_run_id = lr.id ORDER BY taken_at DESC LIMIT 1),
                lr.starting_equity) AS current_equity,
       (SELECT count(*) FROM live_orders
        WHERE live_run_id = lr.id AND status = 'filled') AS filled_count,
       lr.started_at, now() - lr.started_at AS up_for
FROM live_runs lr
JOIN model_configs mc ON mc.id = lr.model_config_id
WHERE lr.status = 'running'
ORDER BY lr.started_at;

\echo
\echo 'Action log gaps (>1 hour between consecutive inference events):'
WITH ordered AS (
  SELECT live_run_id, candle_close,
         lag(candle_close) OVER (PARTITION BY live_run_id ORDER BY candle_close) AS prev
  FROM live_actions WHERE event_type = 'inference'
)
SELECT live_run_id, prev AS gap_start, candle_close AS gap_end,
       candle_close - prev AS gap
FROM ordered
WHERE candle_close - prev > interval '1 hour'
ORDER BY gap DESC LIMIT 20;

\echo
\echo 'Drawdown from start:'
SELECT lr.id, mc.name,
       lr.starting_equity AS start_eq,
       (SELECT min(equity) FROM live_pnl_snapshots WHERE live_run_id = lr.id) AS min_eq,
       1.0 - (SELECT min(equity) FROM live_pnl_snapshots WHERE live_run_id = lr.id)
              / lr.starting_equity AS dd_pct
FROM live_runs lr
JOIN model_configs mc ON mc.id = lr.model_config_id
WHERE lr.status = 'running';

\echo
\echo 'Trades per week:'
SELECT lr.id, mc.name,
       (SELECT count(*) FROM live_orders WHERE live_run_id = lr.id AND status = 'filled') AS total_filled,
       extract(epoch from (now() - lr.started_at)) / 86400.0 / 7.0 AS weeks,
       (SELECT count(*) FROM live_orders WHERE live_run_id = lr.id AND status = 'filled')
         / nullif(extract(epoch from (now() - lr.started_at)) / 86400.0 / 7.0, 0)
         AS trades_per_week
FROM live_runs lr JOIN model_configs mc ON mc.id = lr.model_config_id
WHERE lr.status = 'running';
```

Run weekly with:

```bash
psql "$DATABASE_URL" -f backend/scripts/live_weekly_check.sql
```

- [ ] **Step 2: Document success criteria thresholds**

The success criteria (from the spec) are checked against this query's output:
- **No catastrophic drawdown:** `dd_pct < 0.20` for all active runs.
- **Trade frequency matches backtest:** `trades_per_week` within ±50% of holdout target (Pick 1 ≈ 0.9, Pick 2 ≈ 0.3, Pick 3 ≈ 1.2).
- **Sign of PnL matches sign of backtest expectation:** breakeven or better over the window.
- **No silent failures:** zero rows in the "gaps > 1 hour" output.

A pick that fails the drawdown criterion ends the test for that model. Failures on frequency/PnL demote the model to "needs more training data" without disqualifying the architecture.

- [ ] **Step 3: Commit runbook script**

```bash
git add backend/scripts/live_weekly_check.sql
git commit -m "feat(live): weekly observation check runbook"
```

---

## Phase I — Optional monitoring (deferred)

Out of scope for this plan. If `psql` queries become friction during the 4-week observation window, follow-up work would add:
- FastAPI endpoints `/api/live/runs`, `/api/live/runs/{id}/pnl`, `/api/live/runs/{id}/actions`.
- Frontend "Live" tab.

That is a separate spec/plan if pursued.

---

## Spec coverage check

Cross-referenced against the design spec:

- [x] Parity strategy → Phase A (extracted modules + Gate A1 regression test).
- [x] Crash recovery → Task F.1 (reconciliation), Task F.2 (resume in `run_live`).
- [x] Deployment → Phase G (systemd template).
- [x] VST accounts → Task G.1 step 4 + Task H.1 (per-pick env files).
- [x] Kline feed (poll 30s) → `_POLL_SECONDS` constant in runner.py.
- [x] PnL table → Task B.3 (`live_pnl_snapshots` schema).
- [x] Replay gate (0% target) → Phase C, Task C.8 enforced.
- [x] Database schema → Task B.3.
- [x] LiveRunner state machine → Task F.2.
- [x] Replay correctness gate → Task C.6 + C.8.
- [x] CLI (`live-test run/status/stop`, `live-replay`) → Task F.2 + Task C.7.
- [x] Configs → Task F.3 step 1.
- [x] Systemd integration → Task G.1.
- [x] Secrets handling → Task G.1 step 2 + step 4.
- [x] File map → matches the plan's "Created/Modified" lists.
- [x] Implementation phases → 1:1 with this plan's Phases A–I.
- [x] Operational concerns (time alignment, symbol mapping, fee parity, slippage, stopping conditions, rollback) → Documented in spec; runner.py implements stopping conditions; symbol mapping in BingXAdapter; rollback via systemd-stop sequence.
- [x] Open items deferred (normalization scaler, ccxt VST flag, inference_ms baseline) → Tasks A.1, D.1 step 1, and weekly check runbook respectively.

No gaps.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-09-live-testing-bingx.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
