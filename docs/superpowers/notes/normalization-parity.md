# Normalization Parity Bug Investigation

## Summary
**BUG CONFIRMED.** The `DataFeed` class computes feature normalization statistics (`_mean`, `_std`) at initialization time from whatever feature array is passed in. Training and evaluation currently pass different slices of data, resulting in divergent normalization — the model trains with train-slice statistics but evaluates with holdout-slice statistics. This is a correctness bug that causes train-eval divergence.

## Step 1: Call Sites and Data Slices

### Training (trainer.py)
**File:** `backend/src/trainer/training/trainer.py:269-274`

```python
split_idx = int(data_feed.total_steps * 0.8)
train_timestamps = data_feed.timestamps[: split_idx + config.lookback_window]
train_features = data_feed.raw_features[: split_idx + config.lookback_window]

train_feed = DataFeed(
    timestamps=train_timestamps,
    features=train_features,  # <-- 80% train slice
    lookback=config.lookback_window,
    price_columns=data_feed.price_columns,
)
```

**Data slice passed:** The **first 80%** of `data_feed.raw_features` (indices 0 to `split_idx + lookback_window`).

### Evaluation (evaluator.py)
**File:** `backend/src/trainer/training/evaluator.py:51-56`

```python
split_idx = int(data_feed.total_steps * 0.8)
holdout_start = split_idx
holdout_timestamps = data_feed.timestamps[holdout_start:]
holdout_features = data_feed.raw_features[holdout_start:]

holdout_feed = DataFeed(
    timestamps=holdout_timestamps,
    features=holdout_features,  # <-- 20% holdout slice
    lookback=config.lookback_window,
    price_columns=data_feed.price_columns,
)
```

**Data slice passed:** The **last 20%** of `data_feed.raw_features` (indices `split_idx` to end).

## Step 2: Normalization Computation Confirmation

**File:** `backend/src/trainer/env/data_feed.py:23-25`

```python
self._mean = self.raw_features.mean(axis=0)
self._std = self.raw_features.std(axis=0)
self._std[self._std < 1e-8] = 1.0
```

**Confirmation:** `DataFeed.__init__` computes `_mean` and `_std` from axis-0 mean/std of the features array passed in. There is **no mechanism to load pre-computed statistics** — they are always computed fresh from the input array.

### Usage in Normalization
**File:** `backend/src/trainer/env/data_feed.py:39`

```python
return ((window - self._mean) / self._std).astype(np.float32)
```

All observations returned by `get_observation()` are normalized using these instance-specific statistics.

## Bug Impact

- **Training (trainer.py):** Model learns to predict on data normalized by **train-slice mean/std**.
- **Evaluation (evaluator.py):** Model is tested on data normalized by **holdout-slice mean/std**.
- **Result:** Even if the model generalizes perfectly, the train-eval equity curves will diverge solely due to different normalization constants.

This bug propagates to live testing: the live runner will compute its own normalization from the live data slice, diverging from both train and eval.

## Step 3: Persistence Decision

### Solution: Store Mean and Std as Files

**Where:** In the same directory as the trained SB3 model `.zip` file.

**Files:**
- `mean.npy` — NumPy array of shape `(num_features,)` containing `_mean`
- `std.npy` — NumPy array of shape `(num_features,)` containing `_std`

**Location in existing code:**
```python
model_dir = MODELS_DIR / config.name / str(run_id)
# E.g., models/BTCUSDT_RSI/42/
# Will also contain:
#   checkpoint_1000000_steps.zip
#   best_model.zip
#   mean.npy
#   std.npy
```

### Rationale

1. **Already-shipped models did not save stats.** The trainer must fail **loudly** if stats files are missing during eval or live — this forces explicit migration and prevents silent divergence on old models.

2. **Rejected alternative: Store in `model_configs` DB row.**
   - Reason for rejection: Filesystem is cleaner and consistent with how other artifacts (checkpoints, best_model.zip) are organized.
   - Reason for rejection: No schema migration needed; stats are treated as ephemeral training artifacts stored alongside the model, not metadata.
   - Reason for rejection: Simpler to debug — grep/find on the filesystem vs querying the database.

### Implementation Plan (Phase A.2 onwards)

1. After training, trainer.py persists `_mean` and `_std` to `mean.npy` and `std.npy` in `model_dir`.
2. Evaluator.py loads these files and passes them to a new `DataFeed` constructor parameter (e.g., `mean=`, `std=`).
3. If `mean` and `std` are provided, skip self-computation and use those values.
4. Live runner does the same: load stats from the model directory and use them.

This ensures all three contexts (train, eval, live) normalize identically.
