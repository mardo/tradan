# Phase 4 — Trading Env Risk Audit Design

**Date:** 2026-05-10
**Status:** Design (pre-implementation)
**Depends on:** Phase 4A and 4C results (`docs/plans/2026-05-09-phase4-training-plan.md`).

---

## Motivation

Phase 4A and 4C both failed the seed-robustness gate, with multiple seeds blowing the account through 40–66% drawdowns on holdout. The pattern across both phases:

| failure | how it shows up |
|---|---|
| Active-but-losing | s0/s3/s4 in 4A and 4C: 33–791 trades, drawdowns 50–117% |
| Liquidation cascade | drawdown >100% means margin lost on multiple positions |
| Tight-SL leverage runaway | model picks SL=0.1% → env auto-scales leverage to ~100x |

These are env-design symptoms, not algorithm symptoms. Two algorithmic levers (seed plumbing in 4A, entropy reg in 4C) systematically improved per-seed PnL but didn't clear the bar. The remaining gap is in the env permitting failure modes a real exchange would prevent.

This audit documents the env's risk-control gaps and applies surgical fixes targeting **account blow-ups** specifically. Reward shaping and execution realism (slippage) are deferred — they address the train→holdout overfit gap, which is a separate priority.

## Findings

### F1 — Leverage runaway via tight stop-loss

`backend/src/trainer/env/exchange_sim.py:66-76`:
```python
def compute_leverage(self, entry_price, sl_price, direction) -> float:
    sl_dist_pct = abs(entry_price - sl_price) / entry_price
    ...
    denominator = sl_dist_pct + buf + mm
    leverage = 1.0 / denominator
    return min(leverage, self.config.max_leverage)
```

The model controls SL distance via `action[3]`, mapped to the range `[min_sl_pct=0.1%, max_sl_pct=10%]` in `trading_env.py:150`. With `max_leverage=125`, `liquidation_buffer_pct=0.5`, `maintenance_margin_pct=0.4`:

| SL distance | denom (frac) | leverage | clamp |
|---|---|---|---|
| 0.1% | 0.001 + 0.005 + 0.004 = 0.010 | 100x | passes |
| 1.0% | 0.019 | 53x | passes |
| 5.0% | 0.059 | 17x | passes |
| 10.0% | 0.109 | 9.2x | passes |

A model that learns to use tight stops auto-leverages to ~100x. With realistic price noise, a 0.5% intra-candle move on a 100x position is a -50% account drawdown. This compounds across positions.

**Real-world reference**: Binance Futures retail caps at 50x for BTCUSDT (with margin tiers stepping down). 125x is far above any sane retail limit. 10x is what most prop shops use.

### F2 — Full-balance bets per trade

`backend/src/trainer/env/trading_env.py:178-181`:
```python
size_raw = (action[4 + 2 * num_tp] + 1.0) / 2.0   # in [0, 1]
margin = size_raw * self.account.available_balance
```

When `size_raw` is close to 1.0, the model puts ALL available balance into a single position. Combined with F1, a model with `size_raw=1.0` and `sl=0.1%` is `100x leveraged on the full account` — a 1% move wipes 100% of equity. The only risk control on size is `min_order_size_usd=10`.

### F3 — Episode runs to total wipeout

`backend/src/trainer/env/trading_env.py:103`:
```python
terminated = equity <= 0
```

The episode terminates only on full account wipeout. There's no early termination at any drawdown threshold. SB3's auto-reset means the policy gets a fresh account after each wipeout during training, but each wipeout pollutes the policy update with high-variance reward noise (account goes from $10K to $0 in a few steps).

A wipe-friendly env teaches the policy that there's no penalty for "hitting the wall" because the wall just resets. With an early-termination signal at, say, -50% drawdown, the policy receives a clearer "this strategy is uneconomic" gradient.

### F4 — Reward = Δ equity (no risk shaping) — DEFERRED

`backend/src/trainer/env/trading_env.py:90`:
```python
reward = float(equity - self._prev_equity)
```

Pure mark-to-market change. No drawdown penalty, no Sharpe shaping, no risk-adjusted return. A policy is rewarded equally for steady $100 wins as for a $1000 win followed by $900 loss-streak.

This is a real concern but addresses the train→holdout overfit gap, not blow-ups. Deferred to a follow-up audit. Recommended treatment: replace with `equity_change - λ * drawdown_increment`, where `drawdown_increment` is `max(0, peak_equity_so_far - equity)` minus its prior value. Forces the policy to value capital preservation alongside growth.

### F5 — No slippage / market impact — DEFERRED

`backend/src/trainer/env/exchange_sim.py:121-137` (_fill_order): orders fill at `order.trigger_price` exactly. Same for SL/TP fills (`_check_stop_losses`, `_check_take_profits`). On a real exchange, market orders eat the order book and limit orders may fill at slightly worse prices on volatile candles. Backtest PnL is systematically optimistic.

Deferred. The first three fixes don't depend on this; revisit after blow-ups are tamed.

### F6 — Liquidation = full margin loss — KEEP AS-IS

`backend/src/trainer/env/exchange_sim.py:174-179`: on liquidation, the account loses the position's margin entirely (`account.realize_pnl(-pos.margin)`). This is realistic — exchanges seize margin on liquidation. Combined with F1 + F2, this means one bad trade can erase the whole account. Not a bug; the upstream caps in F1 and F2 are what prevent this from compounding.

## Applied fixes (this PR)

All three are additive to `ExchangeConfig`, with new defaults that protect new sweeps without invalidating any existing saved configs (`from_dict` already filters by `__dataclass_fields__`, so old persisted JSON without these fields just falls back to defaults — but old `model_configs` rows have the OLD defaults baked into their `config_json`, so they keep their old behavior).

### Fix 1 — Lower `max_leverage` default to 10x

```python
# backend/src/trainer/config.py: ExchangeConfig
max_leverage: float = 10.0  # was 125.0
```

The clamp at the end of `compute_leverage` already does the work; we just lower the cap. A model that picks SL=0.1% now gets clamped to 10x leverage instead of 100x.

Trade-off: this also caps strategies that legitimately want higher leverage on highly-liquid pairs. Acceptable: the failure mode under 100x leverage is catastrophic, and 10x is the conventional retail-prop ceiling.

### Fix 2 — Cap per-trade position size

```python
# backend/src/trainer/config.py: ExchangeConfig
max_position_size_pct: float = 0.25  # NEW: max fraction of available_balance per trade
```

In `trading_env._process_actions`, after computing `margin`:
```python
margin = size_raw * self.account.available_balance
margin = min(margin, self.config.exchange.max_position_size_pct * self.account.available_balance)
```

A model that picks `size_raw=1.0` now gets at most 25% of available balance. With Fix 1's leverage cap of 10x, the worst-case single-trade exposure is `25% * 10x = 2.5x notional of available balance`. A 40% adverse move is the absolute worst-case loss on a single position before SL or liquidation kicks in.

### Fix 3 — Early termination on drawdown

```python
# backend/src/trainer/config.py: ExchangeConfig
max_drawdown_pct: float = 0.5  # NEW: terminate episode at this drawdown from peak equity
```

In `TradingEnv.__init__` and `reset()`:
```python
self._peak_equity = config.initial_balance
```

In `TradingEnv.step()`, after computing `equity`:
```python
self._peak_equity = max(self._peak_equity, equity)
drawdown_threshold = self._peak_equity * (1.0 - self.config.exchange.max_drawdown_pct)
terminated = (equity <= 0) or (equity <= drawdown_threshold)
```

The policy now receives a "this episode is over" signal at -50% drawdown rather than -100%. SB3 auto-resets, but the truncated trajectory provides a much sharper bootstrap target for value learning ("don't get into states like this").

Important: this is `max_drawdown_pct=0.5` on `ExchangeConfig`, not on `ModelConfig`. Threshold lives with the other env constraints. A future config could set it to e.g. 0.3 for stricter risk control or 1.0 to disable.

## Tests

Three new test files mirror the structure of existing trainer tests:

- `backend/tests/trainer/test_env_max_leverage.py`: parameterize `max_leverage` ∈ {10, 50, 125}; assert that `compute_leverage` returns `min(formula, max_leverage)`. Already verified by existing exchange_sim tests but explicitly anchor at the lowered default.
- `backend/tests/trainer/test_env_position_size_cap.py`: build a TradingEnv with `max_position_size_pct=0.25`, drive `_process_actions` with `size_raw=1.0`, assert the resulting Position's `margin` is `0.25 * initial_balance`.
- `backend/tests/trainer/test_env_drawdown_termination.py`: build a TradingEnv with `max_drawdown_pct=0.5`. Manually set `_peak_equity=10000`, then drive `step()` with a candle that produces equity ≤ 5000; assert `terminated=True`. Conversely, equity = 5001 should NOT terminate.

Existing trainer tests should continue to pass — none of them rely on `max_leverage=125` as a value (they just exercise the formula). Position-size and drawdown termination are new behaviors with new tests.

## Operational handoff

The fixes are committed to `main`. The training host is currently on `worktree-live-testing-bingx` (the user's parallel branch). When ready to re-run the seed-robustness gate under the new env:

1. Switch training host to `main` (or merge main into bingx if cross-compat needed).
2. Register a fresh sweep (e.g. `_p4d_` or `_p4e_` phase tag) — the new configs inherit the new defaults via `ExchangeConfig()`.
3. Drive sequentially via the per-process pattern from 4A/4C (the watchdog torch interop bug is unrelated and unfixed).
4. Eval, summarize, decide.

Old `_p4a_` / `_p4c_` runs are not affected — their saved `config_json` still contains `max_leverage=125` and lacks the new fields, so re-evaluating any of them runs against the old (permissive) env. We could optionally re-eval some old runs under the new env to see how they'd behave, but it's a separate analysis.

## Decision tree after fixes

After re-running 4A under the new env:
- **All three architectures still FAIL** → reward shape (F4) is the next gate. Implement risk-adjusted reward; re-run.
- **Some architecture PASSES (median > 0 AND ≥3/5 positive)** → walk-forward eval (4B) for regime robustness; if it survives walk-forward, promote.
- **Borderline pass (e.g., 3/5 positive but median small)** → ent_coef sweep (0.005, 0.05) on the passing arch to optimize.

## Out of scope

- Reward shaping (F4 deferred)
- Slippage model (F5 deferred)
- Liquidation mechanics (F6 — already realistic enough)
- Fixing the watchdog `torch.set_num_interop_threads` bug — separate concern
- Refactoring `_process_actions` for cleanliness — the changes here are minimally invasive
- Removing the now-redundant `min_order_size_usd` floor — it remains a sanity check for tiny rounding orders
