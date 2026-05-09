# Tradan — Phase 4 Training Plan & State Snapshot

**Date:** 2026-05-09
**Status:** Phase 1-3 complete. Phase 2 produced strong holdout-eval winners; Phase 3 (5M timesteps) over-trained and degraded. Ready to promote 2-3 models to paper testing and start Phase 4.

---

## Current state (what we know)

### Pipeline
- **Phase 1**: Baseline grid sweep — algos × intervals × seeds with default HPs.
- **Phase 2**: Around the top P1 architecture, expand HPs (lookback × learning_rate × seeds) at 1M timesteps.
- **Phase 3**: Top P2 winners → long-train (5M timesteps × 3 seeds).

### Data
- Source: BTC/USDT klines (Binance), table `klines`.
- 4h interval: 13,692 candles, 2020-01-01 → ~2026-04-30 (~6.25 years).
- Split: **80% train / 20% holdout, chronological** (train: 2020-01 → 2024-12; holdout: ~2024-12 → 2026-04, ~16 months unseen).

### Phase 1 outcome
- Coverage: intervals {1m, 5m, 15m, 30m, 1h, 4h, 1d} × algos {PPO, A2C, SAC} × 3 seeds.
- Survivor: **`btc_4h_a2c`** only. One seed (s1) made $3,474 eval / sharpe 1.79 / 234 trades. All other intervals/algos either blew up (>99% drawdown) or collapsed to "do-nothing" policies (0 trades).
- Implication: 4h-A2C was the only branch with enough signal to be worth expanding; do **not** assume the others are broken — they were under-explored at default HPs.

### Phase 2 outcome
Sweep around 4h-A2C: lookback ∈ {100, 250, 500, 1000} × lr ∈ {1e-4, 3e-4, 1e-3} × 3 seeds × 1M timesteps.

Top by holdout sharpe:

| Model | Eval PnL | Sharpe | DD | Trades |
|---|---|---|---|---|
| `btc_4h_a2c_lb500_3em4_p2_s1` | **$126,547** | 3.08 | 0.25 | 59 |
| `btc_4h_a2c_lb100_3em4_p2_s0` | $8,795 | 2.12 | 0.06 | 20 |
| `btc_4h_a2c_lb250_3em4_p2_s0` | $39,168 | 2.07 | 0.43 | 77 |
| `btc_4h_a2c_lb500_3em4_p2_s0` | $3,465 | 1.77 | 0.20 | 80 |
| `btc_4h_a2c_lb500_3em4_p2_s2` | -$9,987 | -5.54 | 1.00 | 24 |

Key findings:
- **`lr=3e-4` dominates winners.** 1e-4 is too slow, 1e-3 too noisy.
- **lb500 produced the home-run; lb100 produced a low-drawdown small winner.**
- **Same arch, different seed → wild range** (s1: +$126K, s2: -$10K). Single-seed results are unreliable.

### Phase 3 outcome
Top P2 architecture (`lb500 + lr=3e-4`) trained 5× longer (5M timesteps), 3 seeds.

| Model | Eval PnL | Sharpe | DD | Trades |
|---|---|---|---|---|
| `..._p3_s0` | $0 | 0.00 | 0.00 | 0 |
| `..._p3_s1` | -$1,398 | -1.05 | 0.18 | 10 |
| `..._p3_s2` | -$10,492 | -7.55 | 1.05 | 0 |

Conclusion: **5M timesteps strictly degraded performance**. Two seeds collapsed (do-nothing / total blowup), one traded unprofitably. Classic A2C failure mode — entropy decays toward zero, policy becomes deterministic and over-fits training trajectory. The sweet spot for this arch is at or below 1M timesteps.

---

## Models to promote (paper testing with test tokens)

Rank order = priority. All three are 4h A2C, lr=3e-4, Phase 2.

### Pick 1 — `btc_4h_a2c_lb500_3em4_p2_s1`
- Holdout: **+$126,547 / sharpe 3.08 / dd 25% / 59 trades**
- Strongest performer in the sweep
- Risk: sister seed (s2) blew up on same arch — winner may be partly lucky on this holdout window

### Pick 2 — `btc_4h_a2c_lb100_3em4_p2_s0`
- Holdout: +$8,795 / sharpe 2.12 / **dd 5.6%** / 20 trades
- Shorter lookback → different signal timescale; very low drawdown suggests cautious policy
- Diversification: failure modes uncorrelated with Pick 1

### Pick 3 (optional) — `btc_4h_a2c_lb500_3em4_p2_s0`
- Holdout: +$3,465 / sharpe 1.77 / dd 20% / 80 trades
- Same architecture as Pick 1, different seed
- Purpose: if both #1 and #3 work live, the architecture is real (not just s1 luck)

**Skip**: any P3 model, any seed-2 model, `lb250_3em4_p2_s0` (43% dd is too aggressive for a confidence-builder).

---

## Known limitations / risks

1. **High seed variance.** Same architecture goes from -$10K to +$126K depending on seed. Need 5+ seeds to characterize true distribution.
2. **Single holdout window.** One sample per model = prone to "lucky regime" overfit. No bull/bear/chop diversity.
3. **Idealized execution.** Env uses Binance fee schedule but no slippage / spread / market-impact. Live performance will be weaker than backtest.
4. **A2C over-trains.** Confirmed via Phase 3 — past ~1M timesteps, policies collapse. Need entropy regularization or different algorithm for longer horizons.
5. **Coverage is thin.** Only 4h-A2C lineage explored deeply. PPO/SAC, other intervals are essentially untested at proper HPs.
6. **No live-inference bridge.** `bot/strategy.ts` has placeholder `basicStrategy`; loading a saved SB3 model and producing live actions is unimplemented.

---

## Phase 4 plan

Five sub-phases, ordered by recommended priority. 4A and 4E are highest value; 4B unlocks credible promotions.

### 4A — Seed robustness (immediate, ~5h wall-clock)
**Goal:** characterize seed variance; reject architectures whose median seed is unprofitable.

- For each top-3 P2 architecture (lb100, lb250, lb500 — all with lr=3e-4), run **5 additional seeds** at 1M timesteps.
- Total: 15 runs × ~19 min each ≈ 5h.
- **Success criterion**: median holdout PnL > 0 AND ≥3 of 5 seeds positive.
- Architectures that fail this should not be promoted — period.

### 4B — Walk-forward eval (highest robustness ROI)
**Goal:** replace the single 16-month holdout with 6+ rolling out-of-sample windows.

- Define rolling windows: e.g. train 4 yr / eval 6 mo, slide forward 6 mo at a time.
- For each promotion candidate, eval on every window.
- Output: distribution of (PnL, sharpe, dd) across regimes.
- **Filter**: a model must be profitable in ≥70% of windows to be promotable.
- New code: `backend/scripts/walk_forward_eval.py`. Existing `evaluator.py` already does single-window eval; reuse its TradingEnv with custom holdout slices.

### 4C — Algorithm diversification (PPO + entropy)
**Goal:** find an algorithm that is stable at longer horizons (so we can use more compute).

- PPO at the proven `lb500 + lr=3e-4` architecture:
  - 3 seeds at each of {1M, 2M, 5M} timesteps = 9 runs (~10h wall-clock since PPO is slower per step than A2C).
  - Hypothesis: PPO's clipped objective prevents the entropy collapse that killed A2C P3.
- A2C with explicit entropy regularization:
  - 3 seeds at lb500 + lr=3e-4 + ent_coef=0.01 (default is 0.0), 2M timesteps.
  - If this rescues longer training, it's strong evidence the failure was entropy collapse.

### 4D — Interval expansion
**Goal:** establish whether 4h is genuinely best or just easiest.

- Re-test 1h and 1d at the proven `lb500 + lr=3e-4` setting (Phase 1 used defaults).
- 3 seeds each × 1M timesteps.
- Wall-clock: ~3h (1h is similar load; 1d is faster due to fewer episodes).

### 4E — Live inference bridge
**Goal:** make a saved model actually decide trades on live data.

Architecture:
1. **`backend/scripts/serve_model.py`** — small FastAPI/Flask service. POST /predict with last N candles + account state → returns the 51-float action vector. Loads the SB3 model once at startup.
2. **`bot/strategy.ts` — `realStrategy()`** — fetch live klines → POST to `/predict` → decode action vector → translate "open order" / "cancel order" / "close position" intents into Drift orders via the existing `drift-client.ts`.
3. **Action decoding** — replicate the env's action interpretation (see `backend/src/trainer/env/`) so live behavior matches simulation.
4. **Logging table** — `live_actions(timestamp, model_id, raw_action, decoded_intent, executed_order_id, fill_price)` for later forensic comparison vs backtest.

Constraints:
- **Devnet test tokens for ≥4 weeks before mainnet.** Track live PnL vs predicted; large divergence triggers stop.
- **Kill switch.** Use existing `KILL_SWITCH` env in `bot/config.ts`.
- **One model per bot instance** initially (no ensembling). Three Drift accounts → three bot instances → three models in parallel.

---

## Phase 5+ (not committed)
- **Execution realism**: slippage model based on order size vs candle volume; bid-ask spread; partial fills.
- **Multi-symbol transfer**: train on BTC, evaluate on ETH/SOL with no fine-tuning. Tests whether learned policies generalize or just memorize BTC.
- **Online fine-tuning**: keep model weights updating from live data once paper-trading is stable.
- **Architecture upgrade**: transformer policy over OHLCV sequence vs current MLP. Likely captures longer-range structure that lookback-MLPs cannot.
- **Risk-aware reward shaping**: current reward is Δ equity; consider Sharpe-like or downside-penalized rewards.

---

## Open decisions

- [ ] Confirm the 3 models to promote (default: Pick 1 + Pick 2; Pick 3 optional)
- [ ] Phase 4A or 4E first?
  - 4A first = cheap robustness data before committing engineering effort.
  - 4E first = start collecting live data while Phase 4A trains in parallel.
  - Recommendation: **4A and 4E in parallel** — 4A is unattended GPU/CPU work; 4E is human engineering.
- [ ] Walk-forward window size for 4B — 3, 6, or 12 month eval blocks?
- [ ] Devnet → mainnet promotion criteria — minimum live duration, max acceptable drift between live PnL and backtest expectation?

---

## Quick reference — useful queries

```sql
-- Top holdout-eval performers, all phases
SELECT mc.name, tr.total_pnl, tr.sharpe_ratio, tr.max_drawdown, tr.total_trades
FROM training_runs tr
JOIN model_configs mc ON mc.id = tr.model_config_id
WHERE tr.run_type = 'evaluate' AND tr.status = 'completed'
ORDER BY tr.sharpe_ratio DESC NULLS LAST
LIMIT 20;

-- Train-runs missing eval
SELECT mc.name, tr.id
FROM training_runs tr
JOIN model_configs mc ON mc.id = tr.model_config_id
WHERE tr.run_type = 'train' AND tr.status = 'completed'
  AND NOT EXISTS (
    SELECT 1 FROM training_runs ev
    WHERE ev.model_config_id = mc.id AND ev.run_type = 'evaluate'
  )
ORDER BY tr.total_pnl DESC NULLS LAST;
```
