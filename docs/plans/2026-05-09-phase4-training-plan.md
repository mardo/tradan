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

**Outcome (executed 2026-05-09):**

Implementation: see `docs/plans/2026-05-09-phase4a-seed-robustness.md`. Key changes:
- Added `ModelConfig.seed: int | None`; previously the `SEEDS = [42, 123, 456]` lists in P1/P2/P3 sweeps were declared but never passed to SB3 — variance came from non-deterministic torch/numpy init alone.
- Forwarded the seed to the SB3 algorithm (`seed=config.seed`) in `train_model`.
- Registered 15 configs `btc_4h_a2c_lb{100,250,500}_3em4_p4a_s{0..4}` with seeds `[1001, 2002, 3003, 4004, 5005]`. Each seed value is reused across the three lookbacks (so `_s0` always = 1001), giving paired-design control of seed luck across architectures.

Decision matrix (`scripts/phase4a_summary.py`):

| arch  | seeds | pos | median PnL  | P2 median   | decision |
|-------|-------|-----|-------------|-------------|----------|
| lb100 | 5     | 0   | -$2,378     | -$7,881     | **FAIL** |
| lb250 | 5     | 1   | -$10,510    | $0          | **FAIL** |
| lb500 | 5     | 0   | -$10,155    | +$3,465     | **FAIL** |

Per-seed holdout detail:

| name                          | pnl     | sharpe | dd %  | trades | seed |
|-------------------------------|--------:|-------:|------:|-------:|-----:|
| lb100_p4a_s0                  |    -596 |  -2.05 |   6.0 |      2 | 1001 |
| lb100_p4a_s1                  |  -4,851 |  -2.63 |  48.5 |     45 | 2002 |
| lb100_p4a_s2                  | -28,365 |  -6.51 | 283.6 |      5 | 3003 |
| lb100_p4a_s3                  |    -217 |  -1.49 |   2.2 |      1 | 4004 |
| lb100_p4a_s4                  |  -2,378 |  -1.46 |  25.0 |      5 | 5005 |
| lb250_p4a_s0                  |       0 |   0.00 |   0.0 |      0 | 1001 |
| lb250_p4a_s1                  | -12,116 |  -3.33 | 111.0 |    100 | 2002 |
| lb250_p4a_s2                  |  +2,782 |   1.01 |  15.5 |     14 | 3003 |
| lb250_p4a_s3                  | -11,567 |  -3.72 | 115.0 |     43 | 4004 |
| lb250_p4a_s4                  | -10,510 |  -3.15 | 105.1 |      7 | 5005 |
| lb500_p4a_s0                  |  -5,195 |  -0.87 |  71.7 |     54 | 1001 |
| lb500_p4a_s1                  | -10,155 |  -3.53 | 101.6 |     33 | 2002 |
| lb500_p4a_s2                  | -10,316 |  -3.88 | 102.7 |     10 | 3003 |
| lb500_p4a_s3                  | -11,703 |  -3.47 | 117.0 |      6 | 4004 |
| lb500_p4a_s4                  |  -7,428 |   0.23 |  93.7 |    117 | 5005 |

**Verdict:** all three architectures fail the pass criterion. Only 1 of 15 holdout evals (lb250_p4a_s2, +$2,782) is profitable. Drawdowns above 100% indicate full account blow-ups under leverage.

Training-time PnL (where some seeds had +$22K to +$135K) does not survive holdout. The original P2 winners (`lb500_3em4_p2_s1` +$126K, `lb100_3em4_p2_s0` +$8.8K, `lb500_3em4_p2_s0` +$3.5K) were almost certainly seed-luck artifacts of unseeded init — under explicit seeded training, the same architectures produce uniformly losing or do-nothing policies on holdout.

**Implication for promotion plan:** the original "Pick 1 + Pick 2" promotion list is invalidated. **No model from these three architectures should be paper-traded based on Phase 1–3 results alone.** A successful 4A pass — or stronger evidence from 4B walk-forward — is required before committing live capital, even on devnet test tokens.

Operational notes for future sweeps:
- The systemd watchdog (`tradan-train-worker-health.service`) repeatedly restarts `train worker`, but the worker's in-process loop has a `torch.set_num_interop_threads` constraint that fails on the 2nd model in a single process. The watchdog respawn pattern accidentally works around this (one model per process), but only at the cost of 120-min stale-claim waits when many configs claim simultaneously. For 4A I bypassed the watchdog with a sequential `train start` driver script (`/tmp/p4a_drive.sh`); future sweeps should either fix the trainer (catch the RuntimeError or guard via flag) or use the same per-process driver.

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

**4C Outcome — A2C entropy regularization (executed 2026-05-10):**

Implementation: see `docs/plans/2026-05-09-phase4c-entropy-design.md`. Added `ModelConfig.ent_coef` (default 0.0, conditionally forwarded to SB3 only when > 0 to preserve SAC's 'auto' default) and a sweep + summary script that reuses `phase4a_summary.evaluate_arch` so the pass/fail rule stays in one place.

Scope: 5 seeds × `lb500 + lr=3e-4 + ent_coef=0.01`, 2M timesteps, paired with 4A's seed values (1001..5005) for direct per-seed comparison.

Decision matrix:

| arch          | seeds | pos | median PnL  | 4A median   | decision |
|---------------|------:|----:|------------:|------------:|----------|
| lb500_ent01   | 5     | 1   | -$2,442     | -$10,155    | **FAIL** |

Per-seed paired delta (`4C – 4A`, both on the same seed):

| seed | 4A holdout PnL | 4C holdout PnL | delta     |
|-----:|---------------:|---------------:|----------:|
| 1001 |       -$5,195 |        -$3,770 |   +$1,425 |
| 2002 |      -$10,155 |        -$2,442 |   +$7,713 |
| 3003 |      -$10,316 |          +$553 |  +$10,869 |
| 4004 |      -$11,703 |        -$5,134 |   +$6,569 |
| 5005 |       -$7,428 |          -$485 |   +$6,943 |
| —    | —              | —              |           |
| median delta |       |                |   +$6,943 |
| seeds where 4C beat 4A |  5 of 5 |   |            |

**Verdict: FAIL the gate, but the entropy hypothesis is partially confirmed.**

The architecture still doesn't pass the seed-robustness gate (need median > 0 AND ≥3/5 positive; got median -$2,442 with 1/5 positive). However, every paired seed improved with entropy regularization — median delta +$6,943 — and one seed (3003) crossed into positive territory. The improvement is systematic, not noise.

Two readings:
1. **Entropy reg is helping, just not enough.** An `ent_coef` sweep at higher values (e.g. 0.05) or longer training (5M timesteps with `ent_coef=0.01`) might push the architecture over the line.
2. **Entropy reg is necessary but not sufficient.** Even with consistent improvement, the holdout-positive bar is high. The remaining gap may be due to the env permitting >40% drawdowns (s0/s1/s3 all blew through 49–66% on holdout); an env/reward audit is the bigger lever.

Failure-mode shift: 4A's lb500 produced active-but-losing policies (10–117 trades, mostly catastrophic drawdowns). 4C produced two distinct families: high-trade chaotic ones (s0: 791 trades, s3: 119 trades) and low-trade conservative ones (s4: 2 trades, s2: 38 trades). The conservative s2/s4 had the smallest drawdowns (4.8–8.8%) — the only seed to break even (s2 +$553) was in this family. Suggests entropy reg can produce both directions of behavior depending on seed.

Operational note: env code drift between branches caused an apples-to-oranges hazard during this experiment. The training host had been switched to the user's `worktree-live-testing-bingx` branch (which has env-internal refactors not on `main`) between training and eval. The first eval pass on bingx-code produced visibly different trade counts (e.g. s1 holdout: 219 trades on bingx vs 29 on main with the same model). All numbers above are from the re-evaluation on `main`, which is the commit set under which training happened. Future cross-phase comparisons need a single env-code version pinned across train+eval.

**Recommended next step:** env/reward audit (master plan's Path C). Two algorithmic levers (seeds, entropy reg) have now been pulled with diminishing returns; the >40% drawdowns on losing seeds suggest the env's leverage and stop-loss handling permits failure modes a real exchange wouldn't. A small `ent_coef` sweep (0.05) is a cheaper sanity check first if compute is free, but the env audit is where the next material gain likely comes from.

### 4D — Interval expansion
**Goal:** establish whether 4h is genuinely best or just easiest.

- Re-test 1h and 1d at the proven `lb500 + lr=3e-4` setting (Phase 1 used defaults).
- 3 seeds each × 1M timesteps.
- Wall-clock: ~3h (1h is similar load; 1d is faster due to fewer episodes).

### 4E — Idle-step penalty (env audit follow-up)
**Goal:** address the F4 finding from the env audit (reward = Δ equity has no risk shaping) with the simplest possible slice — penalize idle steps to discourage policy collapse to "do nothing". This is Idea 1 from a three-idea menu (also: trade-completion bonus, Sharpe-shaped reward); Ideas 2 and 3 are deferred.

- For each of the three architectures (lb100/250/500) at A2C, lr=3e-4, 1M timesteps under the env audit caps (max_leverage=10, max_position_size_pct=0.25, max_drawdown_pct=0.5), run 5 paired seeds at each of two penalty magnitudes:
  - 0.05 USD/step (~5 bps/step at $10K notional) — light pressure
  - 0.5  USD/step (~50 bps/step) — strong pressure
- Total: 3 archs × 2 values × 5 seeds = 30 runs at 1M timesteps.
- **Success criterion** (per cell): median holdout PnL > 0 AND ≥3 of 5 seeds positive.

**4E Outcome (executed 2026-05-11):**

Implementation: see `docs/plans/2026-05-10-phase4-env-audit-design.md` for the parent F4 finding. Added `ExchangeConfig.idle_step_penalty_usd` (default 0.0, additive penalty subtracted from the Δ-equity reward when both `open_positions` and `open_orders` are empty). Default 0 preserves all pre-4E configs bit-for-bit.

Decision matrix (`scripts/phase4e_idle_penalty_summary.py`):

| arch × penalty | seeds | pos | median PnL  | 4D median  | decision |
|----------------|------:|----:|------------:|-----------:|----------|
| lb100 × 0.05   |   5   |  0  |    -$1,939  |     -$213  | **FAIL** |
| lb100 × 0.5    |   5   |  1  |    -$3,803  |     -$213  | **FAIL** |
| lb250 × 0.05   |   5   |  2  |    -$3,235  |     -$139  | **FAIL** |
| lb250 × 0.5    |   5   |  0  |    -$5,033  |     -$139  | **FAIL** |
| lb500 × 0.05   |   5   |  0  |    -$4,572  |   -$5,012  | **FAIL** |
| lb500 × 0.5    |   5   |  0  |    -$3,543  |   -$5,012  | **FAIL** |

Paired delta per cell (4E – 4D, same env caps + same seeds, only `idle_step_penalty_usd` differs):

| arch × penalty | 4E > 4D | median Δ  | direction |
|----------------|--------:|----------:|-----------|
| lb100 × 0.05   |  2/5    |  -$2,391  | **worse** |
| lb100 × 0.5    |  2/5    |  -$2,080  | **worse** |
| lb250 × 0.05   |  2/5    |  -$4,022  | **worse** |
| lb250 × 0.5    |  1/5    |  -$4,953  | **worse** |
| lb500 × 0.05   |  3/5    |     +$2   | neutral   |
| lb500 × 0.5    |  3/5    |    +$380  | ~neutral  |

Per-seed holdout detail (4E):

| name                          |    pnl | sharpe | dd %  | trades | seed |
|-------------------------------|-------:|-------:|------:|-------:|-----:|
| lb100_idle05_p4e_s0           | -1,584 |   0.57 |  50.7 |     14 | 1001 |
| lb100_idle05_p4e_s1           | -1,939 |  -0.75 |  30.4 |     15 | 2002 |
| lb100_idle05_p4e_s2           | -3,339 |  -2.86 |  37.2 |     46 | 3003 |
| lb100_idle05_p4e_s3           | -5,033 | -16.72 |  50.3 |     16 | 4004 |
| lb100_idle05_p4e_s4           | -1,047 |  -0.27 |  24.1 |    160 | 5005 |
| lb100_idle5_p4e_s0            | -3,803 |  -2.64 |  45.6 |     49 | 1001 |
| lb100_idle5_p4e_s1            | -5,086 |  -3.73 |  50.9 |     82 | 2002 |
| lb100_idle5_p4e_s2            | -5,053 |  -6.35 |  51.6 |     19 | 3003 |
| lb100_idle5_p4e_s3            | -2,293 |  -0.71 |  40.3 |     51 | 4004 |
| lb100_idle5_p4e_s4            |   +553 |   0.52 |  21.7 |     56 | 5005 |
| lb250_idle05_p4e_s0           |    +52 |   0.24 |  29.1 |     65 | 1001 |
| lb250_idle05_p4e_s1           | -4,556 |  -2.41 |  50.3 |     17 | 2002 |
| lb250_idle05_p4e_s2           | +1,262 |   0.99 |  13.9 |    115 | 3003 |
| lb250_idle05_p4e_s3           | -3,235 |  -1.27 |  42.2 |     76 | 4004 |
| lb250_idle05_p4e_s4           | -5,190 | -16.77 |  52.4 |     14 | 5005 |
| lb250_idle5_p4e_s0            | -4,312 |  -2.69 |  45.6 |    187 | 1001 |
| lb250_idle5_p4e_s1            | -5,092 |  -3.92 |  52.4 |     29 | 2002 |
| lb250_idle5_p4e_s2            | -3,555 |  -1.61 |  51.8 |     42 | 3003 |
| lb250_idle5_p4e_s3            | -5,033 |  -7.67 |  50.3 |     22 | 4004 |
| lb250_idle5_p4e_s4            | -5,065 | -13.56 |  52.3 |     19 | 5005 |
| lb500_idle05_p4e_s0           | -4,594 |  -2.11 |  50.4 |    275 | 1001 |
| lb500_idle05_p4e_s1           | -5,075 |  -5.76 |  50.7 |     55 | 2002 |
| lb500_idle05_p4e_s2           | -4,572 |  -3.16 |  51.1 |    320 | 3003 |
| lb500_idle05_p4e_s3           |   -905 |  -1.52 |  12.6 |     26 | 4004 |
| lb500_idle05_p4e_s4           | -3,848 |  -3.32 |  38.5 |     69 | 5005 |
| lb500_idle5_p4e_s0            | -5,301 |  -5.80 |  53.0 |     42 | 1001 |
| lb500_idle5_p4e_s1            | -3,543 |  -3.78 |  36.7 |     80 | 2002 |
| lb500_idle5_p4e_s2            | -5,060 |  -4.55 |  50.6 |     46 | 3003 |
| lb500_idle5_p4e_s3            | -1,508 |  -1.09 |  19.1 |     91 | 4004 |
| lb500_idle5_p4e_s4            |   -751 |  -0.78 |  16.7 |    216 | 5005 |

For context, the 4D baseline (same archs, same seeds, no idle penalty — env audit caps only) had its evals run as part of this phase since 4D's trains existed but were never evaluated; full 4D detail:

| name                          |    pnl | sharpe | dd %  | trades | seed |
|-------------------------------|-------:|-------:|------:|-------:|-----:|
| lb100_p4d_s0                  |   +807 |   1.09 |   6.9 |     46 | 1001 |
| lb100_p4d_s1                  | -5,399 |-100.52 |  54.0 |     13 | 2002 |
| lb100_p4d_s2                  |    +55 |   0.16 |   4.4 |     41 | 3003 |
| lb100_p4d_s3                  |   -213 |  -1.15 |   2.2 |      4 | 4004 |
| lb100_p4d_s4                  | -2,003 |  -1.03 |  30.2 |    115 | 5005 |
| lb250_p4d_s0                  | -4,946 |  -9.45 |  51.2 |     36 | 1001 |
| lb250_p4d_s1                  |   -139 |  -0.17 |   8.0 |     14 | 2002 |
| lb250_p4d_s2                  | -1,992 |  -1.21 |  37.3 |     97 | 3003 |
| lb250_p4d_s3                  |   +788 |   0.59 |  14.7 |     72 | 4004 |
| lb250_p4d_s4                  |    -55 |  -0.15 |   3.8 |      3 | 5005 |
| lb500_p4d_s0                  | -2,889 |  -1.58 |  37.5 |    167 | 1001 |
| lb500_p4d_s1                  | -5,077 |  -4.24 |  50.8 |     36 | 2002 |
| lb500_p4d_s2                  | -5,439 | -11.33 |  55.0 |     18 | 3003 |
| lb500_p4d_s3                  |   +399 |   0.46 |  40.0 |    357 | 4004 |
| lb500_p4d_s4                  | -5,012 |  -3.49 |  50.1 |     23 | 5005 |

4D aggregate: 5 of 15 seeds positive (33%), median per arch lb100/lb250/lb500 = -$213/-$139/-$5,012 — all FAIL the gate as well. 4D is the natural baseline for 4E but was itself never a winning configuration; 4E was a follow-up reward-shaping attempt on top.

**Verdict: all six 4E cells fail the pass criterion. The idle-step penalty hypothesis is rejected.**

Direction of effect:
- **lb100 and lb250**: penalty makes things significantly worse (median Δ -$2K to -$5K vs 4D). Forcing the policy to trade when it would rather hold creates additional losses, not gains. The "do nothing" failure mode that motivated this experiment was a 4A symptom under the OLD permissive env (max_leverage=125, no caps); under the 4D env caps the policy already trades, so penalizing idleness no longer addresses the actual remaining failure mode (which is uneconomic trading, not insufficient trading).
- **lb500**: penalty is roughly neutral (median Δ ~0 to +$380). lb500 already trades actively at baseline (4D trade counts 18-357 vs lb100's 4-115); there is no "do nothing" collapse to penalize, so adding the penalty neither helps nor materially hurts.

Aggregate positive-seed counts:
- 4D: 5 of 15 seeds positive (33%, max +$807)
- 4E: 3 of 30 seeds positive (10%, max +$1,262)

The penalty cut positive-seed rate by ~3×. Even where individual 4E seeds outperformed (e.g. lb250_idle05_s2 at +$1,262), the cell still failed because only 2/5 seeds crossed zero.

**Implication for plan:** Idea 1 (idle-step penalty) is rejected; do not pursue further magnitudes or hybrid combinations with this lever alone. The remaining ideas from the experiment design are independent of the idle-penalty result:

1. **Trade-completion bonus** (Idea 2) — bonus on closed round-trips, possibly gated on net positive PnL. Selects FOR profitable activity rather than penalizing absence. Cheap to add (same shape as 4E).
2. **Sharpe-shaped reward** (Idea 3 / F4 proper) — `reward = Δ equity − λ × drawdown_increment`, the design F4 recommended directly. Bigger refactor.
3. **Higher ent_coef sweep** — 4C confirmed entropy reg systematically improves paired PnL but with diminishing magnitude at 0.01; 0.05 was suggested as a sanity check and remains untried.

If compute is constrained, Idea 2 is the lightest experiment with the most direct hypothesis (target the bad-trading failure mode, not the inactive failure mode). If Idea 2 also fails to clear the gate, that's strong signal that reward shaping alone won't fix this — the issue is closer to the data/architecture choice (4h-A2C-MLP), and Phase 4B walk-forward or a deeper algorithm/architecture change is the right next move.

Operational notes:
- 4D trains existed in DB but had never been evaluated; the 15-eval baseline pass ran first (~1 min), then the 30 4E evals (~2 min). All 45 evals completed within 3 min.
- Driver script `/tmp/p4e_drive.sh` (sequential per-process train) ran cleanly through all 30 configs in ~8h wall-clock without watchdog conflict; the watchdog services were already inactive at sweep start.
- The env-code-drift hazard from the 4C operational notes (train and eval must run on the same branch) was avoided by leaving the host on `main` for both train and eval in this phase.

### 4F — Live inference bridge
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

- [x] Confirm the 3 models to promote — **none**. 4A failed all three architectures; the prior "Pick 1 + Pick 2" list was seed-luck and is rejected. Promotion is blocked until either (a) an architecture passes a re-run of 4A (e.g. with entropy regularization per 4C) or (b) walk-forward (4B) provides multi-window evidence that the original P2 winners are robust across regimes.
- [x] Phase 4A or 4E first? — **4A done first**, as recommended. Outcome above. 4E should not be started until at least one architecture passes the seed-robustness gate.
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
