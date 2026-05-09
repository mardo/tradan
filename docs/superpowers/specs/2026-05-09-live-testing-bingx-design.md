# Tradan — Live Testing on BingX VST (Design Spec)

**Date:** 2026-05-09
**Status:** Design approved; ready for implementation plan.
**Supersedes (refines):** `docs/plans/2026-05-09-live-testing-bingx.md`

---

## Goal

Take the promoted Phase-2 winners (per `docs/plans/2026-05-09-phase4-training-plan.md`) and run them against real-time exchange data on **BingX VST (Virtual Simulation Trading)** for a 4-week paper-test window, verifying live behavior matches backtested behavior. Architecture is exchange-agnostic via [ccxt](https://github.com/ccxt/ccxt); BingX is the first concrete adapter.

This spec is the bridge between "we have backtested winners in `model_configs`" and "we have a live trading bot." It deliberately stops short of real-money trading.

## Non-goals

- No real-money trading. **All testing happens on demo/paper-trading endpoints.**
- No new model training (covered in `2026-05-09-phase4-training-plan.md`).
- No frontend changes.
- No multi-symbol or portfolio logic (one model per bot instance, single symbol).
- No reward-shaping or online fine-tuning.
- No on-chain integration (Drift bot stays as-is).

## Success criteria

A model is "live-validated" when, over a rolling 4-week window on BingX VST:

1. **No catastrophic drawdown** — live equity never drops below 80% of starting equity.
2. **Trade frequency matches backtest** — live trades/week within ±50% of holdout-eval trade rate (e.g. `lb500_3em4_p2_s1` had 59 trades over 16 mo ≈ 0.9/week; live target 0.5–1.5/week).
3. **Sign of PnL matches sign of backtest expectation more often than not** — at least breakeven over the window. Magnitudes are not required to match.
4. **No silent failures** — every action recorded; no >1h gaps in `live_actions`; kill switch tested at least once and recovers.

A failure on (1) ends the test for that model. Failures on (2)/(3) demote to "needs more training data" without disqualifying the architecture.

## Models to promote

Per the Phase 4 plan:

| Pick | Model | Holdout result |
|---|---|---|
| 1 | `btc_4h_a2c_lb500_3em4_p2_s1` | +$126,547 / sharpe 3.08 / dd 25% / 59 trades |
| 2 | `btc_4h_a2c_lb100_3em4_p2_s0` | +$8,795 / sharpe 2.12 / dd 5.6% / 20 trades |
| 3 | `btc_4h_a2c_lb500_3em4_p2_s0` | +$3,465 / sharpe 1.77 / dd 20% / 80 trades |

All three run in parallel, each in its own VST account.

---

## Decisions locked through brainstorming

- **Parity strategy** — Extract shared pure modules from `trainer/env/` (observation builder, action decoder, fill simulator). Live and trainer literally call the same code; drift is impossible by construction.
- **Crash recovery** — Resume-with-reconciliation. On startup the runner re-attaches to any `live_runs` row with `status='running'` for the same model+exchange, fetches current orders/positions from the exchange, and writes a `reconciliation` event to `live_actions`. If the exchange has orders/positions that don't match the last logged state, the runner refuses to recover and stops with `reason='reconciliation_failed'`.
- **Deployment** — The existing `tradan` server (DO droplet, `/opt/tradan`, systemd 255, uv at `/root/.local/bin/uv`) hosts the live runners. systemd template unit `tradan-live@.service`, instances `s1`/`s2`/`s3`, `Restart=always`. Mirrors the existing `tradan-train-worker-health.service` pattern.
- **VST accounts** — Three separate BingX VST accounts. Per-pick env vars (`BINGX_VST_S1_API_KEY`, etc.). No shared credentials.
- **Kline feed** — Poll every 30s via ccxt REST `fetch_ohlcv`. Defer ccxt-pro websockets until 30s latency at candle close materially affects fills.
- **PnL table** — New `live_pnl_snapshots` table. Do not reuse the existing `pnl_snapshots` (different semantics; avoid future schema-coupling pain).
- **Replay gate** — Required before any pick's systemd unit is enabled. Run the full live code path against a `ReplayAdapter` that serves historical klines from the DB and uses the shared simulator. Target: **0% divergence** vs the trainer's stored holdout-eval terminal equity.

---

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ BingX (VST API) │◄──►│ ExchangeAdapter  │◄──►│   LiveRunner    │
│  - klines       │    │  (ccxt-backed,   │    │   (main loop)   │
│  - orders       │    │   exchange-      │    │                 │
│  - positions    │    │   agnostic API)  │    │                 │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                         │
                       ┌──────────────────┐              │
                       │ feature_pipeline │◄─────────────┤
                       │ (ccxt → trainer  │              │
                       │  pure modules)   │              │
                       └─────────┬────────┘              │
                                 │                       │
                       ┌─────────▼────────┐              │
                       │   ModelRunner    │◄─────────────┤
                       │  (SB3 .predict)  │              │
                       └─────────┬────────┘              │
                                 │                       │
                       ┌─────────▼────────┐              │
                       │  action_decoder  ├──────────────┘
                       │  (trainer pure   │
                       │   + risk clamp)  │
                       └──────────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │   PostgreSQL     │
                       │  - live_runs     │
                       │  - live_actions  │
                       │  - live_orders   │
                       │  - live_pnl_*    │
                       └──────────────────┘
```

`ExchangeAdapter` is the only piece that knows BingX exists. Adding Binance/Bybit later is a config change plus a new adapter class registration.

## Module boundaries

### `trainer/env/` — pure modules extracted from `TradingEnv`

These are imported by both `TradingEnv` (training/eval) and `live/` (production). They are pure functions: no I/O, no DB, no exchange.

- **`trainer/env/observation.py`** — `build_observation(inputs, cfg) -> dict[str, np.ndarray]`. Returns the same Dict observation that `TradingEnv.observation_space` describes: `{"market": (lookback, num_features), "account": (5,), "orders": (max_open_orders, 11), "positions": (max_open_positions, 6)}`.
- **`trainer/env/action_decoder.py`** — `decode_action(action, state, cfg) -> OrderIntent`. Translates the 51-float action vector into an `OrderIntent` dataclass (open / cancels / closes).
- **`trainer/env/exchange_sim.py`** — gains an `apply_intent(intent)` method so both `TradingEnv.step` and `ReplayAdapter` apply intents through one path.

`TradingEnv.step` is rewritten to delegate:
```python
intent = decode_action(action, self._decoder_state(), self.config.action_cfg)
self.exchange.apply_intent(intent)
obs = build_observation(self._obs_inputs(), self.config.obs_cfg)
```

External API of `TradingEnv` (`reset`, `step`, `observation_space`, `action_space`) is unchanged. Existing saved models, training scripts, and `model_configs` rows are unaffected.

### `live/` — new package

```
backend/src/live/
├── __init__.py
├── config.py              Pydantic schemas (LiveConfig & friends)
├── exchange/
│   ├── __init__.py
│   ├── base.py            ExchangeAdapter ABC + DTOs (Kline, Order, Position, Balance)
│   ├── registry.py        name → adapter class (lazy import)
│   ├── bingx.py           BingX VST via ccxt
│   └── replay.py          DB-kline-backed adapter for the replay gate
├── feature_pipeline.py    Thin: ccxt DTOs → trainer.env.observation.build_observation
├── action_decoder.py      Thin: trainer.env.action_decoder.decode_action + risk clamps
├── model_runner.py        SB3 model load + .predict()
├── runner.py              LiveRunner (state machine: starting → running → stopping)
├── reconciliation.py      Startup reconcile vs exchange
├── db.py                  Writes to live_* tables
└── cli.py                 `uv run live-test [--config|--dry-run|--status|--stop]`
```

`backend/scripts/live_replay.py` is the entry point for the replay gate (`uv run live-replay`).

## Database schema

Migration: `backend/migrations/006_live_testing_tables.sql` (next sequence after `005_model_ping.sql`).

```sql
CREATE TABLE live_runs (
    id              SERIAL PRIMARY KEY,
    model_config_id INT  NOT NULL REFERENCES model_configs(id),
    exchange        TEXT NOT NULL,                  -- 'bingx', 'binance', ...
    mode            TEXT NOT NULL,                  -- 'demo' | 'paper' | 'live'
    symbol          TEXT NOT NULL,                  -- 'BTC/USDT:USDT'
    interval        TEXT NOT NULL,                  -- '4h'
    starting_equity NUMERIC NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    stopped_at      TIMESTAMPTZ,
    stop_reason     TEXT,                           -- 'manual'|'kill_switch'|'drawdown'|'error'|'crashed'|'reconciliation_failed'
    status          TEXT NOT NULL DEFAULT 'running',-- 'running' | 'stopped'
    kill_requested  BOOLEAN NOT NULL DEFAULT FALSE, -- set by `live-test --stop`
    config_yaml     TEXT NOT NULL,                  -- snapshot of YAML used (forensics)
    git_sha         TEXT NOT NULL                   -- which build started this run
);

CREATE UNIQUE INDEX live_runs_one_running_per_model
    ON live_runs(model_config_id, exchange) WHERE status = 'running';

CREATE TABLE live_actions (
    id              SERIAL PRIMARY KEY,
    live_run_id     INT  NOT NULL REFERENCES live_runs(id),
    event_type      TEXT NOT NULL DEFAULT 'inference', -- 'inference'|'reconciliation'|'kill_switch'|'error'
    candle_close    TIMESTAMPTZ,                       -- null for non-inference events
    raw_action      JSONB,                             -- 51-float vector; null for non-inference
    decoded_intent  JSONB,                             -- {open,cancel,close}; null for non-inference
    account_state   JSONB NOT NULL,                    -- snapshot at the moment of the event
    inference_ms    INT,
    notes           TEXT,                              -- reconciliation diff, error message, etc.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE live_orders (
    id                SERIAL PRIMARY KEY,
    live_run_id       INT  NOT NULL REFERENCES live_runs(id),
    live_action_id    INT  REFERENCES live_actions(id),
    exchange_order_id TEXT NOT NULL,
    side              TEXT NOT NULL,                  -- 'buy' | 'sell'
    type              TEXT NOT NULL,                  -- 'limit'|'market'|'stop'|'take_profit'
    price             NUMERIC,
    amount            NUMERIC NOT NULL,
    status            TEXT NOT NULL,                  -- 'open'|'filled'|'cancelled'|'rejected'
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

Schema notes:

- `live_runs_one_running_per_model` enforces the no-double-runner invariant at the DB level. Resume-on-restart works because the new process attaches to the existing row instead of inserting.
- `config_yaml` and `git_sha` snapshot the inputs that produced the run, so forensic analysis weeks later isn't blocked by config drift.
- `event_type` on `live_actions` lets reconciliation, kill, and error events live in the same timeline as inference, ordered by `created_at`.

## LiveRunner state machine

Three states: **starting**, **running**, **stopping**.

### `starting`

```
attached_run = SELECT * FROM live_runs
               WHERE model_config_id=? AND exchange=? AND status='running'

if attached_run is None:
    INSERT new live_runs row
else:
    exchange_state = adapter.fetch_balance() + fetch_positions() + fetch_open_orders()
    last_logged    = SELECT account_state FROM live_actions
                     WHERE live_run_id = attached_run.id
                     ORDER BY created_at DESC LIMIT 1
    diff = compare(exchange_state, last_logged)

    if exchange has orders/positions NOT in last_logged:
        # Refuse: someone or something else touched the account.
        log error live_action; UPDATE live_runs SET status='stopped',
            stop_reason='reconciliation_failed'
        exit 2

    INSERT live_actions(event_type='reconciliation', account_state=exchange_state, notes=diff)
```

### `running` — loop body, every 30s

```
1. if kill_requested OR env TRADAN_KILL_SWITCH_<S>=true → stopping
2. fetch latest kline for symbol/interval
3. if newest_candle.close_time > last_processed_candle_close:
     a. fetch lookback klines (lookback + buffer)
     b. fetch account, positions, open_orders
     c. obs = build_observation(...)              # trainer.env.observation
     d. action, ms = model_runner.predict(obs)
     e. INSERT live_action(event_type='inference', raw_action, account_state, inference_ms=ms)
     f. intent = decode_action(action, state)     # trainer.env.action_decoder
     g. clamp intent against risk config (max_position_size_pct, max_leverage)
     h. UPDATE live_action.decoded_intent = clamped_intent
     i. for each cancel: adapter.cancel_order → INSERT live_order
        for each close: adapter.close_position → INSERT live_order
        for open: adapter.place_order(..., sl, tp) → INSERT live_order(s)
     j. last_processed_candle_close = newest_candle.close_time
4. if elapsed since last pnl snapshot ≥ pnl_snapshot_interval_minutes:
     INSERT live_pnl_snapshots
     if equity < starting_equity * (1 - max_drawdown_pct) → stopping(reason='drawdown')
5. on 3 consecutive adapter exceptions → stopping(reason='error')
```

### `stopping` — single graceful shutdown path

Same code runs for: drawdown, kill-switch (env or DB), SIGINT/SIGTERM, 3-consecutive-error.

```
1. for each open order: adapter.cancel_order (best-effort; log failures)
2. for each open position: adapter.close_position(fraction=1.0)
3. final live_pnl_snapshots row
4. UPDATE live_runs SET status='stopped', stopped_at=now(), stop_reason=…
exit 0
```

Risk clamping happens after model inference, before placement. The raw action is logged so divergences can be analyzed; the clamped intent is what was placed.

## Replay gate (correctness checkpoint)

`scripts/live_replay.py` runs `LiveRunner` with a `ReplayAdapter` (DB-kline-backed, shared-simulator-backed) over a historical window. Compares terminal equity to the trainer's stored holdout-eval result.

```bash
uv run live-replay \
  --model btc_4h_a2c_lb500_3em4_p2_s1 \
  --start 2024-12-01 --end 2026-04-30 \
  --tolerance-pct 0.5
```

Because `ExchangeSim` and `ReplayAdapter` use the **same** simulator module, the divergence target is **0%**. The `--tolerance-pct` flag is paranoia, not the spec — any non-zero divergence is a bug to find and fix.

The replay gate is a hard prerequisite: a pick's systemd unit is not enabled until its replay run passes.

## CLI

```bash
# Start (or resume) a run
uv run live-test --config configs/live/live-s1.yaml

# Dry run — no orders placed; every other step (incl. DB writes) happens
uv run live-test --config configs/live/live-s1.yaml --dry-run

# Inspect active runs
uv run live-test --status
# → table: run_id, model, exchange, mode, started_at, current_equity, dd, last_action_age

# Request graceful stop (sets live_runs.kill_requested=true; runner picks it up within 30s)
uv run live-test --stop <run_id>

# Replay gate
uv run live-replay --model <name> --start <date> --end <date> [--tolerance-pct <pct>]
```

Both `live-test` and `live-replay` are registered as console scripts in `backend/pyproject.toml`.

## Configuration

Per-pick YAML in `backend/configs/live/`:

```yaml
# configs/live/live-s1.yaml
exchange:
  name: bingx
  mode: demo
  api_key_env: BINGX_VST_S1_API_KEY
  api_secret_env: BINGX_VST_S1_API_SECRET

market:
  symbol: BTC/USDT:USDT          # ccxt unified symbol for BTCUSDT perp
  interval: 4h

model:
  name: btc_4h_a2c_lb500_3em4_p2_s1   # path resolved via model_configs row

risk:
  starting_equity_quote: 10000   # match backtest starting capital
  max_drawdown_pct: 0.20         # kill switch
  max_position_size_pct: 0.50    # cap per-order size as fraction of equity
  max_leverage: 3
  kill_switch_env: TRADAN_KILL_SWITCH_S1   # per-pick to avoid global blast radius

logging:
  pnl_snapshot_interval_minutes: 60
```

`live-s2.yaml` and `live-s3.yaml` are the same shape with the model name and env-var names swapped (`_S2`, `_S3`).

## Deployment (systemd)

Mirrors the existing `tradan-train-worker-health.service` pattern.

**`infra/systemd/tradan-live@.service`** — single template, instance per pick (`s1`, `s2`, `s3`):

```ini
[Unit]
Description=Tradan live runner (%i)
After=network-online.target

[Service]
Type=simple
Environment=PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EnvironmentFile=/etc/tradan/live-%i.env
WorkingDirectory=/opt/tradan/backend
ExecStart=/root/.local/bin/uv run live-test --config configs/live/live-%i.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enabled per-pick: `systemctl enable --now tradan-live@s1.service`. Three units, three env files (`/etc/tradan/live-s1.env`, etc.), zero shared state.

**`infra/scripts/live_runner_deploy.sh`** copies the unit file + env templates and runs `systemctl daemon-reload`.

Secrets: `/etc/tradan/live-sN.env`, mode 600, owned by root, provisioned manually by the operator. Each contains `BINGX_VST_S<N>_API_KEY`, `BINGX_VST_S<N>_API_SECRET`, `TRADAN_KILL_SWITCH_S<N>=false`, and the shared `DATABASE_URL`. No secret manager for v1; revisit when going to mainnet.

## File map

| File | Purpose |
|---|---|
| `backend/migrations/006_live_testing_tables.sql` | live_runs, live_actions, live_orders, live_pnl_snapshots |
| `backend/src/trainer/env/observation.py` | **NEW** extracted pure observation builder |
| `backend/src/trainer/env/action_decoder.py` | **NEW** extracted pure action decoder |
| `backend/src/trainer/env/exchange_sim.py` | gains `apply_intent()` (shared with ReplayAdapter) |
| `backend/src/trainer/env/trading_env.py` | refactored to delegate to the above |
| `backend/src/live/__init__.py` | package marker |
| `backend/src/live/config.py` | Pydantic config schemas |
| `backend/src/live/exchange/base.py` | abstract `ExchangeAdapter` + DTOs |
| `backend/src/live/exchange/registry.py` | name → adapter |
| `backend/src/live/exchange/bingx.py` | BingX VST via ccxt |
| `backend/src/live/exchange/replay.py` | DB-kline-backed adapter for replay gate |
| `backend/src/live/feature_pipeline.py` | ccxt DTOs → trainer.env.observation |
| `backend/src/live/action_decoder.py` | trainer.env.action_decoder + risk clamps |
| `backend/src/live/model_runner.py` | SB3 load + predict |
| `backend/src/live/runner.py` | LiveRunner main loop |
| `backend/src/live/reconciliation.py` | startup reconcile vs exchange |
| `backend/src/live/db.py` | live_* writes |
| `backend/src/live/cli.py` | `uv run live-test` |
| `backend/scripts/live_replay.py` | `uv run live-replay` (replay gate) |
| `backend/configs/live/live-s1.yaml` | Pick 1 — `lb500_3em4_p2_s1` |
| `backend/configs/live/live-s2.yaml` | Pick 2 — `lb100_3em4_p2_s0` |
| `backend/configs/live/live-s3.yaml` | Pick 3 — `lb500_3em4_p2_s0` |
| `infra/systemd/tradan-live@.service` | systemd template, instance per pick |
| `infra/scripts/live_runner_deploy.sh` | deploy helper |

Modified:
- `backend/pyproject.toml` — add `ccxt`, `pyyaml`; verify `pydantic` already present; register `live-test` and `live-replay` console scripts.

## Implementation phases

Strict ordering: each phase is a checkpoint; do not start the next until the prior is green.

### Phase A — Trainer/env extraction (no `live/` code yet)
1. Investigate `trainer/env/data_feed.py` to determine if normalization is stateless or requires a fitted scaler. Document findings; if a scaler exists, design how it is saved alongside the model.
2. Extract `observation.py`, `action_decoder.py`, and add `apply_intent()` to `exchange_sim.py`.
3. Refactor `TradingEnv.step` to delegate.
4. **Gate A1:** Run holdout-eval on Pick 1, Pick 2, Pick 3, and one P3 model before and after the refactor. Assert PnL series numerically identical (≤1e-9 per-step) and final equity exactly equal. If any model diverges, fix before proceeding.
5. Existing trainer unit tests (if any) all pass.

### Phase B — Live infrastructure scaffolding (no exchange yet)
6. Pydantic config schemas (`live/config.py`).
7. DB migration (`006_live_testing_tables.sql`); apply on dev DB; verify in `\dt`.
8. `ExchangeAdapter` ABC + DTOs.
9. `db.py` write helpers with transactional unit tests.
10. **Gate B1:** Config loads from a known-good YAML; migration applies cleanly; DB writes are reversible.

### Phase C — Replay adapter and gate
11. `ReplayAdapter` — uses the shared simulator with klines from the DB. Same `ExchangeAdapter` interface.
12. `feature_pipeline.py` and `action_decoder.py` (live thin wrappers).
13. `model_runner.py` — load + predict.
14. `runner.py` — main loop without reconciliation yet (not needed for replay).
15. `live_replay.py` script.
16. **Gate C1:** Replay gate passes for Pick 1 with **0% divergence**. Repeat for Picks 2 and 3.

### Phase D — BingX adapter (read-only first)
17. `BingXAdapter`: `fetch_klines`, `fetch_balance`, `fetch_positions`, `fetch_open_orders`. No order placement yet.
18. **Gate D1:** With VST credentials, fetch BTC/USDT:USDT 4h klines and account state from BingX VST; print last 5 candles and current balance.

### Phase E — BingX adapter (writes)
19. Implement `place_order` (limit, market, stop-market, take-profit-market), `cancel_order`, `close_position`. Set leverage.
20. **Gate E1:** Smoke tests against VST: place tiny limit far below market → cancel; place market with attached SL+TP → see all three at exchange → close position. All recorded in `live_orders`.

### Phase F — Reconciliation + full LiveRunner
21. `reconciliation.py` with the strict "unknown exchange state → refuse" rule.
22. Wire reconciliation into `runner.py` startup.
23. Add `--dry-run`, `--status`, `--stop` to `cli.py`.
24. **Gate F1:** Pick 1 in `--dry-run` mode against BingX VST for 24h. Verify: `live_actions` rows accumulate, decoded intents look sensible, `--status` shows the run, `--stop` stops it cleanly.

### Phase G — Systemd + first live (Pick 1)
25. systemd template + deploy script.
26. `/etc/tradan/live-s1.env` provisioned manually with VST account 1's credentials.
27. **Gate G1:** Enable `tradan-live@s1.service` with `mode: demo` and not `--dry-run`. Run for 24h. Verify: orders on VST dashboard match `live_orders`; PnL snapshots accumulating; resume-on-restart works (kill the process, watch systemd respawn it, confirm reconciliation event).
28. **Gate G2 (kill-switch test):** Set `TRADAN_KILL_SWITCH_S1=true` in env file and reload service. Verify: positions closed, orders cancelled, run stopped within 60s, `stop_reason='kill_switch'`.

### Phase H — Picks 2 & 3 + 4-week test
29. Pick 2 systemd unit + env file. Run Phase G gates again.
30. Pick 3 systemd unit + env file. Run Phase G gates again.
31. Begin the 4-week observation period. Define a weekly check runbook step: query `live_runs`, `live_pnl_snapshots`, `live_actions` for each pick and verify the success criteria (drawdown, trade frequency, no >1h action gaps).

### Phase I — Optional monitoring (deferred)
32. FastAPI endpoints `/api/live/runs`, `/api/live/runs/{id}/pnl`, `/api/live/runs/{id}/actions`. Frontend "Live" tab. Ship only if `psql` queries become friction during the 4-week period.

## Testing approach

- **Unit tests:** observation builder, action decoder, risk clamping, reconciliation diff logic, DB write helpers.
- **Parity tests:** the before/after-refactor eval comparison (Gate A1) stays in the repo as a regression test against a small frozen kline window.
- **Integration tests:** the replay gate (Gate C1) — strongest correctness check.
- **Manual smoke tests:** the BingX adapter VST gates (D1, E1).
- **End-to-end:** Phase G dry-run and live runs.

## Operational concerns

### Time alignment
4h candles close at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC. The 30s polling loop detects new candles within at most 30s of close.

### Symbol mapping
BingX swap: ccxt unified `BTC/USDT:USDT` ↔ BingX REST `BTC-USDT`. The adapter owns the mapping.

### Fee parity
Trainer env uses Binance futures fees (maker 0.02%, taker 0.04%). BingX VST fees may differ. Document the divergence; live fees a few bps off backtest expectation is acceptable for paper testing but must be revisited before mainnet.

### Price source divergence
Trainer used Binance klines; live reads BingX prices. BingX VST mirrors real BTC prices but liquidity and fast-move price ticks differ. This is unavoidable for testing on BingX and is part of what the 4-week observation window measures.

### Slippage
VST has limited slippage modeling. Treat live PnL as best-case until real-money pilot.

### Stopping conditions (auto)
- Drawdown > config threshold → stop (`reason='drawdown'`)
- 3 consecutive errors from the exchange → stop (`reason='error'`)
- No new candle for 2× expected interval → stop and alert (`reason='error'`)
- `TRADAN_KILL_SWITCH_<S>=true` or `live_runs.kill_requested=true` → stop (`reason='kill_switch'`)

### Rollback
A pick that breaches drawdown self-stops with `reason='drawdown'` and `exit 0`; systemd's `Restart=always` does not bring it back (process exited cleanly). To resume after drawdown:
1. Investigate logs and `live_actions`.
2. If software bug: fix, redeploy, restart with same `live_runs.id` (resume-on-restart).
3. If model issue: rotate to a different pick; do not restart the failing model on the same VST account.

Emergency: `systemctl stop tradan-live@s1 tradan-live@s2 tradan-live@s3` — each runner enters the same graceful `stopping` state via SIGTERM and flattens its account.

## Open items deferred to implementation

These are unknowns that must be resolved during the implementation plan but don't change the design:

- Whether `model_configs` needs new columns to store normalization scaler state (depends on Phase A.1 finding).
- Whether ccxt's `bingx` exchange uses `options.test=True` or a separate VST endpoint.
- Per-pick `inference_ms` baseline to flag latency regressions (derive from the first week of data).

## Phase 5+ (out of scope for this spec)

- Add Binance VST adapter — verify behavior consistent across exchanges.
- Real-money small-position pilot — same code, `mode: live`, position size capped at $50 USD.
- Multi-symbol — train BTC, evaluate on ETH/SOL.
- On-chain leg — bring `bot/strategy.ts` Drift integration online with the same model service.
