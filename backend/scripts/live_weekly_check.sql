-- live_weekly_check.sql — Weekly observation runbook for active live runs.
--
-- Run weekly during the 4-week BingX VST observation window:
--     psql "$DATABASE_URL" -f backend/scripts/live_weekly_check.sql
--
-- Success criteria (from docs/superpowers/specs/2026-05-09-live-testing-bingx-design.md):
--   1. No catastrophic drawdown: dd_pct < 0.20 for all active runs.
--      A pick that breaches this is failed and pulled from the test.
--   2. Trade frequency within ±50% of holdout-eval rate:
--        Pick 1 (lb500_s1):  ~0.9 trades/week  (target band 0.45–1.35)
--        Pick 2 (lb100_s0):  ~0.3 trades/week  (target band 0.15–0.45)
--        Pick 3 (lb500_s0):  ~1.2 trades/week  (target band 0.60–1.80)
--      Out-of-band: model is "needs more training data" but architecture is not disqualified.
--   3. Sign of PnL matches sign of backtest expectation more often than not — at least breakeven over the window.
--   4. No silent failures: zero gaps > 1h between consecutive inference events.

\echo
\echo === Active runs ===
SELECT
    lr.id,
    mc.name,
    lr.exchange,
    lr.starting_equity,
    COALESCE(
        (SELECT equity FROM live_pnl_snapshots
         WHERE live_run_id = lr.id
         ORDER BY taken_at DESC LIMIT 1),
        lr.starting_equity
    ) AS current_equity,
    (SELECT count(*) FROM live_orders
     WHERE live_run_id = lr.id AND status = 'filled') AS filled_count,
    lr.started_at,
    now() - lr.started_at AS up_for
FROM live_runs lr
JOIN model_configs mc ON mc.id = lr.model_config_id
WHERE lr.status = 'running'
ORDER BY lr.started_at;

\echo
\echo === Action-log gaps (>1 hour between consecutive inference events) — should be EMPTY ===
WITH ordered AS (
    SELECT live_run_id,
           candle_close,
           lag(candle_close) OVER (PARTITION BY live_run_id ORDER BY candle_close) AS prev_close
    FROM live_actions
    WHERE event_type = 'inference'
)
SELECT live_run_id,
       prev_close   AS gap_start,
       candle_close AS gap_end,
       candle_close - prev_close AS gap
FROM ordered
WHERE candle_close - prev_close > interval '1 hour'
ORDER BY gap DESC
LIMIT 20;

\echo
\echo === Drawdown from start (criterion #1: dd_pct must stay < 0.20) ===
SELECT lr.id,
       mc.name,
       lr.starting_equity AS start_eq,
       (SELECT min(equity) FROM live_pnl_snapshots WHERE live_run_id = lr.id) AS min_eq,
       1.0 - (SELECT min(equity) FROM live_pnl_snapshots WHERE live_run_id = lr.id)
              / NULLIF(lr.starting_equity, 0) AS dd_pct
FROM live_runs lr
JOIN model_configs mc ON mc.id = lr.model_config_id
WHERE lr.status = 'running';

\echo
\echo === Trades per week (criterion #2: must be within ±50% of holdout target) ===
SELECT lr.id,
       mc.name,
       (SELECT count(*) FROM live_orders WHERE live_run_id = lr.id AND status = 'filled') AS total_filled,
       extract(epoch from (now() - lr.started_at)) / 86400.0 / 7.0 AS weeks_up,
       (SELECT count(*) FROM live_orders WHERE live_run_id = lr.id AND status = 'filled')
         / NULLIF(extract(epoch from (now() - lr.started_at)) / 86400.0 / 7.0, 0) AS trades_per_week
FROM live_runs lr
JOIN model_configs mc ON mc.id = lr.model_config_id
WHERE lr.status = 'running';

\echo
\echo === Recent reconciliation / error events (last 14 days) — investigate any rows ===
SELECT la.live_run_id,
       mc.name AS model,
       la.event_type,
       la.created_at,
       la.notes
FROM live_actions la
JOIN live_runs lr ON lr.id = la.live_run_id
JOIN model_configs mc ON mc.id = lr.model_config_id
WHERE la.event_type IN ('reconciliation', 'error', 'kill_switch')
  AND la.created_at > now() - interval '14 days'
ORDER BY la.created_at DESC
LIMIT 50;

\echo
\echo === Stopped runs in the last 14 days (note stop_reason) ===
SELECT lr.id, mc.name, lr.exchange, lr.started_at, lr.stopped_at, lr.stop_reason
FROM live_runs lr
JOIN model_configs mc ON mc.id = lr.model_config_id
WHERE lr.status = 'stopped'
  AND lr.stopped_at > now() - interval '14 days'
ORDER BY lr.stopped_at DESC;
