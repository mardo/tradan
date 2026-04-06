-- Ranked winner table: apply filter chain then sort by holdout Sharpe.
-- Run from the base droplet: psql $DATABASE_URL -f winners.sql
SELECT
    mc.name,
    tr_train.total_pnl                                          AS train_pnl,
    tr_eval.total_pnl                                           AS holdout_pnl,
    ROUND(tr_eval.sharpe_ratio::numeric, 3)                     AS sharpe,
    ROUND((tr_eval.max_drawdown * 100)::numeric, 1)             AS drawdown_pct,
    tr_eval.total_trades,
    ROUND((tr_eval.win_rate * 100)::numeric, 1)                 AS win_rate_pct,
    ROUND(
        (tr_eval.total_pnl / NULLIF(tr_train.total_pnl, 0))::numeric, 2
    )                                                           AS generalization_ratio
FROM model_configs mc
JOIN training_runs tr_train ON tr_train.model_config_id = mc.id
    AND tr_train.run_type = 'train' AND tr_train.status = 'completed'
JOIN training_runs tr_eval ON tr_eval.model_config_id = mc.id
    AND tr_eval.run_type = 'evaluate' AND tr_eval.status = 'completed'
WHERE tr_eval.total_trades > 10
  AND tr_eval.total_pnl > 0
  AND tr_eval.max_drawdown < 0.25
  AND tr_eval.total_pnl / NULLIF(tr_train.total_pnl, 0) > 0.5
ORDER BY tr_eval.sharpe_ratio DESC
LIMIT 20;
