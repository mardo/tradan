-- Top 20 completed training runs with no evaluation run yet.
-- Output: model_name<tab>run_id (for use with GNU parallel in evaluate_winners.sh)
SELECT mc.name, tr.id
FROM model_configs mc
JOIN training_runs tr ON tr.model_config_id = mc.id
    AND tr.run_type = 'train' AND tr.status = 'completed'
WHERE NOT EXISTS (
    SELECT 1 FROM training_runs ev
    WHERE ev.model_config_id = mc.id AND ev.run_type = 'evaluate'
)
ORDER BY tr.total_pnl DESC NULLS LAST
LIMIT 20;
