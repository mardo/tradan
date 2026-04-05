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
