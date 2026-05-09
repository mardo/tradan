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
