-- Kline candlestick data from Binance USDT-M futures
CREATE TABLE klines (
    symbol               TEXT    NOT NULL,
    interval             TEXT    NOT NULL,
    open_time            BIGINT  NOT NULL,  -- ms epoch (open of candle)
    open                 NUMERIC NOT NULL,
    high                 NUMERIC NOT NULL,
    low                  NUMERIC NOT NULL,
    close                NUMERIC NOT NULL,
    volume               NUMERIC NOT NULL,  -- base asset volume
    close_time           BIGINT  NOT NULL,  -- ms epoch (close of candle)
    quote_volume         NUMERIC NOT NULL,
    num_trades           INTEGER NOT NULL,
    taker_buy_base_vol   NUMERIC NOT NULL,
    taker_buy_quote_vol  NUMERIC NOT NULL,
    PRIMARY KEY (symbol, interval, open_time)
);

CREATE INDEX klines_symbol_interval_idx ON klines (symbol, interval);

-- Job queue for parallel ingestion workers
CREATE TABLE ingest_jobs (
    id           SERIAL      PRIMARY KEY,
    symbol       TEXT        NOT NULL,
    interval     TEXT        NOT NULL,
    year         SMALLINT    NOT NULL,
    month        SMALLINT    NOT NULL,
    status       TEXT        NOT NULL DEFAULT 'pending',  -- pending/running/done/failed
    claimed_at   TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error        TEXT,
    UNIQUE (symbol, interval, year, month)
);

CREATE INDEX ingest_jobs_pending_idx ON ingest_jobs (status)
    WHERE status IN ('pending', 'running');
