-- Fix klines rows where open_time/close_time were stored in microseconds
-- instead of milliseconds (Binance changed CSV precision for 2025+ data).
-- Any open_time above 9999999999999 ms is beyond year 2286 and must be in
-- microseconds - divide both timestamps by 1000 to restore correct values.

-- Step 1: insert corrected rows (divide both timestamps by 1000).
-- ON CONFLICT DO NOTHING handles the edge case where a correct-ms row
-- already exists from a different ingest path.
INSERT INTO klines (
    symbol, interval, open_time, open, high, low, close, volume,
    close_time, quote_volume, num_trades, taker_buy_base_vol, taker_buy_quote_vol
)
SELECT
    symbol, interval, open_time / 1000, open, high, low, close, volume,
    close_time / 1000, quote_volume, num_trades, taker_buy_base_vol, taker_buy_quote_vol
FROM klines
WHERE open_time > 9999999999999
ON CONFLICT (symbol, interval, open_time) DO NOTHING;

-- Step 2: remove the now-duplicated microsecond rows.
DELETE FROM klines WHERE open_time > 9999999999999;
