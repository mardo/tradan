from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg

from trainer.config import ModelConfig


class DataFeed:
    def __init__(
        self,
        timestamps: np.ndarray,
        features: np.ndarray,
        lookback: int = 500,
        price_columns: dict[str, int] | None = None,
    ) -> None:
        self.timestamps = timestamps
        self.raw_features = features.astype(np.float32)
        self.lookback = lookback
        self.price_columns = price_columns or {}

        self._mean = self.raw_features.mean(axis=0)
        self._std = self.raw_features.std(axis=0)
        self._std[self._std < 1e-8] = 1.0

    @property
    def total_steps(self) -> int:
        return len(self.timestamps) - self.lookback

    @property
    def num_features(self) -> int:
        return self.raw_features.shape[1]

    def get_observation(self, step: int) -> np.ndarray:
        start = step
        end = step + self.lookback
        window = self.raw_features[start:end]
        return ((window - self._mean) / self._std).astype(np.float32)

    def get_raw_observation(self, step: int) -> np.ndarray:
        start = step
        end = step + self.lookback
        return self.raw_features[start:end]

    def get_candle_prices(self, step: int) -> dict[str, float]:
        idx = step + self.lookback
        row = self.raw_features[idx] if idx < len(self.raw_features) else self.raw_features[-1]
        return {name: float(row[col]) for name, col in self.price_columns.items()}

    def get_current_raw(self, step: int) -> np.ndarray:
        idx = step + self.lookback - 1
        return self.raw_features[idx]

    def get_timestamp(self, step: int) -> int:
        return int(self.timestamps[step + self.lookback])


def load_data_feed(config: ModelConfig, conn: psycopg.Connection) -> DataFeed:
    primary_interval = config.intervals[0]

    dfs: list[pd.DataFrame] = []
    for symbol in config.symbols:
        rows = conn.execute(
            """
            SELECT open_time, {} FROM klines
            WHERE symbol = %s AND interval = %s
            ORDER BY open_time
            """.format(", ".join(config.columns)),
            (symbol, primary_interval),
        ).fetchall()

        col_names = ["open_time"] + [f"{symbol}_{c}" for c in config.columns]
        df = pd.DataFrame(rows, columns=col_names)
        df = df.set_index("open_time")
        dfs.append(df)

    if len(dfs) == 1:
        merged = dfs[0]
    else:
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.join(df, how="inner")

    merged = merged.sort_index()

    timestamps = merged.index.values.astype(np.int64)
    features = merged.values.astype(np.float32)

    primary_symbol = config.symbols[0]
    price_columns: dict[str, int] = {}
    for name in ["open", "high", "low", "close"]:
        col_name = f"{primary_symbol}_{name}"
        if col_name in merged.columns:
            price_columns[name] = list(merged.columns).index(col_name)

    return DataFeed(
        timestamps=timestamps,
        features=features,
        lookback=config.lookback_window,
        price_columns=price_columns,
    )
