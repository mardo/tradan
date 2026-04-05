from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig


def make_sol_config() -> ModelConfig:
    return ModelConfig(
        name="sol_v1",
        symbols=["BTCUSDT", "SOLUSDT"],
        intervals=["1h"],
        columns=list(ALL_KLINE_COLUMNS),
        exchange=ExchangeConfig(),
    )
