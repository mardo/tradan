from trainer.config import ALL_KLINE_COLUMNS, ExchangeConfig, ModelConfig


def make_btc_config() -> ModelConfig:
    return ModelConfig(
        name="btc_v1",
        symbols=["BTCUSDT"],
        intervals=["1h"],
        columns=list(ALL_KLINE_COLUMNS),
        exchange=ExchangeConfig(),
    )
