import pytest

from trainer.config import ExchangeConfig
from trainer.env.account import Account
from trainer.env.exchange_sim import ExchangeSim, Order, Position


@pytest.fixture
def exchange() -> ExchangeSim:
    return ExchangeSim(
        config=ExchangeConfig(
            maker_fee_pct=0.02,
            taker_fee_pct=0.04,
            flat_fee_usd=0.0,
            max_leverage=125.0,
            liquidation_buffer_pct=0.5,
            maintenance_margin_pct=0.4,
            max_open_orders=5,
            max_open_positions=5,
            min_order_size_usd=10.0,
        ),
        account=Account(initial_balance=10_000.0),
    )


class TestLeverageCalculation:
    def test_long_leverage_from_sl(self, exchange: ExchangeSim):
        leverage = exchange.compute_leverage(
            entry_price=50_000.0, sl_price=49_000.0, direction=1
        )
        assert leverage > 0
        liq = exchange.liquidation_price(50_000.0, leverage, 1)
        assert liq < 49_000.0

    def test_short_leverage_from_sl(self, exchange: ExchangeSim):
        leverage = exchange.compute_leverage(
            entry_price=50_000.0, sl_price=51_000.0, direction=-1
        )
        assert leverage > 0
        liq = exchange.liquidation_price(50_000.0, leverage, -1)
        assert liq > 51_000.0

    def test_leverage_clamped_to_max(self, exchange: ExchangeSim):
        leverage = exchange.compute_leverage(
            entry_price=50_000.0, sl_price=10_000.0, direction=1
        )
        assert leverage <= exchange.config.max_leverage

    def test_tight_sl_allows_higher_leverage(self, exchange: ExchangeSim):
        wide = exchange.compute_leverage(50_000.0, 45_000.0, 1)
        tight = exchange.compute_leverage(50_000.0, 49_500.0, 1)
        assert tight > wide


class TestPlaceOrder:
    def test_place_long_order(self, exchange: ExchangeSim):
        order = exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[51_000.0, 52_000.0], tp_size_pcts=[0.5, 0.5], margin=1_000.0,
        )
        assert order is not None
        assert order.direction == 1
        assert len(exchange.open_orders) == 1
        assert exchange.account.margin_used == 1_000.0

    def test_place_order_rejected_no_margin(self, exchange: ExchangeSim):
        order = exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[51_000.0], tp_size_pcts=[1.0], margin=20_000.0,
        )
        assert order is None

    def test_place_order_rejected_max_orders(self, exchange: ExchangeSim):
        for i in range(5):
            exchange.place_order(
                direction=1, trigger_price=50_000.0 + i, sl_price=49_000.0,
                tp_prices=[51_000.0], tp_size_pcts=[1.0], margin=100.0,
            )
        order = exchange.place_order(
            direction=1, trigger_price=50_010.0, sl_price=49_000.0,
            tp_prices=[51_000.0], tp_size_pcts=[1.0], margin=100.0,
        )
        assert order is None


class TestOrderFill:
    def test_long_order_fills_when_price_drops_to_trigger(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=49_000.0, sl_price=48_000.0,
            tp_prices=[50_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        fills = exchange.process_candle(high=50_000.0, low=48_500.0, close=49_500.0)
        assert len(exchange.open_orders) == 0
        assert len(exchange.open_positions) == 1
        pos = exchange.open_positions[0]
        assert pos.direction == 1
        assert pos.entry_price == 49_000.0

    def test_short_order_fills_when_price_rises_to_trigger(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=-1, trigger_price=51_000.0, sl_price=52_000.0,
            tp_prices=[50_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        fills = exchange.process_candle(high=51_500.0, low=50_000.0, close=50_500.0)
        assert len(exchange.open_positions) == 1
        assert exchange.open_positions[0].direction == -1


class TestStopLoss:
    def test_long_sl_triggers(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        assert len(exchange.open_positions) == 1
        exchange.process_candle(high=50_000.0, low=48_500.0, close=48_800.0)
        assert len(exchange.open_positions) == 0


class TestTakeProfit:
    def test_partial_tp_triggers(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[51_000.0, 53_000.0], tp_size_pcts=[0.5, 0.5], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        pos = exchange.open_positions[0]
        original_size = pos.size
        exchange.process_candle(high=51_500.0, low=50_500.0, close=51_200.0)
        assert len(exchange.open_positions) == 1
        assert exchange.open_positions[0].size == pytest.approx(original_size * 0.5, rel=1e-6)

    def test_all_tps_close_position(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[51_000.0, 52_000.0], tp_size_pcts=[0.5, 0.5], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        exchange.process_candle(high=53_000.0, low=50_500.0, close=52_500.0)
        assert len(exchange.open_positions) == 0


class TestManualClose:
    def test_close_position_fully(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        assert len(exchange.open_positions) == 1
        exchange.close_position(0, fraction=1.0, current_price=50_200.0)
        assert len(exchange.open_positions) == 0

    def test_close_position_partially(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        original_size = exchange.open_positions[0].size
        exchange.close_position(0, fraction=0.3, current_price=50_200.0)
        assert len(exchange.open_positions) == 1
        assert exchange.open_positions[0].size == pytest.approx(original_size * 0.7, rel=1e-6)


class TestCancelOrder:
    def test_cancel_releases_margin(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        assert exchange.account.margin_used == 1_000.0
        exchange.cancel_order(0)
        assert len(exchange.open_orders) == 0
        assert exchange.account.margin_used == 0.0


class TestFees:
    def test_maker_fee_on_order_fill(self, exchange: ExchangeSim):
        initial_balance = exchange.account.balance
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        assert exchange.account.balance < initial_balance


class TestUnrealizedPnl:
    def test_total_unrealized_pnl(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_900.0, close=50_200.0)
        pnl = exchange.total_unrealized_pnl(current_price=50_500.0)
        assert pnl > 0

    def test_short_unrealized_pnl(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=-1, trigger_price=50_000.0, sl_price=51_000.0,
            tp_prices=[49_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.process_candle(high=50_500.0, low=49_500.0, close=49_800.0)
        pnl = exchange.total_unrealized_pnl(current_price=49_500.0)
        assert pnl > 0


class TestReset:
    def test_reset_clears_state(self, exchange: ExchangeSim):
        exchange.place_order(
            direction=1, trigger_price=50_000.0, sl_price=49_000.0,
            tp_prices=[52_000.0], tp_size_pcts=[1.0], margin=1_000.0,
        )
        exchange.reset()
        assert len(exchange.open_orders) == 0
        assert len(exchange.open_positions) == 0
        assert exchange.account.balance == 10_000.0
