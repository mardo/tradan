from __future__ import annotations

from dataclasses import dataclass, field

from trainer.config import ExchangeConfig
from trainer.env.account import Account


@dataclass
class Order:
    id: int
    direction: int
    trigger_price: float
    sl_price: float
    tp_prices: list[float]
    tp_size_pcts: list[float]
    margin: float


@dataclass
class Position:
    id: int
    direction: int
    entry_price: float
    size: float
    leverage: float
    sl_price: float
    tp_prices: list[float]
    tp_size_pcts: list[float]
    margin: float
    liquidation_price: float

    def unrealized_pnl(self, current_price: float) -> float:
        return self.direction * (current_price - self.entry_price) * self.size


@dataclass
class FillEvent:
    event_type: str
    position_id: int | None = None
    order_id: int | None = None
    pnl: float = 0.0
    fee: float = 0.0


class ExchangeSim:
    def __init__(self, config: ExchangeConfig, account: Account) -> None:
        self.config = config
        self.account = account
        self.open_orders: list[Order] = []
        self.open_positions: list[Position] = []
        self._next_order_id = 0
        self._next_position_id = 0
        self.total_trades = 0
        self.winning_trades = 0

    def reset(self) -> None:
        self.open_orders.clear()
        self.open_positions.clear()
        self._next_order_id = 0
        self._next_position_id = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.account.reset()

    def compute_leverage(
        self, entry_price: float, sl_price: float, direction: int
    ) -> float:
        sl_dist_pct = abs(entry_price - sl_price) / entry_price
        if sl_dist_pct == 0:
            return 1.0
        mm = self.config.maintenance_margin_pct / 100.0
        buf = self.config.liquidation_buffer_pct / 100.0
        denominator = sl_dist_pct + buf + mm
        leverage = 1.0 / denominator
        return min(leverage, self.config.max_leverage)

    def liquidation_price(
        self, entry_price: float, leverage: float, direction: int
    ) -> float:
        mm = self.config.maintenance_margin_pct / 100.0
        if direction == 1:
            return entry_price * (1.0 - (1.0 / leverage) + mm)
        else:
            return entry_price * (1.0 + (1.0 / leverage) - mm)

    def _compute_fee(self, notional: float, fee_type: str) -> float:
        if fee_type == "maker":
            pct = self.config.maker_fee_pct / 100.0
        else:
            pct = self.config.taker_fee_pct / 100.0
        return notional * pct + self.config.flat_fee_usd

    def place_order(
        self, direction: int, trigger_price: float, sl_price: float,
        tp_prices: list[float], tp_size_pcts: list[float], margin: float,
    ) -> Order | None:
        if len(self.open_orders) >= self.config.max_open_orders:
            return None
        if margin > self.account.available_balance:
            return None
        leverage = self.compute_leverage(trigger_price, sl_price, direction)
        notional = margin * leverage
        if notional < self.config.min_order_size_usd:
            return None
        self.account.allocate_margin(margin)
        order = Order(
            id=self._next_order_id, direction=direction,
            trigger_price=trigger_price, sl_price=sl_price,
            tp_prices=list(tp_prices), tp_size_pcts=list(tp_size_pcts), margin=margin,
        )
        self._next_order_id += 1
        self.open_orders.append(order)
        return order

    def cancel_order(self, index: int) -> None:
        if 0 <= index < len(self.open_orders):
            order = self.open_orders.pop(index)
            self.account.release_margin(order.margin)

    def _fill_order(self, order: Order) -> Position:
        leverage = self.compute_leverage(order.trigger_price, order.sl_price, order.direction)
        notional = order.margin * leverage
        size = notional / order.trigger_price
        liq_price = self.liquidation_price(order.trigger_price, leverage, order.direction)
        fee = self._compute_fee(notional, "maker")
        self.account.apply_fee(fee)
        position = Position(
            id=self._next_position_id, direction=order.direction,
            entry_price=order.trigger_price, size=size, leverage=leverage,
            sl_price=order.sl_price, tp_prices=list(order.tp_prices),
            tp_size_pcts=list(order.tp_size_pcts), margin=order.margin,
            liquidation_price=liq_price,
        )
        self._next_position_id += 1
        self.open_positions.append(position)
        return position

    def close_position(self, index: int, fraction: float, current_price: float) -> float:
        if not (0 <= index < len(self.open_positions)):
            return 0.0
        pos = self.open_positions[index]
        fraction = min(max(fraction, 0.0), 1.0)
        close_size = pos.size * fraction
        pnl = pos.direction * (current_price - pos.entry_price) * close_size
        notional = close_size * current_price
        fee = self._compute_fee(notional, "taker")
        margin_released = pos.margin * fraction
        self.account.release_margin(margin_released)
        self.account.realize_pnl(pnl)
        self.account.apply_fee(fee)
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        if fraction >= 1.0 - 1e-9:
            self.open_positions.pop(index)
        else:
            pos.size -= close_size
            pos.margin -= margin_released
        return pnl

    def _check_liquidations(self, high: float, low: float) -> list[FillEvent]:
        events: list[FillEvent] = []
        to_remove: list[int] = []
        for i, pos in enumerate(self.open_positions):
            liquidated = False
            if pos.direction == 1 and low <= pos.liquidation_price:
                liquidated = True
            elif pos.direction == -1 and high >= pos.liquidation_price:
                liquidated = True
            if liquidated:
                notional = pos.size * pos.liquidation_price
                fee = self._compute_fee(notional, "taker")
                self.account.apply_fee(fee)
                self.account.release_margin(pos.margin)
                self.account.realize_pnl(-pos.margin)
                self.total_trades += 1
                to_remove.append(i)
                events.append(FillEvent("liquidation", position_id=pos.id, pnl=-pos.margin, fee=fee))
        for i in reversed(to_remove):
            self.open_positions.pop(i)
        return events

    def _check_stop_losses(self, high: float, low: float) -> list[FillEvent]:
        events: list[FillEvent] = []
        to_remove: list[int] = []
        for i, pos in enumerate(self.open_positions):
            hit = False
            if pos.direction == 1 and low <= pos.sl_price:
                hit = True
            elif pos.direction == -1 and high >= pos.sl_price:
                hit = True
            if hit:
                pnl = pos.direction * (pos.sl_price - pos.entry_price) * pos.size
                notional = pos.size * pos.sl_price
                fee = self._compute_fee(notional, "taker")
                self.account.release_margin(pos.margin)
                self.account.realize_pnl(pnl)
                self.account.apply_fee(fee)
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                to_remove.append(i)
                events.append(FillEvent("sl", position_id=pos.id, pnl=pnl, fee=fee))
        for i in reversed(to_remove):
            self.open_positions.pop(i)
        return events

    def _check_take_profits(self, high: float, low: float) -> list[FillEvent]:
        events: list[FillEvent] = []
        to_remove: list[int] = []
        for i, pos in enumerate(self.open_positions):
            tps_hit: list[int] = []
            for j, tp_price in enumerate(pos.tp_prices):
                if pos.direction == 1 and high >= tp_price:
                    tps_hit.append(j)
                elif pos.direction == -1 and low <= tp_price:
                    tps_hit.append(j)
            for j in sorted(tps_hit, reverse=True):
                tp_price = pos.tp_prices[j]
                tp_frac = pos.tp_size_pcts[j]
                close_size = pos.size * tp_frac
                pnl = pos.direction * (tp_price - pos.entry_price) * close_size
                notional = close_size * tp_price
                fee = self._compute_fee(notional, "maker")
                margin_released = pos.margin * tp_frac
                self.account.release_margin(margin_released)
                self.account.realize_pnl(pnl)
                self.account.apply_fee(fee)
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                pos.size -= close_size
                pos.margin -= margin_released
                pos.tp_prices.pop(j)
                pos.tp_size_pcts.pop(j)
                if pos.tp_size_pcts:
                    total = sum(pos.tp_size_pcts)
                    if total > 0:
                        pos.tp_size_pcts = [p / total for p in pos.tp_size_pcts]
                events.append(FillEvent("tp", position_id=pos.id, pnl=pnl, fee=fee))
            if pos.size < 1e-12 or not pos.tp_prices:
                if pos.margin > 0:
                    self.account.release_margin(pos.margin)
                to_remove.append(i)
        for i in reversed(to_remove):
            self.open_positions.pop(i)
        return events

    def _check_order_fills(self, high: float, low: float) -> list[FillEvent]:
        events: list[FillEvent] = []
        to_remove: list[int] = []
        for i, order in enumerate(self.open_orders):
            filled = False
            if order.direction == 1 and low <= order.trigger_price:
                filled = True
            elif order.direction == -1 and high >= order.trigger_price:
                filled = True
            if filled and len(self.open_positions) < self.config.max_open_positions:
                pos = self._fill_order(order)
                to_remove.append(i)
                events.append(FillEvent("fill", position_id=pos.id, order_id=order.id))
        for i in reversed(to_remove):
            self.open_orders.pop(i)
        return events

    def process_candle(self, high: float, low: float, close: float) -> list[FillEvent]:
        events: list[FillEvent] = []
        events.extend(self._check_liquidations(high, low))
        events.extend(self._check_stop_losses(high, low))
        events.extend(self._check_take_profits(high, low))
        events.extend(self._check_order_fills(high, low))
        return events

    def total_unrealized_pnl(self, current_price: float) -> float:
        return sum(p.unrealized_pnl(current_price) for p in self.open_positions)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
