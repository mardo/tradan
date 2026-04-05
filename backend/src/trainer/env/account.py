from __future__ import annotations


class Account:
    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.margin_used = 0.0

    @property
    def available_balance(self) -> float:
        return self.balance - self.margin_used

    def equity(self, unrealized_pnl: float = 0.0) -> float:
        return self.balance + unrealized_pnl

    def allocate_margin(self, amount: float) -> None:
        if amount > self.available_balance:
            raise ValueError(
                f"Insufficient balance: need {amount:.2f}, "
                f"available {self.available_balance:.2f}"
            )
        self.margin_used += amount

    def release_margin(self, amount: float) -> None:
        self.margin_used = max(0.0, self.margin_used - amount)

    def realize_pnl(self, pnl: float) -> None:
        self.balance += pnl

    def apply_fee(self, fee: float) -> None:
        self.balance -= fee

    def reset(self) -> None:
        self.balance = self.initial_balance
        self.margin_used = 0.0
