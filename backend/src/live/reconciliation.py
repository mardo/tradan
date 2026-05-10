"""Reconcile exchange state against the runner's last logged state.

Rule (per design spec):
- If the exchange has positions/orders the runner did not log → refuse to
  resume. The runner does not know how to handle state it did not create.
- Logged positions/orders that no longer exist at the exchange are fine —
  they were closed/filled/cancelled while we were down.
- Balance differences are informational; we record them but do not block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from live.exchange.base import Balance, Order, Position


@dataclass(frozen=True)
class ReconciliationOutcome:
    action: str            # "resume" | "refuse"
    diff_notes: str        # human-readable summary


def reconcile(
    *,
    last_logged_account_state: dict[str, Any],
    exchange_balance: Balance,
    exchange_positions: list[Position],
    exchange_orders: list[Order],
    known_order_ids: set[str] | None = None,
) -> ReconciliationOutcome:
    logged_position_ids = {
        p["id"] for p in (last_logged_account_state.get("positions") or [])
    }
    logged_order_ids = {
        o["id"] for o in (last_logged_account_state.get("open_orders") or [])
    }
    known_order_ids = known_order_ids or set()
    all_known_orders = logged_order_ids | known_order_ids

    unknown_positions = [
        p for p in exchange_positions if p.id not in logged_position_ids
    ]
    unknown_orders = [
        o for o in exchange_orders if o.id not in all_known_orders
    ]

    if unknown_positions or unknown_orders:
        notes = "Refusing resume: unknown exchange state."
        if unknown_positions:
            notes += f" Unknown positions: {[p.id for p in unknown_positions]}."
        if unknown_orders:
            notes += f" Unknown orders: {[o.id for o in unknown_orders]}."
        return ReconciliationOutcome(action="refuse", diff_notes=notes.strip())

    closed_positions = (
        logged_position_ids
        - {p.id for p in exchange_positions}
    )
    cancelled_or_filled_orders = (
        logged_order_ids
        - {o.id for o in exchange_orders}
    )
    notes = []
    if closed_positions:
        notes.append(f"closed positions: {sorted(closed_positions)}")
    if cancelled_or_filled_orders:
        notes.append(f"closed orders: {sorted(cancelled_or_filled_orders)}")
    if not notes:
        notes.append("matched cleanly")

    return ReconciliationOutcome(action="resume", diff_notes="; ".join(notes))
