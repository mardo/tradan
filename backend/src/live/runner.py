"""LiveRunner — orchestrates feature pipeline → model → action decoder
→ adapter dispatch.

This file ships ONLY the replay entry point. The production loop (run_live)
is implemented in Phase F.
"""
from __future__ import annotations

from dataclasses import dataclass

from live.action_decoder import RiskClampConfig, clamp_intent
from live.exchange.replay import ReplayAdapter
from live.feature_pipeline import build_live_observation
from live.model_runner import ModelRunner
from trainer.config import ModelConfig
from trainer.env.action_decoder import DecoderState, decode_action
from trainer.env.normalization import NormalizationStats, load_stats
from trainer.env.observation import ObservationConfig


@dataclass
class ReplayResult:
    final_equity: float
    total_steps: int


def run_replay(
    *,
    adapter: ReplayAdapter,
    model_runner: ModelRunner,
    model_config: ModelConfig,
    stats: NormalizationStats,
    max_position_size_pct: float = 1.0,
    max_leverage: float = 125.0,
) -> ReplayResult:
    """Drive the replay adapter forward one candle at a time.

    Mirrors TradingEnv.step's control flow:
      1. process the current candle (adapter.advance does this)
      2. observe (build_live_observation)
      3. predict (model_runner.predict)
      4. decode + clamp (decode_action, clamp_intent)
      5. apply intent via sim.apply_intent (replay path uses the trainer's
         simulator directly, bypassing the adapter's place_order/cancel_order
         which are NotImplementedError on ReplayAdapter)
    """
    obs_cfg = ObservationConfig(
        lookback=model_config.lookback_window,
        num_features=len(model_config.columns),
        max_open_orders=model_config.exchange.max_open_orders,
        max_open_positions=model_config.exchange.max_open_positions,
        max_leverage=model_config.exchange.max_leverage,
        initial_balance=model_config.initial_balance,
    )

    # Prime the adapter so it has at least `lookback` klines available.
    while adapter.cursor < model_config.lookback_window:
        adapter.advance()

    steps = 0
    total_features = len(adapter._state.features)
    while adapter.cursor < total_features:
        klines = adapter.fetch_klines(
            adapter._state.symbol,
            adapter._state.interval,
            limit=model_config.lookback_window,
        )
        balance = adapter.fetch_balance()
        positions = adapter.fetch_positions(adapter._state.symbol)
        open_orders = adapter.fetch_open_orders(adapter._state.symbol)

        obs = build_live_observation(
            klines=klines, columns=model_config.columns,
            balance=balance, positions=positions, open_orders=open_orders,
            stats=stats, obs_cfg=obs_cfg,
        )
        result = model_runner.predict(obs)

        state = DecoderState(
            close=float(klines[-1].close),
            available_balance=balance.available,
            num_open_orders=len(open_orders),
            num_open_positions=len(positions),
        )
        intent = decode_action(result.action, state, model_config)
        intent = clamp_intent(intent, RiskClampConfig(
            equity=balance.total,
            max_position_size_pct=max_position_size_pct,
            max_leverage=max_leverage,
        ))

        # Replay path: bypass the adapter's place_order and call sim.apply_intent
        # directly — the same code TradingEnv.step uses, ensuring 0% divergence.
        adapter.sim.apply_intent(intent, current_price=state.close)
        adapter.advance()
        steps += 1

    final_balance = adapter.fetch_balance()
    return ReplayResult(final_equity=final_balance.total, total_steps=steps)


# ----------------------------------------------------------------------------
# Production live runner (F.2). The replay path above is unaffected.
# ----------------------------------------------------------------------------

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Json

from ingester.db import connect
from live.config import LiveConfig
from live.db import (
    find_running_run,
    is_kill_requested,
    log_action,
    log_order,
    log_pnl_snapshot,
    start_run,
    stop_run,
)
from live.exchange.base import Balance, ExchangeAdapter, Kline, Order, OrderRequest, Position
from live.exchange.registry import get_adapter_class
from live.reconciliation import reconcile
from trainer.env.action_decoder import OpenIntent, OrderIntent


_POLL_SECONDS = 30


@dataclass
class _LiveContext:
    cfg: LiveConfig
    adapter: ExchangeAdapter
    model_runner: ModelRunner
    model_config: ModelConfig
    stats: NormalizationStats
    obs_cfg: ObservationConfig
    conn: psycopg.Connection
    run_id: int
    dry_run: bool
    interval_ms: int            # candle close-to-close, ms
    last_processed_close: int   # ms timestamp
    last_pnl_snapshot_at: float # monotonic seconds
    consecutive_errors: int


class _GracefulExit(Exception):
    """Raised by signal handlers to request orderly shutdown."""
    def __init__(self, reason: str):
        self.reason = reason


def run_live(*, config_path: str, dry_run: bool = False) -> int:
    """Production entry: load config, attach to or create a live_runs row,
    reconcile if resuming, run the polling loop, shut down gracefully."""
    cfg = LiveConfig.from_yaml(config_path)

    adapter_cls = get_adapter_class(cfg.exchange.name)
    # Concrete adapters (e.g. BingXAdapter) implement a `from_env` classmethod.
    adapter = adapter_cls.from_env(
        api_key_env=cfg.exchange.api_key_env,
        api_secret_env=cfg.exchange.api_secret_env,
        mode=cfg.exchange.mode,
    )

    conn = connect()
    try:
        model_id, model_cfg = _load_model_cfg(conn, cfg.model.name)
        models_dir = Path(os.environ.get("MODELS_DIR", "/var/lib/tradan/models"))
        model_path = _resolve_model_path(conn, model_id, models_dir, cfg.model.name)
        stats = load_stats(model_path.with_suffix(""))
        model_runner = ModelRunner(
            model_path=model_path, algorithm=model_cfg.algorithm,
        )
        obs_cfg = ObservationConfig(
            lookback=model_cfg.lookback_window,
            num_features=len(model_cfg.columns),
            max_open_orders=model_cfg.exchange.max_open_orders,
            max_open_positions=model_cfg.exchange.max_open_positions,
            max_leverage=model_cfg.exchange.max_leverage,
            initial_balance=model_cfg.initial_balance,
        )

        existing = find_running_run(
            conn, model_config_id=model_id, exchange=cfg.exchange.name,
        )
        if existing is not None:
            run_id = existing
            _do_reconciliation(conn, adapter, run_id, cfg.market.symbol)
        else:
            with conn.transaction():
                run_id = start_run(
                    conn, model_config_id=model_id,
                    exchange=cfg.exchange.name, mode=cfg.exchange.mode,
                    symbol=cfg.market.symbol, interval=cfg.market.interval,
                    starting_equity=cfg.risk.starting_equity_quote,
                    config_yaml=Path(config_path).read_text(),
                    git_sha=_git_sha(),
                )

        try:
            adapter.set_leverage(cfg.market.symbol, cfg.risk.max_leverage)
        except NotImplementedError:
            pass

        ctx = _LiveContext(
            cfg=cfg, adapter=adapter, model_runner=model_runner,
            model_config=model_cfg, stats=stats, obs_cfg=obs_cfg,
            conn=conn, run_id=run_id, dry_run=dry_run,
            interval_ms=_interval_to_ms(cfg.market.interval),
            last_processed_close=0,
            last_pnl_snapshot_at=time.monotonic(),
            consecutive_errors=0,
        )

        _install_signal_handlers()
        return _loop(ctx)
    finally:
        conn.close()


def _loop(ctx: _LiveContext) -> int:
    try:
        while True:
            # 1. kill checks
            if is_kill_requested(ctx.conn, ctx.run_id):
                _shutdown(ctx, reason="kill_switch")
                return 0
            if os.environ.get(ctx.cfg.risk.kill_switch_env, "").lower() == "true":
                _shutdown(ctx, reason="kill_switch")
                return 0

            # 2. fetch latest klines, detect new candle
            try:
                klines = ctx.adapter.fetch_klines(
                    ctx.cfg.market.symbol, ctx.cfg.market.interval,
                    limit=ctx.model_config.lookback_window,
                )
                ctx.consecutive_errors = 0
            except Exception as e:
                ctx.consecutive_errors += 1
                log_action(
                    ctx.conn, live_run_id=ctx.run_id, event_type="error",
                    account_state={"error": str(e)},
                    notes=f"fetch_klines failed: {e!r}",
                )
                if ctx.consecutive_errors >= 3:
                    _shutdown(ctx, reason="error")
                    return 1
                time.sleep(_POLL_SECONDS)
                continue

            if not klines:
                time.sleep(_POLL_SECONDS)
                continue

            newest_close = klines[-1].open_time_ms + ctx.interval_ms
            if newest_close > ctx.last_processed_close:
                _on_new_candle(ctx, klines)
                ctx.last_processed_close = newest_close

            # 3. periodic pnl snapshot + drawdown check
            elapsed = time.monotonic() - ctx.last_pnl_snapshot_at
            if elapsed >= ctx.cfg.logging.pnl_snapshot_interval_minutes * 60:
                _take_snapshot(ctx)
                ctx.last_pnl_snapshot_at = time.monotonic()
                bal = ctx.adapter.fetch_balance()
                threshold = ctx.cfg.risk.starting_equity_quote * (
                    1.0 - ctx.cfg.risk.max_drawdown_pct
                )
                if bal.total < threshold:
                    _shutdown(ctx, reason="drawdown")
                    return 0

            time.sleep(_POLL_SECONDS)

    except _GracefulExit as e:
        _shutdown(ctx, reason=e.reason)
        return 0
    except Exception as e:
        log_action(
            ctx.conn, live_run_id=ctx.run_id, event_type="error",
            account_state={"error": str(e)}, notes=repr(e),
        )
        _shutdown(ctx, reason="error")
        return 1


def _on_new_candle(ctx: _LiveContext, klines: list[Kline]) -> None:
    bal = ctx.adapter.fetch_balance()
    positions = ctx.adapter.fetch_positions(ctx.cfg.market.symbol)
    open_orders = ctx.adapter.fetch_open_orders(ctx.cfg.market.symbol)

    obs = build_live_observation(
        klines=klines, columns=ctx.model_config.columns,
        balance=bal, positions=positions, open_orders=open_orders,
        stats=ctx.stats, obs_cfg=ctx.obs_cfg,
    )
    pred = ctx.model_runner.predict(obs)

    candle_close_ts = datetime.fromtimestamp(
        (klines[-1].open_time_ms + ctx.interval_ms) / 1000.0, tz=timezone.utc,
    )
    account_state = _account_state_dict(bal, positions, open_orders)

    action_id = log_action(
        ctx.conn, live_run_id=ctx.run_id, event_type="inference",
        candle_close=candle_close_ts,
        raw_action=pred.action.tolist(),
        account_state=account_state, inference_ms=pred.inference_ms,
    )

    state = DecoderState(
        close=float(klines[-1].close),
        available_balance=bal.available,
        num_open_orders=len(open_orders),
        num_open_positions=len(positions),
    )
    intent = decode_action(pred.action, state, ctx.model_config)
    intent = clamp_intent(intent, RiskClampConfig(
        equity=bal.total,
        max_position_size_pct=ctx.cfg.risk.max_position_size_pct,
        max_leverage=ctx.cfg.risk.max_leverage,
    ))

    ctx.conn.execute(
        "UPDATE live_actions SET decoded_intent = %s WHERE id = %s",
        (Json(_intent_dict(intent)), action_id),
    )

    if not ctx.dry_run:
        _execute_intent(ctx, intent, action_id, positions, open_orders)


def _execute_intent(
    ctx: _LiveContext,
    intent: OrderIntent,
    action_id: int,
    positions: list[Position],
    open_orders: list[Order],
) -> None:
    # Cancel — descending so removing earlier indices doesn't shift later ones.
    for i in sorted(intent.cancels, reverse=True):
        if i < len(open_orders):
            target = open_orders[i]
            try:
                ctx.adapter.cancel_order(ctx.cfg.market.symbol, target.id)
                log_order(
                    ctx.conn, live_run_id=ctx.run_id, live_action_id=action_id,
                    exchange_order_id=target.id, side=target.side, type=target.type,
                    price=target.price, amount=target.amount, status="cancelled",
                )
            except Exception as e:
                log_action(
                    ctx.conn, live_run_id=ctx.run_id, event_type="error",
                    account_state=_account_state_dict(
                        ctx.adapter.fetch_balance(), positions, open_orders,
                    ),
                    notes=f"cancel_order failed for {target.id}: {e!r}",
                )

    # Close
    for ci in intent.closes:
        if ci.position_index < len(positions):
            target = positions[ci.position_index]
            try:
                order = ctx.adapter.close_position(
                    ctx.cfg.market.symbol, target.id, ci.fraction,
                )
                log_order(
                    ctx.conn, live_run_id=ctx.run_id, live_action_id=action_id,
                    exchange_order_id=order.id, side=order.side, type=order.type,
                    price=order.price, amount=order.amount, status=order.status,
                )
            except Exception as e:
                log_action(
                    ctx.conn, live_run_id=ctx.run_id, event_type="error",
                    account_state=_account_state_dict(
                        ctx.adapter.fetch_balance(), positions, open_orders,
                    ),
                    notes=f"close_position failed for {target.id}: {e!r}",
                )

    # Open
    if intent.open is not None:
        op = intent.open
        leverage = _approx_leverage(op, ctx)
        amount = (op.margin * leverage) / op.trigger_price
        side = "buy" if op.direction == 1 else "sell"
        try:
            order = ctx.adapter.place_order(
                ctx.cfg.market.symbol,
                OrderRequest(
                    side=side, type="limit", amount=amount, price=op.trigger_price,
                    stop_loss=op.sl_price,
                    take_profit=op.tp_prices[0] if op.tp_prices else None,
                ),
            )
            log_order(
                ctx.conn, live_run_id=ctx.run_id, live_action_id=action_id,
                exchange_order_id=order.id, side=order.side, type=order.type,
                price=order.price, amount=order.amount, status=order.status,
            )
        except Exception as e:
            log_action(
                ctx.conn, live_run_id=ctx.run_id, event_type="error",
                account_state=_account_state_dict(
                    ctx.adapter.fetch_balance(), positions, open_orders,
                ),
                notes=f"place_order failed: {e!r}",
            )


def _take_snapshot(ctx: _LiveContext) -> None:
    bal = ctx.adapter.fetch_balance()
    positions = ctx.adapter.fetch_positions(ctx.cfg.market.symbol)
    open_orders = ctx.adapter.fetch_open_orders(ctx.cfg.market.symbol)
    unrealized = sum(p.unrealized_pnl for p in positions)
    realized = bal.total - unrealized - ctx.cfg.risk.starting_equity_quote
    log_pnl_snapshot(
        ctx.conn, live_run_id=ctx.run_id,
        equity=bal.total,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        open_positions=len(positions),
        open_orders=len(open_orders),
    )


def _shutdown(ctx: _LiveContext, *, reason: str) -> None:
    """Graceful close: cancel orders, flatten positions, finalize run row."""
    try:
        try:
            for o in ctx.adapter.fetch_open_orders(ctx.cfg.market.symbol):
                try:
                    ctx.adapter.cancel_order(ctx.cfg.market.symbol, o.id)
                except Exception as e:
                    log_action(
                        ctx.conn, live_run_id=ctx.run_id, event_type="error",
                        account_state={},
                        notes=f"cancel during shutdown failed for {o.id}: {e!r}",
                    )
        except Exception as e:
            log_action(
                ctx.conn, live_run_id=ctx.run_id, event_type="error",
                account_state={}, notes=f"fetch_open_orders during shutdown failed: {e!r}",
            )

        try:
            for p in ctx.adapter.fetch_positions(ctx.cfg.market.symbol):
                try:
                    ctx.adapter.close_position(ctx.cfg.market.symbol, p.id, fraction=1.0)
                except Exception as e:
                    log_action(
                        ctx.conn, live_run_id=ctx.run_id, event_type="error",
                        account_state={},
                        notes=f"close during shutdown failed for {p.id}: {e!r}",
                    )
        except Exception as e:
            log_action(
                ctx.conn, live_run_id=ctx.run_id, event_type="error",
                account_state={}, notes=f"fetch_positions during shutdown failed: {e!r}",
            )

        try:
            _take_snapshot(ctx)
        except Exception as e:
            log_action(
                ctx.conn, live_run_id=ctx.run_id, event_type="error",
                account_state={}, notes=f"final snapshot failed: {e!r}",
            )
    finally:
        with ctx.conn.transaction():
            stop_run(ctx.conn, ctx.run_id, reason=reason)


def _do_reconciliation(
    conn: psycopg.Connection, adapter: ExchangeAdapter, run_id: int, symbol: str,
) -> None:
    last = conn.execute(
        """
        SELECT account_state FROM live_actions
        WHERE live_run_id = %s
        ORDER BY created_at DESC LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    last_state: dict[str, Any] = last[0] if last else {}
    bal = adapter.fetch_balance()
    positions = adapter.fetch_positions(symbol)
    open_orders = adapter.fetch_open_orders(symbol)
    outcome = reconcile(
        last_logged_account_state=last_state,
        exchange_balance=bal,
        exchange_positions=positions,
        exchange_orders=open_orders,
    )
    log_action(
        conn, live_run_id=run_id, event_type="reconciliation",
        account_state=_account_state_dict(bal, positions, open_orders),
        notes=outcome.diff_notes,
    )
    if outcome.action == "refuse":
        with conn.transaction():
            stop_run(conn, run_id, reason="reconciliation_failed")
        sys.exit(2)


# -- helpers ------------------------------------------------------------------


def _install_signal_handlers() -> None:
    def handler(_signum, _frame):
        raise _GracefulExit(reason="manual")
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
        ).decode().strip()
    except Exception:
        return "unknown"


def _interval_to_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000, "5m": 5 * 60_000, "15m": 15 * 60_000,
        "30m": 30 * 60_000, "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000, "1d": 24 * 60 * 60_000,
    }
    return mapping[interval]


def _account_state_dict(
    bal: Balance, positions: list[Position], orders: list[Order],
) -> dict[str, Any]:
    return {
        "balance": {"total": bal.total, "available": bal.available, "used": bal.used},
        "positions": [
            {"id": p.id, "side": p.side, "size": p.size,
             "entry_price": p.entry_price, "leverage": p.leverage}
            for p in positions
        ],
        "open_orders": [
            {"id": o.id, "side": o.side, "type": o.type,
             "price": o.price, "amount": o.amount}
            for o in orders
        ],
    }


def _intent_dict(intent: OrderIntent) -> dict[str, Any]:
    return {
        "open": (intent.open and {
            "direction": intent.open.direction,
            "trigger_price": intent.open.trigger_price,
            "sl_price": intent.open.sl_price,
            "tp_prices": intent.open.tp_prices,
            "tp_size_pcts": intent.open.tp_size_pcts,
            "margin": intent.open.margin,
        }),
        "cancels": list(intent.cancels),
        "closes": [
            {"position_index": c.position_index, "fraction": c.fraction}
            for c in intent.closes
        ],
    }


def _approx_leverage(open_intent: OpenIntent, ctx: _LiveContext) -> float:
    """Mirror ExchangeSim.compute_leverage so the live order's notional matches
    what the trainer's simulator would have produced for the same intent."""
    sl_dist = abs(open_intent.trigger_price - open_intent.sl_price) / open_intent.trigger_price
    if sl_dist == 0:
        return 1.0
    mm = ctx.model_config.exchange.maintenance_margin_pct / 100.0
    buf = ctx.model_config.exchange.liquidation_buffer_pct / 100.0
    lev = 1.0 / (sl_dist + buf + mm)
    return min(lev, ctx.cfg.risk.max_leverage)


def _load_model_cfg(conn: psycopg.Connection, name: str) -> tuple[int, ModelConfig]:
    row = conn.execute(
        "SELECT id, config_json FROM model_configs WHERE name = %s", (name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"model {name!r} not found in model_configs")
    return row[0], ModelConfig.from_dict({"name": name, **row[1]})


def _resolve_model_path(
    conn: psycopg.Connection, model_id: int, models_dir: Path, name: str,
) -> Path:
    """Find the SB3 model.zip path. Prefer training_runs.model_path (most
    reliable); fall back to <MODELS_DIR>/<name>/best_model.zip if the DB
    has no completed training run recorded."""
    row = conn.execute(
        """
        SELECT model_path FROM training_runs
        WHERE model_config_id = %s AND run_type = 'train' AND status = 'completed'
        ORDER BY id DESC LIMIT 1
        """,
        (model_id,),
    ).fetchone()
    if row and row[0]:
        return Path(row[0])
    return models_dir / name / "best_model.zip"
