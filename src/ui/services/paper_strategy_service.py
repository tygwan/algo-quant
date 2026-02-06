"""Paper strategy helpers used by Live Analyzer callbacks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.ui.services.realtime_hub import get_realtime_market_hub

logger = logging.getLogger(__name__)


def _fetch_historical_prices(
    tickers: list[str], period: str
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch historical close prices for a symbol list."""
    from src.data.yfinance_client import YFinanceClient

    yf_client = YFinanceClient()
    prices_dict: dict[str, pd.Series] = {}
    failed: list[str] = []

    for ticker in tickers[:20]:
        try:
            df = yf_client.get_historical_prices(ticker, period=period)
            if not df.empty and "close" in df.columns:
                prices_dict[ticker] = df["close"]
            else:
                failed.append(ticker)
        except Exception as exc:
            failed.append(ticker)
            logger.warning("Failed to fetch %s: %s", ticker, exc)

    return pd.DataFrame(prices_dict), failed


def select_strategy_targets(
    prices_df: pd.DataFrame,
    strategy: str,
    top_n: int,
) -> list[str]:
    """Select target symbols from price history based on strategy type."""
    if prices_df.empty:
        return []

    frame = prices_df.dropna(axis=1, how="all")
    if frame.empty:
        return []

    top_n = max(1, min(int(top_n), len(frame.columns)))
    lookback = min(20, max(2, len(frame) - 1))

    if strategy == "mean_reversion":
        scores = -(frame.iloc[-1] / frame.iloc[-lookback] - 1)
    elif strategy == "low_vol_momentum":
        returns_df = frame.pct_change().dropna(how="all")
        if returns_df.empty:
            return []
        vol = returns_df.tail(lookback).std().replace(0, np.nan)
        mom = frame.iloc[-1] / frame.iloc[-lookback] - 1
        scores = mom.rank(pct=True).fillna(0) * 0.7 + (1 - vol.rank(pct=True)).fillna(0) * 0.3
    else:
        scores = frame.iloc[-1] / frame.iloc[-lookback] - 1

    scores = scores.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    return [str(sym) for sym in scores.head(top_n).index]


def latest_prices_for_symbols(
    prices_df: pd.DataFrame,
    symbols: list[str],
) -> dict[str, float]:
    """Extract valid latest prices for selected symbols."""
    latest_prices: dict[str, float] = {}

    for symbol in symbols:
        if symbol not in prices_df.columns:
            continue
        series = pd.to_numeric(prices_df[symbol], errors="coerce").dropna()
        if series.empty:
            continue
        price = float(series.iloc[-1])
        if np.isfinite(price) and price > 0:
            latest_prices[symbol] = price

    return latest_prices


def build_prices_for_paper_run(
    symbols: list[str],
    data_mode: str,
    period: str,
) -> pd.DataFrame:
    """Build price frame from realtime hub or historical source."""
    if not symbols:
        return pd.DataFrame()

    if data_mode == "realtime":
        hub = get_realtime_market_hub()
        prices_df = hub.get_price_frame(symbols)
        if not prices_df.empty:
            return prices_df

        snapshot_df = hub.get_latest_snapshot(symbols)
        if snapshot_df.empty:
            return pd.DataFrame()

        now = pd.Timestamp.utcnow()
        row = {
            str(r["symbol"]): float(r["price"])
            for _, r in snapshot_df.iterrows()
            if pd.notna(r.get("price"))
        }
        return pd.DataFrame([row], index=[now]) if row else pd.DataFrame()

    prices_df, _ = _fetch_historical_prices(symbols, period)
    return prices_df


async def run_paper_strategy_once(
    prices_df: pd.DataFrame,
    strategy: str,
    top_n: int,
    initial_capital: float,
    commission_rate: float,
) -> dict[str, Any]:
    """Run one-shot paper execution and return result payload."""
    from src.execution.broker import OrderSide, PaperBroker
    from src.execution.executor import ExecutionConfig, ExecutionEngine, ExecutionMode

    selected = select_strategy_targets(prices_df, strategy=strategy, top_n=top_n)
    if not selected:
        return {"ok": False, "message": "No strategy targets selected from current data."}

    latest_prices = latest_prices_for_symbols(prices_df, selected)
    selected = [symbol for symbol in selected if symbol in latest_prices]
    if not selected:
        return {"ok": False, "message": "No valid latest prices found for selected targets."}

    broker = PaperBroker(
        initial_cash=float(initial_capital),
        commission_rate=float(commission_rate),
    )
    broker.update_prices(latest_prices)

    engine = ExecutionEngine(
        config=ExecutionConfig(
            mode=ExecutionMode.PAPER,
            symbols=[],
            enable_auto_rebalance=False,
            max_position_size=1.0,
        ),
        broker=broker,
    )

    order_rows: list[dict[str, Any]] = []
    alloc_cash = float(initial_capital) * 0.995
    cash_per_symbol = alloc_cash / max(1, len(selected))

    try:
        await engine.start()

        for symbol in selected:
            price = latest_prices[symbol]
            qty = max(cash_per_symbol / price, 0.0)
            qty = float(np.floor(qty * 10000) / 10000)
            if qty <= 0:
                continue

            result = await engine.submit_order(symbol, OrderSide.BUY, quantity=qty)
            order_rows.append(
                {
                    "symbol": symbol,
                    "status": result.status.value,
                    "quantity": float(result.filled_quantity),
                    "price": float(result.average_price),
                    "commission": float(result.commission),
                    "message": result.message,
                }
            )

        await engine._update_account_state()
        summary = engine.get_performance_summary()
        state = engine.get_state()
        position_rows = [
            {
                "symbol": pos.symbol,
                "quantity": float(pos.quantity),
                "avg_cost": float(pos.avg_cost),
                "market_value": float(pos.market_value),
            }
            for pos in state.positions.values()
        ]

        return {
            "ok": bool(order_rows),
            "message": "Paper strategy execution completed."
            if order_rows
            else "No orders were filled.",
            "strategy": strategy,
            "selected": selected,
            "orders": order_rows,
            "positions": position_rows,
            "summary": summary,
        }
    finally:
        await engine.stop()
