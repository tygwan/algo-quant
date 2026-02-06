"""Realtime market data hub for dashboard streaming.

This service runs websocket streams in a background thread and exposes
thread-safe snapshots for Dash callbacks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any

import pandas as pd

from src.env import load_local_env
from src.execution.realtime import (
    BinanceStream,
    FinnhubStockStream,
    PriceUpdate,
    RealtimeDataPipeline,
    StreamConfig,
    StreamType,
)

logger = logging.getLogger(__name__)

CRYPTO_QUOTE_SUFFIXES = (
    "USDT",
    "USDC",
    "BUSD",
    "BTC",
    "ETH",
    "BNB",
)


def _normalize_symbols(symbols: list[str]) -> list[str]:
    normalized = []
    for symbol in symbols:
        s = symbol.strip().upper().replace(" ", "")
        if s:
            compact = s.replace("/", "").replace("-", "").replace("_", "")
            if any(compact.endswith(suffix) for suffix in CRYPTO_QUOTE_SUFFIXES):
                s = compact
            normalized.append(s)
    # preserve order while deduplicating
    return list(dict.fromkeys(normalized))


def _is_crypto_symbol(symbol: str) -> bool:
    symbol = symbol.upper()
    return symbol.endswith(CRYPTO_QUOTE_SUFFIXES)


class RealtimeMarketHub:
    """Background hub for stocks+crypto real-time updates."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()

        self._running = False
        self._symbols: list[str] = []
        self._providers: list[str] = []
        self._errors: list[str] = []

        self._latest: dict[str, dict[str, Any]] = {}
        self._history: dict[str, deque] = {}

    def start(self, symbols: list[str]) -> dict[str, Any]:
        """Start or restart stream workers with a symbol universe."""
        normalized = _normalize_symbols(symbols)
        if not normalized:
            return {"started": False, "reason": "No symbols provided"}

        with self._lock:
            if self._running and normalized == self._symbols:
                return {
                    "started": True,
                    "reused": True,
                    "symbols": self._symbols,
                    "providers": self._providers,
                }

        self.stop()

        with self._lock:
            self._symbols = normalized
            self._providers = []
            self._errors = []
            self._latest = {}
            self._history = {}

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_thread,
            name="realtime-market-hub",
            daemon=True,
        )
        self._thread.start()

        # Give the worker a moment to connect
        time.sleep(0.3)

        status = self.get_status()
        status["started"] = status.get("running", False) or bool(status.get("providers"))
        return status

    def stop(self) -> None:
        """Stop background streaming workers."""
        self._stop_event.set()

        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(lambda: None)

        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=4)

        with self._lock:
            self._running = False
            self._thread = None
            self._loop = None

    def get_status(self) -> dict[str, Any]:
        """Get current hub status."""
        with self._lock:
            latest_ts = None
            if self._latest:
                latest_ts = max(
                    (row.get("timestamp") for row in self._latest.values()),
                    default=None,
                )

            return {
                "running": self._running,
                "symbols": list(self._symbols),
                "providers": list(self._providers),
                "errors": list(self._errors[-5:]),
                "latest_update": latest_ts.isoformat() if isinstance(latest_ts, datetime) else None,
            }

    def get_latest_snapshot(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """Return latest tick snapshot for symbols."""
        target = set(_normalize_symbols(symbols or self._symbols))

        with self._lock:
            rows = [
                {
                    "symbol": sym,
                    "price": rec.get("price", 0.0),
                    "volume": rec.get("volume", 0.0),
                    "timestamp": rec.get("timestamp"),
                    "source": rec.get("source", ""),
                }
                for sym, rec in self._latest.items()
                if sym in target
            ]

        if not rows:
            return pd.DataFrame(columns=["symbol", "price", "volume", "timestamp", "source"])

        return pd.DataFrame(rows).sort_values("symbol")

    def get_price_frame(self, symbols: list[str], max_points: int = 400) -> pd.DataFrame:
        """Return aligned time-series frame from in-memory tick history."""
        normalized = _normalize_symbols(symbols)

        with self._lock:
            series_map: dict[str, pd.Series] = {}

            for symbol in normalized:
                history = list(self._history.get(symbol, []))[-max_points:]
                if not history:
                    continue

                idx = pd.to_datetime([item["timestamp"] for item in history])
                vals = [item["price"] for item in history]
                series = pd.Series(vals, index=idx, name=symbol)
                # Use last value in same timestamp bucket
                series = series.groupby(level=0).last().sort_index()
                series_map[symbol] = series

        if not series_map:
            return pd.DataFrame()

        prices_df = pd.DataFrame(series_map).sort_index().ffill()
        return prices_df.dropna(how="all")

    def get_tick_history(self, symbol: str, max_points: int = 5000) -> pd.DataFrame:
        """Return raw tick history for one symbol."""
        normalized = _normalize_symbols([symbol])
        if not normalized:
            return pd.DataFrame(columns=["timestamp", "price", "volume", "source"])

        target = normalized[0]
        with self._lock:
            history = list(self._history.get(target, []))[-max_points:]

        if not history:
            return pd.DataFrame(columns=["timestamp", "price", "volume", "source"])

        frame = pd.DataFrame(history)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "price", "volume", "source"])

        frame = frame.loc[:, ["timestamp", "price", "volume", "source"]]
        frame = frame.sort_values("timestamp")
        return frame

    def _run_thread(self) -> None:
        """Run async streaming pipeline in a dedicated thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._run_async())
        except Exception as e:
            self._append_error(f"Realtime worker failed: {e}")
            logger.exception("Realtime worker failed")
        finally:
            self._loop.close()

    async def _run_async(self) -> None:
        pipeline = RealtimeDataPipeline()

        with self._lock:
            symbols = list(self._symbols)

        crypto_symbols = [s for s in symbols if _is_crypto_symbol(s)]
        stock_symbols = [s for s in symbols if not _is_crypto_symbol(s)]

        if crypto_symbols:
            crypto_stream = BinanceStream(
                StreamConfig(
                    symbols=crypto_symbols,
                    stream_type=StreamType.TRADE,
                    buffer_size=4000,
                    update_interval=1.0,
                ),
                testnet=False,
            )
            crypto_stream.add_callback(self._on_price_update("binance"))
            pipeline.add_stream("crypto", crypto_stream)
            self._append_provider("binance")

        if stock_symbols:
            load_local_env()
            finnhub_key = os.getenv("FINNHUB_API_KEY", "").strip()
            if finnhub_key:
                stock_stream = FinnhubStockStream(
                    StreamConfig(
                        symbols=stock_symbols,
                        stream_type=StreamType.TRADE,
                        buffer_size=4000,
                        update_interval=1.0,
                    ),
                    api_key=finnhub_key,
                )
                stock_stream.add_callback(self._on_price_update("finnhub"))
                pipeline.add_stream("stocks", stock_stream)
                self._append_provider("finnhub")
            else:
                self._append_error(
                    "FINNHUB_API_KEY is not set. Stock realtime stream is disabled."
                )

        with self._lock:
            self._running = bool(pipeline._streams)

        if not pipeline._streams:
            return

        await pipeline.start()
        logger.info("Realtime hub streams started")

        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)
        finally:
            await pipeline.stop()
            with self._lock:
                self._running = False
            logger.info("Realtime hub streams stopped")

    def _on_price_update(self, source: str):
        """Create callback that stores price updates safely."""

        def _callback(update: PriceUpdate) -> None:
            record = {
                "symbol": update.symbol,
                "price": float(update.price),
                "volume": float(update.volume),
                "timestamp": update.timestamp,
                "source": source,
            }
            with self._lock:
                self._latest[update.symbol] = record
                if update.symbol not in self._history:
                    self._history[update.symbol] = deque(maxlen=6000)
                self._history[update.symbol].append(record)

        return _callback

    def _append_error(self, message: str) -> None:
        with self._lock:
            self._errors.append(message)
        logger.warning(message)

    def _append_provider(self, provider: str) -> None:
        with self._lock:
            if provider not in self._providers:
                self._providers.append(provider)


_realtime_hub_singleton = RealtimeMarketHub()


def get_realtime_market_hub() -> RealtimeMarketHub:
    """Return singleton realtime market hub."""
    return _realtime_hub_singleton
