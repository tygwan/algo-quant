"""Real-time data pipeline with WebSocket support."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from collections import deque

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams."""
    PRICE = "price"
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    TICKER = "ticker"


@dataclass
class StreamConfig:
    """Configuration for data stream."""
    symbols: list[str]
    stream_type: StreamType = StreamType.PRICE
    buffer_size: int = 1000
    update_interval: float = 1.0  # seconds
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10


@dataclass
class PriceUpdate:
    """Real-time price update."""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
        }


@dataclass
class OrderBookUpdate:
    """Real-time order book update."""
    symbol: str
    bids: list[tuple[float, float]]  # (price, quantity)
    asks: list[tuple[float, float]]
    timestamp: datetime

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class DataStream(ABC):
    """Abstract base class for data streams."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self._buffer: dict[str, deque] = {
            symbol: deque(maxlen=config.buffer_size)
            for symbol in config.symbols
        }
        self._callbacks: list[Callable] = []
        self._running = False
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Connect to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source."""
        pass

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols."""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        pass

    def add_callback(self, callback: Callable) -> None:
        """Add callback for data updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _notify_callbacks(self, data: Any) -> None:
        """Notify all callbacks with new data."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_latest(self, symbol: str) -> Optional[Any]:
        """Get latest data for symbol."""
        if symbol in self._buffer and self._buffer[symbol]:
            return self._buffer[symbol][-1]
        return None

    def get_history(self, symbol: str, n: int = 100) -> list[Any]:
        """Get recent history for symbol."""
        if symbol in self._buffer:
            return list(self._buffer[symbol])[-n:]
        return []

    def to_dataframe(self, symbol: str) -> pd.DataFrame:
        """Convert buffer to DataFrame."""
        history = self.get_history(symbol, self.config.buffer_size)
        if not history:
            return pd.DataFrame()

        return pd.DataFrame([
            item.to_dict() if hasattr(item, 'to_dict') else item
            for item in history
        ])


class BinanceStream(DataStream):
    """Binance WebSocket stream implementation."""

    BASE_URL = "wss://stream.binance.com:9443/ws"
    TESTNET_URL = "wss://testnet.binance.vision/ws"

    def __init__(self, config: StreamConfig, testnet: bool = False):
        super().__init__(config)
        self.testnet = testnet
        self._ws = None
        self._reconnect_count = 0

    @property
    def url(self) -> str:
        return self.TESTNET_URL if self.testnet else self.BASE_URL

    async def connect(self) -> None:
        """Connect to Binance WebSocket."""
        try:
            import websockets

            streams = self._build_stream_names()
            url = f"{self.url}/{'/'.join(streams)}"

            self._ws = await websockets.connect(url)
            self._connected = True
            self._reconnect_count = 0
            logger.info(f"Connected to Binance WebSocket: {url}")

        except ImportError:
            logger.warning("websockets package not installed, using mock stream")
            self._connected = True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from Binance WebSocket."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
        self._running = False
        logger.info("Disconnected from Binance WebSocket")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to additional symbols."""
        if not self._ws:
            return

        streams = [f"{s.lower()}@trade" for s in symbols]
        msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(datetime.now().timestamp()),
        }
        await self._ws.send(json.dumps(msg))

        for symbol in symbols:
            if symbol not in self._buffer:
                self._buffer[symbol] = deque(maxlen=self.config.buffer_size)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        if not self._ws:
            return

        streams = [f"{s.lower()}@trade" for s in symbols]
        msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": int(datetime.now().timestamp()),
        }
        await self._ws.send(json.dumps(msg))

    async def start(self) -> None:
        """Start receiving data."""
        self._running = True

        while self._running:
            try:
                if not self._connected:
                    await self._reconnect()

                if self._ws:
                    msg = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=self.config.update_interval * 2
                    )
                    await self._handle_message(json.loads(msg))
                else:
                    # Mock data for testing
                    await self._generate_mock_data()
                    await asyncio.sleep(self.config.update_interval)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Stream error: {e}")
                self._connected = False
                await asyncio.sleep(self.config.reconnect_delay)

    async def _reconnect(self) -> None:
        """Attempt to reconnect."""
        if self._reconnect_count >= self.config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self._running = False
            return

        self._reconnect_count += 1
        logger.info(f"Reconnecting ({self._reconnect_count}/{self.config.max_reconnect_attempts})...")

        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            await asyncio.sleep(self.config.reconnect_delay)

    async def _handle_message(self, msg: dict) -> None:
        """Handle incoming WebSocket message."""
        if "e" not in msg:
            return

        event_type = msg["e"]

        if event_type == "trade":
            update = PriceUpdate(
                symbol=msg["s"],
                price=float(msg["p"]),
                volume=float(msg["q"]),
                timestamp=datetime.fromtimestamp(msg["T"] / 1000),
            )
            self._buffer[msg["s"]].append(update)
            await self._notify_callbacks(update)

    async def _generate_mock_data(self) -> None:
        """Generate mock data for testing."""
        for symbol in self.config.symbols:
            # Simulate price movement
            last = self.get_latest(symbol)
            if last:
                change = np.random.normal(0, 0.001)
                price = last.price * (1 + change)
            else:
                price = 100 + np.random.uniform(-5, 5)

            update = PriceUpdate(
                symbol=symbol,
                price=price,
                volume=np.random.uniform(100, 10000),
                timestamp=datetime.now(),
                bid=price * 0.9999,
                ask=price * 1.0001,
            )
            self._buffer[symbol].append(update)
            await self._notify_callbacks(update)

    def _build_stream_names(self) -> list[str]:
        """Build stream names based on config."""
        streams = []

        for symbol in self.config.symbols:
            s = symbol.lower()
            if self.config.stream_type == StreamType.PRICE:
                streams.append(f"{s}@ticker")
            elif self.config.stream_type == StreamType.TRADE:
                streams.append(f"{s}@trade")
            elif self.config.stream_type == StreamType.ORDERBOOK:
                streams.append(f"{s}@depth10")

        return streams


class RealtimeDataPipeline:
    """Manages multiple data streams and aggregates data."""

    def __init__(self):
        self._streams: dict[str, DataStream] = {}
        self._aggregated_data: dict[str, pd.DataFrame] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    def add_stream(self, name: str, stream: DataStream) -> None:
        """Add a data stream."""
        self._streams[name] = stream
        stream.add_callback(self._on_data_update)

    def remove_stream(self, name: str) -> None:
        """Remove a data stream."""
        if name in self._streams:
            del self._streams[name]

    async def start(self) -> None:
        """Start all streams."""
        self._running = True

        for name, stream in self._streams.items():
            try:
                await stream.connect()
                task = asyncio.create_task(stream.start())
                self._tasks.append(task)
                logger.info(f"Started stream: {name}")
            except Exception as e:
                logger.error(f"Failed to start stream {name}: {e}")

    async def stop(self) -> None:
        """Stop all streams."""
        self._running = False

        for task in self._tasks:
            task.cancel()

        for name, stream in self._streams.items():
            try:
                await stream.disconnect()
                logger.info(f"Stopped stream: {name}")
            except Exception as e:
                logger.error(f"Error stopping stream {name}: {e}")

        self._tasks.clear()

    async def _on_data_update(self, data: Any) -> None:
        """Handle data update from stream."""
        if isinstance(data, PriceUpdate):
            symbol = data.symbol
            if symbol not in self._aggregated_data:
                self._aggregated_data[symbol] = pd.DataFrame()

            new_row = pd.DataFrame([data.to_dict()])
            self._aggregated_data[symbol] = pd.concat(
                [self._aggregated_data[symbol], new_row],
                ignore_index=True
            ).tail(10000)  # Keep last 10k rows

    def get_latest_prices(self) -> dict[str, float]:
        """Get latest prices for all symbols."""
        prices = {}
        for name, stream in self._streams.items():
            for symbol in stream.config.symbols:
                latest = stream.get_latest(symbol)
                if latest:
                    prices[symbol] = latest.price
        return prices

    def get_ohlcv(self, symbol: str, timeframe: str = "1min") -> pd.DataFrame:
        """Get OHLCV data resampled to timeframe."""
        if symbol not in self._aggregated_data:
            return pd.DataFrame()

        df = self._aggregated_data[symbol].copy()
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        ohlcv = df["price"].resample(timeframe).ohlc()
        ohlcv["volume"] = df["volume"].resample(timeframe).sum()

        return ohlcv.dropna()

    def calculate_vwap(self, symbol: str, window: int = 100) -> Optional[float]:
        """Calculate VWAP for symbol."""
        if symbol not in self._aggregated_data:
            return None

        df = self._aggregated_data[symbol].tail(window)
        if df.empty:
            return None

        return (df["price"] * df["volume"]).sum() / df["volume"].sum()

    def calculate_spread(self, symbol: str) -> Optional[float]:
        """Calculate bid-ask spread."""
        for stream in self._streams.values():
            latest = stream.get_latest(symbol)
            if latest and latest.bid and latest.ask:
                return (latest.ask - latest.bid) / latest.price * 10000  # bps
        return None
