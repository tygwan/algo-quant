"""Execution engine for live trading."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd

from src.execution.broker import (
    BrokerInterface,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    OrderStatus,
    PaperBroker,
)
from src.execution.realtime import (
    RealtimeDataPipeline,
    BinanceStream,
    StreamConfig,
    StreamType,
    PriceUpdate,
)
from src.execution.rebalancer import (
    AutoRebalancer,
    RebalanceConfig,
    RebalanceTrigger,
    Position,
)

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes."""
    PAPER = "paper"         # Paper trading (simulation)
    LIVE = "live"           # Live trading
    BACKTEST = "backtest"   # Backtesting mode


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    mode: ExecutionMode = ExecutionMode.PAPER
    symbols: list[str] = field(default_factory=list)

    # Risk settings
    max_position_size: float = 0.20    # 20% max per position
    max_daily_loss: float = 0.05       # 5% max daily loss
    max_drawdown: float = 0.15         # 15% max drawdown

    # Order settings
    default_order_type: OrderType = OrderType.MARKET
    slippage_tolerance: float = 0.002  # 0.2%

    # Rebalancing
    enable_auto_rebalance: bool = True
    rebalance_config: Optional[RebalanceConfig] = None

    # Logging
    log_all_trades: bool = True
    save_trade_history: bool = True


@dataclass
class TradeRecord:
    """Record of executed trade."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    order_id: str
    pnl: Optional[float] = None
    notes: str = ""


@dataclass
class ExecutionState:
    """Current state of execution engine."""
    is_running: bool = False
    is_connected: bool = False
    portfolio_value: float = 0.0
    cash: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    pending_orders: list[OrderResult] = field(default_factory=list)


class ExecutionEngine:
    """Main execution engine for live trading."""

    def __init__(
        self,
        config: ExecutionConfig,
        broker: Optional[BrokerInterface] = None,
        strategy_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.broker = broker or PaperBroker()
        self.strategy_fn = strategy_fn

        # Initialize components
        self._pipeline = RealtimeDataPipeline()
        self._rebalancer = AutoRebalancer(
            config.rebalance_config or RebalanceConfig()
        )

        # State
        self._state = ExecutionState()
        self._trade_history: list[TradeRecord] = []
        self._target_weights: dict[str, float] = {}
        self._start_value: float = 0.0

        # Callbacks
        self._on_trade_callbacks: list[Callable] = []
        self._on_signal_callbacks: list[Callable] = []

    async def start(self) -> None:
        """Start the execution engine."""
        logger.info(f"Starting execution engine in {self.config.mode.value} mode")

        # Connect to broker
        connected = await self.broker.connect()
        if not connected:
            raise RuntimeError("Failed to connect to broker")

        self._state.is_connected = True

        # Initialize account state
        await self._update_account_state()
        self._start_value = self._state.portfolio_value

        # Setup data pipeline
        if self.config.symbols:
            stream_config = StreamConfig(
                symbols=self.config.symbols,
                stream_type=StreamType.TRADE,
            )
            stream = BinanceStream(
                stream_config,
                testnet=self.config.mode == ExecutionMode.PAPER
            )
            self._pipeline.add_stream("main", stream)
            await self._pipeline.start()

        self._state.is_running = True
        logger.info("Execution engine started")

    async def stop(self) -> None:
        """Stop the execution engine."""
        logger.info("Stopping execution engine")

        self._state.is_running = False

        # Cancel pending orders
        for order in self._state.pending_orders:
            await self.broker.cancel_order(order.order_id)

        # Disconnect
        await self._pipeline.stop()
        await self.broker.disconnect()

        self._state.is_connected = False
        logger.info("Execution engine stopped")

    async def run(self, duration_seconds: Optional[float] = None) -> None:
        """Run the main execution loop.

        Args:
            duration_seconds: Optional duration to run (None = indefinite)
        """
        start_time = datetime.now()

        while self._state.is_running:
            try:
                # Check duration
                if duration_seconds:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= duration_seconds:
                        logger.info("Duration reached, stopping")
                        break

                # Update state
                await self._update_account_state()

                # Check risk limits
                if not self._check_risk_limits():
                    logger.warning("Risk limit breached, pausing trading")
                    await asyncio.sleep(60)
                    continue

                # Get strategy signals
                if self.strategy_fn:
                    signals = await self._get_strategy_signals()
                    await self._process_signals(signals)

                # Check rebalancing
                if self.config.enable_auto_rebalance:
                    await self._check_rebalance()

                # Small delay
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                await asyncio.sleep(5)

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: Optional[OrderType] = None,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        """Submit a trading order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Order type (default: market)
            limit_price: Limit price for limit orders

        Returns:
            Order result
        """
        if not self._state.is_connected:
            raise RuntimeError("Not connected to broker")

        # Create order request
        order = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type or self.config.default_order_type,
            limit_price=limit_price,
        )

        # Validate order
        valid, msg = order.validate()
        if not valid:
            logger.warning(f"Invalid order: {msg}")
            return OrderResult(
                order_id="",
                client_order_id=order.client_order_id,
                symbol=symbol,
                side=side,
                status=OrderStatus.REJECTED,
                quantity=quantity,
                message=msg,
            )

        # Check position limits
        if not self._check_position_limit(symbol, side, quantity):
            return OrderResult(
                order_id="",
                client_order_id=order.client_order_id,
                symbol=symbol,
                side=side,
                status=OrderStatus.REJECTED,
                quantity=quantity,
                message="Position limit exceeded",
            )

        # Submit to broker
        result = await self.broker.submit_order(order)

        # Record trade
        if result.status == OrderStatus.FILLED:
            self._record_trade(result)

        # Notify callbacks
        for callback in self._on_trade_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        return await self.broker.cancel_order(order_id)

    def set_target_weights(self, weights: dict[str, float]) -> None:
        """Set target portfolio weights for rebalancing.

        Args:
            weights: Target weight for each symbol (must sum to <= 1)
        """
        total = sum(weights.values())
        if total > 1.0001:
            raise ValueError(f"Weights sum to {total}, must be <= 1.0")

        self._target_weights = weights
        logger.info(f"Target weights updated: {weights}")

    def add_trade_callback(self, callback: Callable) -> None:
        """Add callback for trade events."""
        self._on_trade_callbacks.append(callback)

    def add_signal_callback(self, callback: Callable) -> None:
        """Add callback for strategy signals."""
        self._on_signal_callbacks.append(callback)

    def get_state(self) -> ExecutionState:
        """Get current execution state."""
        return self._state

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self._trade_history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "commission": t.commission,
                "pnl": t.pnl,
                "notes": t.notes,
            }
            for t in self._trade_history
        ])

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if self._start_value == 0:
            return {}

        total_return = (self._state.portfolio_value - self._start_value) / self._start_value

        df = self.get_trade_history()
        if df.empty:
            return {
                "total_return": 0,
                "num_trades": 0,
            }

        return {
            "total_return": total_return,
            "portfolio_value": self._state.portfolio_value,
            "cash": self._state.cash,
            "num_trades": len(df),
            "total_commission": df["commission"].sum(),
            "win_rate": (df["pnl"] > 0).mean() if "pnl" in df and df["pnl"].notna().any() else 0,
        }

    async def _update_account_state(self) -> None:
        """Update account state from broker."""
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()
        open_orders = await self.broker.get_open_orders()

        self._state.cash = account.cash
        self._state.portfolio_value = account.portfolio_value

        # Update positions
        self._state.positions = {
            p.symbol: Position(
                symbol=p.symbol,
                quantity=p.quantity,
                avg_cost=p.avg_cost,
                current_price=p.market_value / p.quantity if p.quantity > 0 else 0,
            )
            for p in positions
        }

        self._state.pending_orders = open_orders

        # Calculate PnL
        self._state.total_pnl = self._state.portfolio_value - self._start_value

    def _check_risk_limits(self) -> bool:
        """Check if risk limits are breached."""
        if self._start_value == 0:
            return True

        # Check daily loss
        daily_return = self._state.daily_pnl / self._start_value
        if daily_return < -self.config.max_daily_loss:
            logger.warning(f"Daily loss limit breached: {daily_return:.2%}")
            return False

        # Check drawdown
        drawdown = (self._start_value - self._state.portfolio_value) / self._start_value
        if drawdown > self.config.max_drawdown:
            logger.warning(f"Drawdown limit breached: {drawdown:.2%}")
            return False

        return True

    def _check_position_limit(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> bool:
        """Check if order would exceed position limits."""
        prices = self._pipeline.get_latest_prices()
        price = prices.get(symbol, 0)

        if price == 0:
            return True  # Can't validate without price

        trade_value = quantity * price
        position_pct = trade_value / self._state.portfolio_value

        if side == OrderSide.BUY and position_pct > self.config.max_position_size:
            current_pos = self._state.positions.get(symbol)
            current_value = current_pos.market_value if current_pos else 0
            total_pct = (current_value + trade_value) / self._state.portfolio_value

            if total_pct > self.config.max_position_size:
                logger.warning(
                    f"Position limit exceeded for {symbol}: {total_pct:.2%} > {self.config.max_position_size:.2%}"
                )
                return False

        return True

    async def _get_strategy_signals(self) -> dict[str, float]:
        """Get signals from strategy function."""
        if not self.strategy_fn:
            return {}

        try:
            # Get current data
            prices = self._pipeline.get_latest_prices()
            positions = self._state.positions

            # Call strategy
            signals = self.strategy_fn(prices, positions)

            # Notify callbacks
            for callback in self._on_signal_callbacks:
                callback(signals)

            return signals

        except Exception as e:
            logger.error(f"Strategy error: {e}")
            return {}

    async def _process_signals(self, signals: dict[str, float]) -> None:
        """Process strategy signals and execute trades.

        Args:
            signals: Dictionary of symbol -> signal value
                     Positive = buy, Negative = sell, 0 = hold
        """
        for symbol, signal in signals.items():
            if abs(signal) < 0.01:  # Ignore small signals
                continue

            side = OrderSide.BUY if signal > 0 else OrderSide.SELL
            quantity = abs(signal)

            await self.submit_order(symbol, side, quantity)

    async def _check_rebalance(self) -> None:
        """Check and execute rebalancing if needed."""
        if not self._target_weights:
            return

        # Get current weights
        total_value = self._state.portfolio_value
        if total_value == 0:
            return

        current_weights = {
            symbol: pos.market_value / total_value
            for symbol, pos in self._state.positions.items()
        }

        # Check if rebalance needed
        needs_rebalance, reason = self._rebalancer.check_rebalance_needed(
            current_weights, self._target_weights
        )

        if needs_rebalance:
            logger.info(f"Rebalancing: {reason}")

            prices = self._pipeline.get_latest_prices()

            result = self._rebalancer.execute_rebalance(
                self._state.positions,
                self._target_weights,
                total_value,
                prices,
                execute_fn=self._execute_rebalance_trade,
            )

            logger.info(f"Rebalance result: {result.message}")

    async def _execute_rebalance_trade(self, trade) -> None:
        """Execute a trade from rebalancer."""
        side = OrderSide.BUY if trade.side == "buy" else OrderSide.SELL
        await self.submit_order(trade.symbol, side, trade.quantity)

    def _record_trade(self, result: OrderResult) -> None:
        """Record a completed trade."""
        # Calculate PnL for sells
        pnl = None
        if result.side == OrderSide.SELL:
            pos = self._state.positions.get(result.symbol)
            if pos:
                pnl = (result.average_price - pos.avg_cost) * result.filled_quantity

        record = TradeRecord(
            timestamp=result.timestamp,
            symbol=result.symbol,
            side=result.side.value,
            quantity=result.filled_quantity,
            price=result.average_price,
            commission=result.commission,
            order_id=result.order_id,
            pnl=pnl,
        )

        self._trade_history.append(record)

        if self.config.log_all_trades:
            logger.info(
                f"Trade: {record.side} {record.quantity} {record.symbol} "
                f"@ {record.price:.4f} (PnL: {record.pnl or 'N/A'})"
            )
