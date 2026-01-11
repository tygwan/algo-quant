"""Auto-rebalancing system for portfolio management."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RebalanceTrigger(Enum):
    """Triggers for rebalancing."""
    SCHEDULED = "scheduled"      # Time-based (daily, weekly, monthly)
    THRESHOLD = "threshold"      # Drift threshold exceeded
    SIGNAL = "signal"            # Strategy signal
    MANUAL = "manual"            # User-initiated


class RebalanceMethod(Enum):
    """Methods for rebalancing."""
    FULL = "full"                # Rebalance all positions
    PARTIAL = "partial"          # Only adjust drifted positions
    GRADUAL = "gradual"          # Spread over multiple trades


@dataclass
class RebalanceConfig:
    """Configuration for auto-rebalancer."""
    # Trigger settings
    trigger: RebalanceTrigger = RebalanceTrigger.THRESHOLD
    schedule_interval: str = "weekly"  # daily, weekly, monthly
    drift_threshold: float = 0.05      # 5% drift triggers rebalance

    # Execution settings
    method: RebalanceMethod = RebalanceMethod.PARTIAL
    min_trade_size: float = 100.0      # Minimum trade in dollars
    max_trade_pct: float = 0.25        # Max 25% of position per trade
    gradual_steps: int = 4             # Steps for gradual rebalancing

    # Cost considerations
    commission_rate: float = 0.001     # 0.1% commission
    slippage_estimate: float = 0.0005  # 0.05% slippage

    # Risk limits
    max_single_position: float = 0.30  # Max 30% in single asset
    min_cash_reserve: float = 0.02     # Keep 2% cash


@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class TradeOrder:
    """Order to execute for rebalancing."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    reason: str = ""

    @property
    def is_buy(self) -> bool:
        return self.side == "buy"


@dataclass
class RebalanceResult:
    """Result of rebalance operation."""
    timestamp: datetime
    trigger: RebalanceTrigger
    orders: list[TradeOrder]
    estimated_cost: float
    drift_before: dict[str, float]
    drift_after: dict[str, float]
    success: bool = True
    message: str = ""


class AutoRebalancer:
    """Automatic portfolio rebalancer."""

    def __init__(self, config: RebalanceConfig):
        self.config = config
        self._last_rebalance: Optional[datetime] = None
        self._rebalance_history: list[RebalanceResult] = []

    def check_rebalance_needed(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> tuple[bool, str]:
        """Check if rebalancing is needed.

        Returns:
            Tuple of (needs_rebalance, reason)
        """
        if self.config.trigger == RebalanceTrigger.SCHEDULED:
            return self._check_schedule(), "Scheduled rebalance"

        elif self.config.trigger == RebalanceTrigger.THRESHOLD:
            max_drift = self._calculate_max_drift(current_weights, target_weights)
            if max_drift > self.config.drift_threshold:
                return True, f"Drift threshold exceeded: {max_drift:.2%}"
            return False, f"Within threshold: {max_drift:.2%}"

        elif self.config.trigger == RebalanceTrigger.SIGNAL:
            return False, "Waiting for signal"

        return False, "Unknown trigger"

    def calculate_trades(
        self,
        positions: dict[str, Position],
        target_weights: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float],
    ) -> list[TradeOrder]:
        """Calculate trades needed to rebalance.

        Args:
            positions: Current positions
            target_weights: Target weight for each symbol
            portfolio_value: Total portfolio value
            prices: Current prices

        Returns:
            List of trades to execute
        """
        trades = []

        # Calculate current weights
        current_weights = {}
        for symbol, pos in positions.items():
            current_weights[symbol] = pos.market_value / portfolio_value

        # Add missing symbols with 0 weight
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)

            # Calculate target value and difference
            target_value = target * portfolio_value
            current_value = current * portfolio_value
            diff_value = target_value - current_value

            # Skip small trades
            if abs(diff_value) < self.config.min_trade_size:
                continue

            # Apply position limits
            if target > self.config.max_single_position:
                target_value = self.config.max_single_position * portfolio_value
                diff_value = target_value - current_value

            # Calculate quantity
            price = prices.get(symbol, 0)
            if price <= 0:
                logger.warning(f"Invalid price for {symbol}: {price}")
                continue

            quantity = abs(diff_value) / price

            # Apply gradual rebalancing if configured
            if self.config.method == RebalanceMethod.GRADUAL:
                quantity = quantity / self.config.gradual_steps

            # Apply max trade percentage
            if symbol in positions:
                max_qty = positions[symbol].quantity * self.config.max_trade_pct
                quantity = min(quantity, max_qty)

            trade = TradeOrder(
                symbol=symbol,
                side="buy" if diff_value > 0 else "sell",
                quantity=quantity,
                reason=f"Rebalance: {current:.2%} -> {target:.2%}",
            )
            trades.append(trade)

        return self._optimize_trade_order(trades, prices)

    def execute_rebalance(
        self,
        positions: dict[str, Position],
        target_weights: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float],
        execute_fn: Optional[callable] = None,
    ) -> RebalanceResult:
        """Execute rebalancing.

        Args:
            positions: Current positions
            target_weights: Target weights
            portfolio_value: Total portfolio value
            prices: Current prices
            execute_fn: Function to execute trades (for live trading)

        Returns:
            RebalanceResult with trade details
        """
        # Calculate current drift
        current_weights = {
            s: p.market_value / portfolio_value
            for s, p in positions.items()
        }
        drift_before = self._calculate_drift(current_weights, target_weights)

        # Calculate trades
        trades = self.calculate_trades(
            positions, target_weights, portfolio_value, prices
        )

        # Estimate costs
        estimated_cost = self._estimate_costs(trades, prices)

        # Execute if function provided
        if execute_fn and trades:
            try:
                for trade in trades:
                    execute_fn(trade)
                success = True
                message = f"Executed {len(trades)} trades"
            except Exception as e:
                success = False
                message = f"Execution failed: {e}"
        else:
            success = True
            message = f"Generated {len(trades)} trades (dry run)"

        # Calculate expected drift after
        drift_after = self._simulate_drift_after(
            current_weights, target_weights, trades, portfolio_value, prices
        )

        result = RebalanceResult(
            timestamp=datetime.now(),
            trigger=self.config.trigger,
            orders=trades,
            estimated_cost=estimated_cost,
            drift_before=drift_before,
            drift_after=drift_after,
            success=success,
            message=message,
        )

        self._rebalance_history.append(result)
        self._last_rebalance = datetime.now()

        return result

    def get_rebalance_schedule(self) -> Optional[datetime]:
        """Get next scheduled rebalance time."""
        if self.config.trigger != RebalanceTrigger.SCHEDULED:
            return None

        if self._last_rebalance is None:
            return datetime.now()

        intervals = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30),
            "quarterly": timedelta(days=90),
        }

        interval = intervals.get(self.config.schedule_interval, timedelta(weeks=1))
        return self._last_rebalance + interval

    def get_drift_summary(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> pd.DataFrame:
        """Get summary of portfolio drift."""
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        data = []
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            drift = current - target

            data.append({
                "symbol": symbol,
                "current_weight": current,
                "target_weight": target,
                "drift": drift,
                "drift_pct": drift / target if target > 0 else 0,
                "action": "buy" if drift < 0 else "sell" if drift > 0 else "hold",
            })

        return pd.DataFrame(data).sort_values("drift", key=abs, ascending=False)

    def _check_schedule(self) -> bool:
        """Check if scheduled rebalance is due."""
        next_rebalance = self.get_rebalance_schedule()
        if next_rebalance is None:
            return False
        return datetime.now() >= next_rebalance

    def _calculate_drift(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate drift for each position."""
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        return {
            symbol: current_weights.get(symbol, 0) - target_weights.get(symbol, 0)
            for symbol in all_symbols
        }

    def _calculate_max_drift(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> float:
        """Calculate maximum absolute drift."""
        drift = self._calculate_drift(current_weights, target_weights)
        return max(abs(d) for d in drift.values()) if drift else 0

    def _estimate_costs(
        self,
        trades: list[TradeOrder],
        prices: dict[str, float],
    ) -> float:
        """Estimate transaction costs."""
        total_cost = 0

        for trade in trades:
            price = prices.get(trade.symbol, 0)
            trade_value = trade.quantity * price

            # Commission
            commission = trade_value * self.config.commission_rate

            # Slippage estimate
            slippage = trade_value * self.config.slippage_estimate

            total_cost += commission + slippage

        return total_cost

    def _optimize_trade_order(
        self,
        trades: list[TradeOrder],
        prices: dict[str, float],
    ) -> list[TradeOrder]:
        """Optimize order of trades to minimize cash needs.

        Sells first, then buys.
        """
        sells = [t for t in trades if t.side == "sell"]
        buys = [t for t in trades if t.side == "buy"]

        # Sort sells by value (largest first)
        sells.sort(
            key=lambda t: t.quantity * prices.get(t.symbol, 0),
            reverse=True
        )

        # Sort buys by value (smallest first)
        buys.sort(
            key=lambda t: t.quantity * prices.get(t.symbol, 0)
        )

        return sells + buys

    def _simulate_drift_after(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        trades: list[TradeOrder],
        portfolio_value: float,
        prices: dict[str, float],
    ) -> dict[str, float]:
        """Simulate drift after executing trades."""
        new_weights = current_weights.copy()

        for trade in trades:
            trade_value = trade.quantity * prices.get(trade.symbol, 0)
            weight_change = trade_value / portfolio_value

            if trade.is_buy:
                new_weights[trade.symbol] = new_weights.get(trade.symbol, 0) + weight_change
            else:
                new_weights[trade.symbol] = new_weights.get(trade.symbol, 0) - weight_change

        return self._calculate_drift(new_weights, target_weights)


class RebalanceScheduler:
    """Scheduler for automatic rebalancing."""

    def __init__(self, rebalancer: AutoRebalancer):
        self.rebalancer = rebalancer
        self._running = False
        self._task: Optional[Any] = None

    async def start(
        self,
        get_positions_fn: callable,
        get_targets_fn: callable,
        get_prices_fn: callable,
        execute_fn: Optional[callable] = None,
        check_interval: float = 60.0,
    ) -> None:
        """Start the rebalance scheduler.

        Args:
            get_positions_fn: Function to get current positions
            get_targets_fn: Function to get target weights
            get_prices_fn: Function to get current prices
            execute_fn: Function to execute trades
            check_interval: Interval to check for rebalance (seconds)
        """
        import asyncio

        self._running = True
        logger.info("Rebalance scheduler started")

        while self._running:
            try:
                positions = get_positions_fn()
                targets = get_targets_fn()
                prices = get_prices_fn()

                portfolio_value = sum(p.market_value for p in positions.values())
                current_weights = {
                    s: p.market_value / portfolio_value
                    for s, p in positions.items()
                }

                needs_rebalance, reason = self.rebalancer.check_rebalance_needed(
                    current_weights, targets
                )

                if needs_rebalance:
                    logger.info(f"Rebalancing triggered: {reason}")
                    result = self.rebalancer.execute_rebalance(
                        positions, targets, portfolio_value, prices, execute_fn
                    )
                    logger.info(f"Rebalance result: {result.message}")

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(check_interval)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        logger.info("Rebalance scheduler stopped")
