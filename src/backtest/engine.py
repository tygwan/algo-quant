"""Backtesting engine for strategy evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Callable, Any
from enum import Enum

import numpy as np
import pandas as pd


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Trade:
    """Executed trade record.
    
    Attributes:
        timestamp: Trade timestamp
        symbol: Asset symbol
        side: Buy or sell
        quantity: Trade quantity
        price: Execution price
        commission: Commission paid
        slippage: Slippage incurred
    """
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def value(self) -> float:
        """Trade value (excluding costs)."""
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage."""
        return self.commission + self.slippage


@dataclass
class Position:
    """Portfolio position.
    
    Attributes:
        symbol: Asset symbol
        quantity: Position quantity
        entry_price: Average entry price
        entry_date: First entry date
    """
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_date: datetime | None = None
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    def update(
        self,
        quantity_change: float,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Update position with new trade.
        
        Args:
            quantity_change: Change in quantity (positive = buy)
            price: Trade price
            timestamp: Trade timestamp
        """
        if self.quantity == 0:
            self.entry_price = price
            self.entry_date = timestamp
        elif (self.quantity > 0 and quantity_change > 0) or \
             (self.quantity < 0 and quantity_change < 0):
            # Adding to position - update average price
            total_cost = self.quantity * self.entry_price + quantity_change * price
            self.quantity += quantity_change
            if self.quantity != 0:
                self.entry_price = total_cost / self.quantity
        else:
            self.quantity += quantity_change
            if self.quantity == 0:
                self.entry_price = 0
                self.entry_date = None


@dataclass
class BacktestConfig:
    """Backtest configuration.
    
    Attributes:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        commission: Commission rate (e.g., 0.001 = 0.1%)
        slippage: Slippage rate (e.g., 0.0005 = 0.05%)
        margin_requirement: Margin requirement for short selling
        allow_short: Allow short selling
        rebalance_frequency: Rebalance frequency (daily, weekly, monthly)
    """
    start_date: str | datetime
    end_date: str | datetime
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    margin_requirement: float = 0.5
    allow_short: bool = True
    rebalance_frequency: str = "monthly"
    
    def __post_init__(self):
        if isinstance(self.start_date, str):
            self.start_date = pd.to_datetime(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = pd.to_datetime(self.end_date)


@dataclass
class BacktestResult:
    """Backtest result.
    
    Attributes:
        config: Backtest configuration
        portfolio_values: Portfolio value time series
        returns: Return time series
        trades: List of executed trades
        positions: Final positions
        metrics: Performance metrics
    """
    config: BacktestConfig
    portfolio_values: pd.Series
    returns: pd.Series
    trades: list[Trade]
    positions: dict[str, Position]
    weights_history: pd.DataFrame | None = None
    metrics: dict | None = None
    
    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "initial_capital": self.config.initial_capital,
            "final_value": self.portfolio_values.iloc[-1] if len(self.portfolio_values) > 0 else 0,
            "total_return": (self.portfolio_values.iloc[-1] / self.config.initial_capital - 1) 
                if len(self.portfolio_values) > 0 else 0,
            "num_trades": len(self.trades),
            "metrics": self.metrics,
        }


class Strategy(Protocol):
    """Strategy protocol for backtest engine."""
    
    def generate_weights(
        self,
        prices: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """Generate target weights for rebalancing.
        
        Args:
            prices: Price data up to current date
            **kwargs: Additional data
            
        Returns:
            Target portfolio weights
        """
        ...


class BacktestEngine:
    """Backtesting engine.
    
    Evaluates strategy performance using historical data.
    Handles position management, transaction costs, and performance tracking.
    
    Example:
        >>> config = BacktestConfig(
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31",
        ...     initial_capital=100000,
        ... )
        >>> engine = BacktestEngine(config)
        >>> result = engine.run(strategy, prices)
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.positions: dict[str, Position] = {}
        self.cash = config.initial_capital
        self.trades: list[Trade] = []
        self.portfolio_values: list[tuple[datetime, float]] = []
        self.weights_history: list[tuple[datetime, dict[str, float]]] = []
    
    def run(
        self,
        strategy: Strategy | Callable,
        prices: pd.DataFrame,
        additional_data: dict | None = None,
    ) -> BacktestResult:
        """Run backtest.
        
        Args:
            strategy: Strategy object or callable
            prices: OHLCV price data (index=date, columns=symbols)
            additional_data: Additional data for strategy
            
        Returns:
            BacktestResult with performance data
        """
        # Reset state
        self._reset()
        
        # Filter dates
        prices = prices.loc[self.config.start_date:self.config.end_date]
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(prices.index)
        
        # Run simulation
        for date in prices.index:
            current_prices = prices.loc[date]
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.portfolio_values.append((date, portfolio_value))
            
            # Rebalance if needed
            if date in rebalance_dates:
                historical_prices = prices.loc[:date]
                
                # Generate target weights
                if callable(strategy) and not hasattr(strategy, 'generate_weights'):
                    target_weights = strategy(historical_prices)
                else:
                    kwargs = additional_data or {}
                    target_weights = strategy.generate_weights(
                        historical_prices, **kwargs
                    )
                
                # Execute rebalancing trades
                self._rebalance(target_weights, current_prices, date)
                self.weights_history.append((date, dict(target_weights)))
        
        # Create result
        portfolio_series = pd.Series(
            dict(self.portfolio_values),
            name="portfolio_value"
        )
        returns = portfolio_series.pct_change().dropna()
        
        weights_df = None
        if self.weights_history:
            weights_df = pd.DataFrame(
                [w for _, w in self.weights_history],
                index=[d for d, _ in self.weights_history]
            )
        
        return BacktestResult(
            config=self.config,
            portfolio_values=portfolio_series,
            returns=returns,
            trades=self.trades,
            positions=self.positions.copy(),
            weights_history=weights_df,
        )
    
    def _reset(self) -> None:
        """Reset engine state."""
        self.positions = {}
        self.cash = self.config.initial_capital
        self.trades = []
        self.portfolio_values = []
        self.weights_history = []
    
    def _get_rebalance_dates(
        self,
        dates: pd.DatetimeIndex,
    ) -> set[datetime]:
        """Get rebalancing dates based on frequency.
        
        Args:
            dates: All available dates
            
        Returns:
            Set of rebalancing dates
        """
        if self.config.rebalance_frequency == "daily":
            return set(dates)
        elif self.config.rebalance_frequency == "weekly":
            # Rebalance on Mondays
            return set(d for d in dates if d.dayofweek == 0)
        elif self.config.rebalance_frequency == "monthly":
            # Rebalance on first trading day of month
            return set(dates.to_series().groupby(pd.Grouper(freq='MS')).first())
        elif self.config.rebalance_frequency == "quarterly":
            # Rebalance on first trading day of quarter
            return set(dates.to_series().groupby(pd.Grouper(freq='QS')).first())
        else:
            return set(dates)
    
    def _calculate_portfolio_value(
        self,
        prices: pd.Series,
    ) -> float:
        """Calculate current portfolio value.
        
        Args:
            prices: Current prices for all assets
            
        Returns:
            Total portfolio value
        """
        position_value = sum(
            pos.quantity * prices.get(symbol, 0)
            for symbol, pos in self.positions.items()
            if pos.quantity != 0
        )
        return self.cash + position_value
    
    def _rebalance(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        date: datetime,
    ) -> None:
        """Rebalance portfolio to target weights.
        
        Args:
            target_weights: Target portfolio weights
            prices: Current prices
            date: Current date
        """
        portfolio_value = self._calculate_portfolio_value(prices)
        
        # Calculate current weights
        current_weights = {}
        for symbol in set(target_weights.index) | set(self.positions.keys()):
            if symbol in self.positions and self.positions[symbol].quantity != 0:
                pos_value = self.positions[symbol].quantity * prices.get(symbol, 0)
                current_weights[symbol] = pos_value / portfolio_value
            else:
                current_weights[symbol] = 0.0
        
        # Calculate trades needed
        for symbol in set(target_weights.index) | set(current_weights.keys()):
            target_weight = target_weights.get(symbol, 0)
            current_weight = current_weights.get(symbol, 0)
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) < 0.001:  # Skip small trades
                continue
            
            price = prices.get(symbol, 0)
            if price <= 0:
                continue
            
            # Calculate trade value and quantity
            trade_value = weight_diff * portfolio_value
            quantity = trade_value / price
            
            # Check short selling constraint
            if not self.config.allow_short and quantity < 0:
                current_pos = self.positions.get(symbol)
                if current_pos is None or current_pos.quantity + quantity < 0:
                    quantity = -current_pos.quantity if current_pos else 0
            
            if abs(quantity) < 0.001:
                continue
            
            # Execute trade
            self._execute_trade(symbol, quantity, price, date)
    
    def _execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        date: datetime,
    ) -> Trade:
        """Execute a trade.
        
        Args:
            symbol: Asset symbol
            quantity: Trade quantity (positive = buy)
            price: Trade price
            date: Trade date
            
        Returns:
            Executed trade
        """
        # Apply slippage
        if quantity > 0:
            exec_price = price * (1 + self.config.slippage)
        else:
            exec_price = price * (1 - self.config.slippage)
        
        # Calculate commission
        trade_value = abs(quantity * exec_price)
        commission = trade_value * self.config.commission
        slippage_cost = abs(quantity) * abs(exec_price - price)
        
        # Update cash
        self.cash -= quantity * exec_price + commission
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        self.positions[symbol].update(quantity, exec_price, date)
        
        # Record trade
        trade = Trade(
            timestamp=date,
            symbol=symbol,
            side=OrderSide.BUY if quantity > 0 else OrderSide.SELL,
            quantity=abs(quantity),
            price=exec_price,
            commission=commission,
            slippage=slippage_cost,
        )
        self.trades.append(trade)
        
        return trade


class VectorizedBacktester:
    """Fast vectorized backtester for simple strategies.
    
    Suitable for strategies that can be expressed as weight matrices.
    Much faster than event-driven but less flexible.
    
    Example:
        >>> backtester = VectorizedBacktester()
        >>> result = backtester.run(weights, returns, initial_capital=100000)
    """
    
    def __init__(
        self,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """Initialize vectorized backtester.
        
        Args:
            commission: Commission rate
            slippage: Slippage rate
        """
        self.commission = commission
        self.slippage = slippage
    
    def run(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        initial_capital: float = 100000.0,
    ) -> BacktestResult:
        """Run vectorized backtest.
        
        Args:
            weights: Target weights over time (index=date, columns=assets)
            returns: Asset returns (index=date, columns=assets)
            initial_capital: Starting capital
            
        Returns:
            BacktestResult
        """
        # Align data
        common_dates = weights.index.intersection(returns.index)
        weights = weights.loc[common_dates]
        returns = returns.loc[common_dates]
        
        # Calculate portfolio returns
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
        
        # Calculate turnover and transaction costs
        turnover = weights.diff().abs().sum(axis=1)
        transaction_costs = turnover * (self.commission + self.slippage)
        
        # Net returns
        net_returns = portfolio_returns - transaction_costs
        
        # Calculate portfolio values
        portfolio_values = initial_capital * (1 + net_returns).cumprod()
        
        # Create config
        config = BacktestConfig(
            start_date=common_dates[0],
            end_date=common_dates[-1],
            initial_capital=initial_capital,
            commission=self.commission,
            slippage=self.slippage,
        )
        
        return BacktestResult(
            config=config,
            portfolio_values=portfolio_values,
            returns=net_returns,
            trades=[],  # No individual trades in vectorized backtest
            positions={},
            weights_history=weights,
        )
