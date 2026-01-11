"""Walk-forward analysis for strategy evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any

import numpy as np
import pandas as pd

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .metrics import calculate_metrics, PerformanceMetrics


@dataclass
class WalkForwardConfig:
    """Walk-forward analysis configuration.
    
    Attributes:
        train_period: Training period length (e.g., "2Y" for 2 years)
        test_period: Test period length (e.g., "6M" for 6 months)
        step_size: Step size between windows (e.g., "3M" for 3 months)
        min_train_samples: Minimum training samples required
        initial_capital: Starting capital for each test
        commission: Commission rate
    """
    train_period: str = "2Y"
    test_period: str = "6M"
    step_size: str = "3M"
    min_train_samples: int = 252
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    
    def parse_period(self, period: str) -> pd.DateOffset:
        """Parse period string to DateOffset.
        
        Args:
            period: Period string (e.g., "2Y", "6M", "3M")
            
        Returns:
            pandas DateOffset
        """
        num = int(period[:-1])
        unit = period[-1].upper()
        
        if unit == "Y":
            return pd.DateOffset(years=num)
        elif unit == "M":
            return pd.DateOffset(months=num)
        elif unit == "W":
            return pd.DateOffset(weeks=num)
        elif unit == "D":
            return pd.DateOffset(days=num)
        else:
            raise ValueError(f"Unknown period unit: {unit}")


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result.
    
    Attributes:
        train_start: Training period start
        train_end: Training period end
        test_start: Test period start
        test_end: Test period end
        train_metrics: Training period metrics
        test_metrics: Test period metrics
        optimal_params: Optimal parameters from training
    """
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: PerformanceMetrics | None = None
    test_metrics: PerformanceMetrics | None = None
    optimal_params: dict | None = None


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result.
    
    Attributes:
        config: Analysis configuration
        windows: Individual window results
        combined_test_returns: Combined OOS returns
        combined_metrics: Combined OOS metrics
        efficiency_ratio: OOS/IS performance ratio
        parameter_stability: Parameter stability score
    """
    config: WalkForwardConfig
    windows: list[WalkForwardWindow]
    combined_test_returns: pd.Series | None = None
    combined_metrics: PerformanceMetrics | None = None
    efficiency_ratio: float = 0.0
    parameter_stability: float = 0.0
    
    def summary(self) -> str:
        """Generate text summary."""
        if self.combined_metrics is None:
            return "No results available"
        
        return f"""
Walk-Forward Analysis Summary
============================
Number of Windows:    {len(self.windows)}
Efficiency Ratio:     {self.efficiency_ratio:.2%}
Parameter Stability:  {self.parameter_stability:.2f}

Combined Out-of-Sample Performance:
{self.combined_metrics.summary()}
"""


class WalkForwardAnalyzer:
    """Walk-forward analysis for strategy validation.
    
    Implements rolling window optimization and out-of-sample testing
    to evaluate strategy robustness.
    
    Example:
        >>> config = WalkForwardConfig(train_period="2Y", test_period="6M")
        >>> analyzer = WalkForwardAnalyzer(config)
        >>> result = analyzer.run(strategy_factory, prices)
    """
    
    def __init__(self, config: WalkForwardConfig):
        """Initialize analyzer.
        
        Args:
            config: Walk-forward configuration
        """
        self.config = config
    
    def run(
        self,
        strategy_factory: Callable[..., Any],
        prices: pd.DataFrame,
        optimization_func: Callable | None = None,
        param_grid: dict | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward analysis.
        
        Args:
            strategy_factory: Function to create strategy with params
            prices: Price data (index=date, columns=assets)
            optimization_func: Function to optimize strategy parameters
            param_grid: Parameter grid for optimization
            
        Returns:
            WalkForwardResult
        """
        windows = self._generate_windows(prices.index)
        
        all_test_returns = []
        
        for window in windows:
            # Get training data
            train_data = prices.loc[window.train_start:window.train_end]
            test_data = prices.loc[window.test_start:window.test_end]
            
            if len(train_data) < self.config.min_train_samples:
                continue
            
            # Optimize on training data
            if optimization_func is not None and param_grid is not None:
                optimal_params = optimization_func(
                    strategy_factory, train_data, param_grid
                )
                window.optimal_params = optimal_params
                strategy = strategy_factory(**optimal_params)
            else:
                window.optimal_params = {}
                strategy = strategy_factory()
            
            # Run training backtest
            train_config = BacktestConfig(
                start_date=window.train_start,
                end_date=window.train_end,
                initial_capital=self.config.initial_capital,
                commission=self.config.commission,
                slippage=self.config.slippage,
            )
            train_engine = BacktestEngine(train_config)
            train_result = train_engine.run(strategy, train_data)
            window.train_metrics = calculate_metrics(
                train_result.returns,
                train_result.portfolio_values,
                train_result.weights_history,
                len(train_result.trades),
            )
            
            # Run test backtest (out-of-sample)
            test_config = BacktestConfig(
                start_date=window.test_start,
                end_date=window.test_end,
                initial_capital=self.config.initial_capital,
                commission=self.config.commission,
                slippage=self.config.slippage,
            )
            test_engine = BacktestEngine(test_config)
            test_result = test_engine.run(strategy, test_data)
            window.test_metrics = calculate_metrics(
                test_result.returns,
                test_result.portfolio_values,
                test_result.weights_history,
                len(test_result.trades),
            )
            
            all_test_returns.append(test_result.returns)
        
        # Combine test returns
        if all_test_returns:
            combined_returns = pd.concat(all_test_returns)
            combined_returns = combined_returns.sort_index()
            
            # Calculate combined portfolio values
            combined_values = self.config.initial_capital * (1 + combined_returns).cumprod()
            
            combined_metrics = calculate_metrics(
                combined_returns,
                combined_values,
                num_trades=sum(
                    w.test_metrics.num_trades for w in windows 
                    if w.test_metrics is not None
                ),
            )
        else:
            combined_returns = None
            combined_metrics = None
        
        # Calculate efficiency ratio (OOS / IS performance)
        efficiency_ratio = self._calculate_efficiency_ratio(windows)
        
        # Calculate parameter stability
        param_stability = self._calculate_parameter_stability(windows)
        
        return WalkForwardResult(
            config=self.config,
            windows=windows,
            combined_test_returns=combined_returns,
            combined_metrics=combined_metrics,
            efficiency_ratio=efficiency_ratio,
            parameter_stability=param_stability,
        )
    
    def _generate_windows(
        self,
        dates: pd.DatetimeIndex,
    ) -> list[WalkForwardWindow]:
        """Generate walk-forward windows.
        
        Args:
            dates: Available dates
            
        Returns:
            List of WalkForwardWindow objects
        """
        windows = []
        
        train_offset = self.config.parse_period(self.config.train_period)
        test_offset = self.config.parse_period(self.config.test_period)
        step_offset = self.config.parse_period(self.config.step_size)
        
        start_date = dates[0]
        end_date = dates[-1]
        
        current_start = start_date
        
        while True:
            train_start = current_start
            train_end = train_start + train_offset
            test_start = train_end
            test_end = test_start + test_offset
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            # Find actual dates in the index
            train_start_actual = dates[dates >= train_start].min()
            train_end_actual = dates[dates <= train_end].max()
            test_start_actual = dates[dates >= test_start].min()
            test_end_actual = dates[dates <= test_end].max()
            
            if pd.isna(train_start_actual) or pd.isna(test_end_actual):
                current_start = current_start + step_offset
                continue
            
            windows.append(WalkForwardWindow(
                train_start=train_start_actual,
                train_end=train_end_actual,
                test_start=test_start_actual,
                test_end=test_end_actual,
            ))
            
            current_start = current_start + step_offset
        
        return windows
    
    def _calculate_efficiency_ratio(
        self,
        windows: list[WalkForwardWindow],
    ) -> float:
        """Calculate efficiency ratio (OOS/IS performance).
        
        Args:
            windows: Walk-forward windows
            
        Returns:
            Efficiency ratio
        """
        is_sharpes = []
        oos_sharpes = []
        
        for window in windows:
            if window.train_metrics and window.test_metrics:
                is_sharpes.append(window.train_metrics.sharpe_ratio)
                oos_sharpes.append(window.test_metrics.sharpe_ratio)
        
        if not is_sharpes or sum(is_sharpes) == 0:
            return 0.0
        
        avg_is = np.mean(is_sharpes)
        avg_oos = np.mean(oos_sharpes)
        
        if avg_is == 0:
            return 0.0
        
        return avg_oos / avg_is
    
    def _calculate_parameter_stability(
        self,
        windows: list[WalkForwardWindow],
    ) -> float:
        """Calculate parameter stability across windows.
        
        Lower coefficient of variation = more stable.
        
        Args:
            windows: Walk-forward windows
            
        Returns:
            Parameter stability score (1 - avg CV)
        """
        if len(windows) < 2:
            return 0.0
        
        # Collect parameters
        all_params = [w.optimal_params for w in windows if w.optimal_params]
        
        if not all_params:
            return 1.0  # No parameters to vary
        
        # Get numeric parameters
        param_keys = set()
        for params in all_params:
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    param_keys.add(k)
        
        if not param_keys:
            return 1.0
        
        # Calculate coefficient of variation for each parameter
        cvs = []
        for key in param_keys:
            values = [p.get(key) for p in all_params if key in p]
            if len(values) > 1:
                mean = np.mean(values)
                std = np.std(values)
                if mean != 0:
                    cvs.append(std / abs(mean))
        
        if not cvs:
            return 1.0
        
        # Stability = 1 - average CV (capped at 0)
        return max(0, 1 - np.mean(cvs))


def grid_search_optimizer(
    strategy_factory: Callable,
    prices: pd.DataFrame,
    param_grid: dict,
    metric: str = "sharpe_ratio",
    commission: float = 0.001,
) -> dict:
    """Grid search optimizer for strategy parameters.
    
    Args:
        strategy_factory: Function to create strategy
        prices: Price data
        param_grid: Parameter grid (dict of lists)
        metric: Metric to optimize
        commission: Commission rate
        
    Returns:
        Optimal parameters
    """
    from itertools import product
    
    # Generate all parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    best_metric = float('-inf')
    best_params = combinations[0] if combinations else {}
    
    for params in combinations:
        try:
            strategy = strategy_factory(**params)
            
            config = BacktestConfig(
                start_date=prices.index[0],
                end_date=prices.index[-1],
                initial_capital=100000,
                commission=commission,
            )
            engine = BacktestEngine(config)
            result = engine.run(strategy, prices)
            
            metrics = calculate_metrics(
                result.returns,
                result.portfolio_values,
            )
            
            metric_value = getattr(metrics, metric, 0)
            
            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params
        except Exception:
            continue
    
    return best_params


class AnchoredWalkForward:
    """Anchored walk-forward analysis.
    
    Unlike rolling walk-forward, this keeps the start date fixed
    and only extends the training window.
    
    Example:
        >>> analyzer = AnchoredWalkForward(test_period="6M")
        >>> result = analyzer.run(strategy_factory, prices)
    """
    
    def __init__(
        self,
        test_period: str = "6M",
        min_train_period: str = "1Y",
        step_size: str = "3M",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
    ):
        """Initialize anchored analyzer.
        
        Args:
            test_period: Test period length
            min_train_period: Minimum training period
            step_size: Step size between windows
            initial_capital: Starting capital
            commission: Commission rate
        """
        self.test_period = test_period
        self.min_train_period = min_train_period
        self.step_size = step_size
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run(
        self,
        strategy_factory: Callable[..., Any],
        prices: pd.DataFrame,
    ) -> WalkForwardResult:
        """Run anchored walk-forward analysis.
        
        Args:
            strategy_factory: Function to create strategy
            prices: Price data
            
        Returns:
            WalkForwardResult
        """
        config = WalkForwardConfig(
            train_period=self.min_train_period,
            test_period=self.test_period,
            step_size=self.step_size,
            initial_capital=self.initial_capital,
            commission=self.commission,
        )
        
        windows = []
        all_test_returns = []
        
        # Parse periods
        min_train_offset = config.parse_period(self.min_train_period)
        test_offset = config.parse_period(self.test_period)
        step_offset = config.parse_period(self.step_size)
        
        anchor_date = prices.index[0]
        end_date = prices.index[-1]
        
        # Start after minimum training period
        current_test_start = anchor_date + min_train_offset
        
        while current_test_start + test_offset <= end_date:
            train_end = current_test_start
            test_end = current_test_start + test_offset
            
            # Find actual dates
            train_data = prices.loc[anchor_date:train_end]
            test_data = prices.loc[current_test_start:test_end]
            
            if len(train_data) < 100 or len(test_data) < 10:
                current_test_start = current_test_start + step_offset
                continue
            
            window = WalkForwardWindow(
                train_start=anchor_date,
                train_end=train_end,
                test_start=current_test_start,
                test_end=test_end,
            )
            
            # Create and run strategy
            strategy = strategy_factory()
            
            # Test backtest
            test_config = BacktestConfig(
                start_date=current_test_start,
                end_date=test_end,
                initial_capital=self.initial_capital,
                commission=self.commission,
            )
            test_engine = BacktestEngine(test_config)
            test_result = test_engine.run(strategy, prices.loc[:test_end])
            
            window.test_metrics = calculate_metrics(
                test_result.returns,
                test_result.portfolio_values,
            )
            
            windows.append(window)
            all_test_returns.append(test_result.returns)
            
            current_test_start = current_test_start + step_offset
        
        # Combine results
        if all_test_returns:
            combined_returns = pd.concat(all_test_returns).sort_index()
            # Remove duplicates, keep last
            combined_returns = combined_returns[~combined_returns.index.duplicated(keep='last')]
            combined_values = self.initial_capital * (1 + combined_returns).cumprod()
            combined_metrics = calculate_metrics(combined_returns, combined_values)
        else:
            combined_returns = None
            combined_metrics = None
        
        return WalkForwardResult(
            config=config,
            windows=windows,
            combined_test_returns=combined_returns,
            combined_metrics=combined_metrics,
        )
