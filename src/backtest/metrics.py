"""Performance metrics for backtesting."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PerformanceMetrics:
    """Performance metrics container.
    
    Attributes:
        total_return: Total cumulative return
        cagr: Compound annual growth rate
        volatility: Annualized volatility
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        calmar_ratio: Calmar ratio
        max_drawdown: Maximum drawdown
        max_drawdown_duration: Maximum drawdown duration (days)
        win_rate: Percentage of positive returns
        profit_factor: Gross profit / gross loss
        var_95: Value at Risk (95%)
        cvar_95: Conditional VaR (95%)
        skewness: Return distribution skewness
        kurtosis: Return distribution kurtosis
        best_day: Best daily return
        worst_day: Worst daily return
        avg_win: Average winning return
        avg_loss: Average losing return
        num_trades: Number of trades
        turnover: Average portfolio turnover
    """
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    num_trades: int = 0
    turnover: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_return": f"{self.total_return:.2%}",
            "cagr": f"{self.cagr:.2%}",
            "volatility": f"{self.volatility:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
            "calmar_ratio": f"{self.calmar_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "max_drawdown_duration": f"{self.max_drawdown_duration} days",
            "win_rate": f"{self.win_rate:.2%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "var_95": f"{self.var_95:.2%}",
            "cvar_95": f"{self.cvar_95:.2%}",
            "skewness": f"{self.skewness:.2f}",
            "kurtosis": f"{self.kurtosis:.2f}",
            "best_day": f"{self.best_day:.2%}",
            "worst_day": f"{self.worst_day:.2%}",
            "avg_win": f"{self.avg_win:.2%}",
            "avg_loss": f"{self.avg_loss:.2%}",
            "num_trades": self.num_trades,
            "turnover": f"{self.turnover:.2%}",
        }
    
    def summary(self) -> str:
        """Generate text summary."""
        return f"""
Performance Summary
==================
Total Return:   {self.total_return:>10.2%}
CAGR:           {self.cagr:>10.2%}
Volatility:     {self.volatility:>10.2%}
Sharpe Ratio:   {self.sharpe_ratio:>10.2f}
Sortino Ratio:  {self.sortino_ratio:>10.2f}
Calmar Ratio:   {self.calmar_ratio:>10.2f}
Max Drawdown:   {self.max_drawdown:>10.2%}
Win Rate:       {self.win_rate:>10.2%}
VaR (95%):      {self.var_95:>10.2%}
"""


def calculate_total_return(
    portfolio_values: pd.Series,
) -> float:
    """Calculate total return.
    
    Args:
        portfolio_values: Portfolio value time series
        
    Returns:
        Total return (decimal)
    """
    if len(portfolio_values) < 2:
        return 0.0
    return portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1


def calculate_cagr(
    portfolio_values: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate Compound Annual Growth Rate.
    
    Args:
        portfolio_values: Portfolio value time series
        periods_per_year: Number of periods per year
        
    Returns:
        CAGR (decimal)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0]
    n_periods = len(portfolio_values)
    years = n_periods / periods_per_year
    
    if years <= 0:
        return 0.0
    
    return total_return ** (1 / years) - 1


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility.
    
    Args:
        returns: Return time series
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Return time series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    vol = returns.std()
    
    if vol == 0:
        return 0.0
    
    return (excess_returns.mean() / vol) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino ratio.
    
    Uses downside deviation instead of standard deviation.
    
    Args:
        returns: Return time series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0
    
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return 0.0
    
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)


def calculate_max_drawdown(
    portfolio_values: pd.Series,
) -> tuple[float, int]:
    """Calculate maximum drawdown and duration.
    
    Args:
        portfolio_values: Portfolio value time series
        
    Returns:
        Tuple of (max drawdown, max drawdown duration in days)
    """
    if len(portfolio_values) < 2:
        return 0.0, 0
    
    # Calculate running maximum
    running_max = portfolio_values.cummax()
    
    # Calculate drawdown series
    drawdown = (portfolio_values - running_max) / running_max
    
    # Max drawdown
    max_dd = abs(drawdown.min())
    
    # Max drawdown duration
    underwater = drawdown < 0
    
    # Find consecutive underwater periods
    max_duration = 0
    current_duration = 0
    
    for is_underwater in underwater:
        if is_underwater:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_dd, max_duration


def calculate_calmar_ratio(
    returns: pd.Series,
    portfolio_values: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate Calmar ratio.
    
    CAGR divided by maximum drawdown.
    
    Args:
        returns: Return time series
        portfolio_values: Portfolio value time series
        periods_per_year: Number of periods per year
        
    Returns:
        Calmar ratio
    """
    cagr = calculate_cagr(portfolio_values, periods_per_year)
    max_dd, _ = calculate_max_drawdown(portfolio_values)
    
    if max_dd == 0:
        return 0.0
    
    return cagr / max_dd


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Calculate Value at Risk.
    
    Args:
        returns: Return time series
        confidence: Confidence level
        method: VaR method (historical or parametric)
        
    Returns:
        VaR (positive number representing potential loss)
    """
    if len(returns) < 2:
        return 0.0
    
    if method == "historical":
        var = np.percentile(returns, (1 - confidence) * 100)
    else:
        # Parametric (assumes normal distribution)
        mean = returns.mean()
        std = returns.std()
        var = mean + std * stats.norm.ppf(1 - confidence)
    
    return abs(var)


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Calculate Conditional VaR (Expected Shortfall).
    
    Average loss beyond VaR threshold.
    
    Args:
        returns: Return time series
        confidence: Confidence level
        
    Returns:
        CVaR (positive number)
    """
    if len(returns) < 2:
        return 0.0
    
    var = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= var]
    
    if len(tail_losses) == 0:
        return abs(var)
    
    return abs(tail_losses.mean())


def calculate_win_rate(
    returns: pd.Series,
) -> float:
    """Calculate win rate (percentage of positive returns).
    
    Args:
        returns: Return time series
        
    Returns:
        Win rate (0-1)
    """
    if len(returns) == 0:
        return 0.0
    
    return (returns > 0).sum() / len(returns)


def calculate_profit_factor(
    returns: pd.Series,
) -> float:
    """Calculate profit factor.
    
    Gross profit divided by gross loss.
    
    Args:
        returns: Return time series
        
    Returns:
        Profit factor
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    
    return gains / losses


def calculate_turnover(
    weights: pd.DataFrame,
) -> float:
    """Calculate average portfolio turnover.
    
    Args:
        weights: Portfolio weights over time
        
    Returns:
        Average turnover per period
    """
    if weights is None or len(weights) < 2:
        return 0.0
    
    # Calculate absolute weight changes
    weight_changes = weights.diff().abs().sum(axis=1)
    
    return weight_changes.mean()


def calculate_metrics(
    returns: pd.Series,
    portfolio_values: pd.Series,
    weights: pd.DataFrame | None = None,
    num_trades: int = 0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """Calculate all performance metrics.
    
    Args:
        returns: Return time series
        portfolio_values: Portfolio value time series
        weights: Portfolio weights over time
        num_trades: Number of trades executed
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        PerformanceMetrics object
    """
    max_dd, max_dd_duration = calculate_max_drawdown(portfolio_values)
    
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    
    return PerformanceMetrics(
        total_return=calculate_total_return(portfolio_values),
        cagr=calculate_cagr(portfolio_values, periods_per_year),
        volatility=calculate_volatility(returns, periods_per_year),
        sharpe_ratio=calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        sortino_ratio=calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        calmar_ratio=calculate_calmar_ratio(returns, portfolio_values, periods_per_year),
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        win_rate=calculate_win_rate(returns),
        profit_factor=calculate_profit_factor(returns),
        var_95=calculate_var(returns, 0.95),
        cvar_95=calculate_cvar(returns, 0.95),
        skewness=float(stats.skew(returns)) if len(returns) > 2 else 0.0,
        kurtosis=float(stats.kurtosis(returns)) if len(returns) > 2 else 0.0,
        best_day=float(returns.max()) if len(returns) > 0 else 0.0,
        worst_day=float(returns.min()) if len(returns) > 0 else 0.0,
        avg_win=float(winning_returns.mean()) if len(winning_returns) > 0 else 0.0,
        avg_loss=float(losing_returns.mean()) if len(losing_returns) > 0 else 0.0,
        num_trades=num_trades,
        turnover=calculate_turnover(weights),
    )


class BenchmarkComparison:
    """Compare strategy performance against benchmark.
    
    Example:
        >>> comparison = BenchmarkComparison(strategy_returns, benchmark_returns)
        >>> print(f"Alpha: {comparison.alpha:.2%}")
        >>> print(f"Beta: {comparison.beta:.2f}")
    """
    
    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """Initialize benchmark comparison.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        # Align returns
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        self.strategy_returns = strategy_returns.loc[common_idx]
        self.benchmark_returns = benchmark_returns.loc[common_idx]
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calculate comparison metrics."""
        # Excess returns
        rf_daily = self.risk_free_rate / self.periods_per_year
        excess_strategy = self.strategy_returns - rf_daily
        excess_benchmark = self.benchmark_returns - rf_daily
        
        # Beta (regression coefficient)
        if len(excess_benchmark) > 2:
            cov = np.cov(excess_strategy, excess_benchmark)[0, 1]
            var_benchmark = excess_benchmark.var()
            self.beta = cov / var_benchmark if var_benchmark > 0 else 0.0
        else:
            self.beta = 0.0
        
        # Alpha (Jensen's alpha, annualized)
        strategy_mean = excess_strategy.mean() * self.periods_per_year
        benchmark_mean = excess_benchmark.mean() * self.periods_per_year
        self.alpha = strategy_mean - self.beta * benchmark_mean
        
        # Correlation
        if len(self.strategy_returns) > 2:
            self.correlation = self.strategy_returns.corr(self.benchmark_returns)
        else:
            self.correlation = 0.0
        
        # Tracking error
        tracking_diff = self.strategy_returns - self.benchmark_returns
        self.tracking_error = tracking_diff.std() * np.sqrt(self.periods_per_year)
        
        # Information ratio
        if self.tracking_error > 0:
            active_return = (self.strategy_returns.mean() - self.benchmark_returns.mean())
            self.information_ratio = (
                active_return * self.periods_per_year / self.tracking_error
            )
        else:
            self.information_ratio = 0.0
        
        # Up/Down capture
        up_periods = self.benchmark_returns > 0
        down_periods = self.benchmark_returns < 0
        
        if up_periods.sum() > 0:
            self.up_capture = (
                self.strategy_returns[up_periods].mean() / 
                self.benchmark_returns[up_periods].mean()
            )
        else:
            self.up_capture = 0.0
        
        if down_periods.sum() > 0:
            self.down_capture = (
                self.strategy_returns[down_periods].mean() / 
                self.benchmark_returns[down_periods].mean()
            )
        else:
            self.down_capture = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alpha": f"{self.alpha:.2%}",
            "beta": f"{self.beta:.2f}",
            "correlation": f"{self.correlation:.2f}",
            "tracking_error": f"{self.tracking_error:.2%}",
            "information_ratio": f"{self.information_ratio:.2f}",
            "up_capture": f"{self.up_capture:.2%}",
            "down_capture": f"{self.down_capture:.2%}",
        }
    
    def summary(self) -> str:
        """Generate text summary."""
        return f"""
Benchmark Comparison
===================
Alpha:             {self.alpha:>10.2%}
Beta:              {self.beta:>10.2f}
Correlation:       {self.correlation:>10.2f}
Tracking Error:    {self.tracking_error:>10.2%}
Information Ratio: {self.information_ratio:>10.2f}
Up Capture:        {self.up_capture:>10.2%}
Down Capture:      {self.down_capture:>10.2%}
"""
