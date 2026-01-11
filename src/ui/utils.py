"""Utility functions for Streamlit dashboard."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any


def generate_sample_prices(
    symbols: list[str],
    periods: int = 252,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Generate sample price data for demo.
    
    Args:
        symbols: List of symbols
        periods: Number of periods
        start_date: Start date
        
    Returns:
        DataFrame with price data
    """
    np.random.seed(42)
    dates = pd.date_range(start_date, periods=periods, freq="B")
    
    data = {}
    for symbol in symbols:
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, periods)
        prices = 100 * np.cumprod(1 + returns)
        data[symbol] = prices
    
    return pd.DataFrame(data, index=dates)


def generate_sample_returns(
    symbols: list[str],
    periods: int = 252,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Generate sample return data for demo."""
    prices = generate_sample_prices(symbols, periods, start_date)
    return prices.pct_change().dropna()


def plot_price_chart(
    prices: pd.DataFrame,
    title: str = "Price History",
) -> go.Figure:
    """Create price chart with Plotly.
    
    Args:
        prices: Price DataFrame
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for col in prices.columns:
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices[col],
            name=col,
            mode="lines",
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
) -> go.Figure:
    """Create returns histogram."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Returns",
    ))
    
    # Add normal distribution overlay
    x_range = np.linspace(returns.min(), returns.max(), 100)
    from scipy import stats
    normal = stats.norm.pdf(x_range, returns.mean(), returns.std())
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal * len(returns) * (returns.max() - returns.min()) / 50,
        name="Normal Dist",
        line=dict(color="red", dash="dash"),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Return",
        yaxis_title="Frequency",
    )
    
    return fig


def plot_drawdown(
    portfolio_values: pd.Series,
    title: str = "Drawdown",
) -> go.Figure:
    """Create drawdown chart."""
    # Calculate drawdown
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="red"),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis=dict(autorange="reversed"),
    )
    
    return fig


def plot_allocation_pie(
    weights: dict[str, float],
    title: str = "Asset Allocation",
) -> go.Figure:
    """Create allocation pie chart."""
    labels = list(weights.keys())
    values = list(weights.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo="label+percent",
    )])
    
    fig.update_layout(title=title)
    
    return fig


def plot_performance_comparison(
    strategy_values: pd.Series,
    benchmark_values: pd.Series,
    title: str = "Strategy vs Benchmark",
) -> go.Figure:
    """Create performance comparison chart."""
    fig = go.Figure()
    
    # Normalize to starting value
    strategy_norm = strategy_values / strategy_values.iloc[0] * 100
    benchmark_norm = benchmark_values / benchmark_values.iloc[0] * 100
    
    fig.add_trace(go.Scatter(
        x=strategy_norm.index,
        y=strategy_norm,
        name="Strategy",
        line=dict(color="blue"),
    ))
    
    fig.add_trace(go.Scatter(
        x=benchmark_norm.index,
        y=benchmark_norm,
        name="Benchmark",
        line=dict(color="gray", dash="dash"),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value (Indexed to 100)",
        hovermode="x unified",
    )
    
    return fig


def plot_factor_exposure(
    exposures: dict[str, float],
    title: str = "Factor Exposure",
) -> go.Figure:
    """Create factor exposure bar chart."""
    factors = list(exposures.keys())
    values = list(exposures.values())
    colors = ["green" if v > 0 else "red" for v in values]
    
    fig = go.Figure(data=[go.Bar(
        x=factors,
        y=values,
        marker_color=colors,
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title="Factor",
        yaxis_title="Exposure (Beta)",
    )
    
    return fig


def plot_regime_timeline(
    regimes: pd.Series,
    title: str = "Market Regime History",
) -> go.Figure:
    """Create regime timeline chart."""
    regime_colors = {
        "Expansion": "green",
        "Peak": "yellow",
        "Contraction": "red",
        "Trough": "orange",
        "Unknown": "gray",
    }
    
    fig = go.Figure()
    
    for regime, color in regime_colors.items():
        mask = regimes == regime
        if mask.any():
            fig.add_trace(go.Scatter(
                x=regimes[mask].index,
                y=[regime] * mask.sum(),
                mode="markers",
                name=regime,
                marker=dict(color=color, size=10),
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Regime",
    )
    
    return fig


def format_metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Format metrics as a display table."""
    formatted = []
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if "return" in key.lower() or "rate" in key.lower():
                formatted.append({"Metric": key, "Value": f"{value:.2%}"})
            elif "ratio" in key.lower():
                formatted.append({"Metric": key, "Value": f"{value:.2f}"})
            else:
                formatted.append({"Metric": key, "Value": f"{value:.4f}"})
        else:
            formatted.append({"Metric": key, "Value": str(value)})
    
    return pd.DataFrame(formatted)


def create_backtest_summary(
    returns: pd.Series,
    portfolio_values: pd.Series,
) -> dict[str, Any]:
    """Create backtest summary metrics."""
    from src.backtest.metrics import calculate_metrics
    
    metrics = calculate_metrics(returns, portfolio_values)
    
    return {
        "Total Return": metrics.total_return,
        "CAGR": metrics.cagr,
        "Volatility": metrics.volatility,
        "Sharpe Ratio": metrics.sharpe_ratio,
        "Sortino Ratio": metrics.sortino_ratio,
        "Max Drawdown": metrics.max_drawdown,
        "Win Rate": metrics.win_rate,
        "VaR (95%)": metrics.var_95,
    }
