"""Utility functions for Streamlit dashboard with dark theme support."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any


# Dark theme color palette (Claude-inspired)
COLORS = {
    "primary": "#8b5cf6",      # Purple
    "secondary": "#6366f1",    # Indigo
    "success": "#10b981",      # Green
    "warning": "#f59e0b",      # Amber
    "danger": "#ef4444",       # Red
    "info": "#3b82f6",         # Blue
    "background": "rgba(0,0,0,0)",
    "surface": "rgba(255,255,255,0.02)",
    "text": "#ffffff",
    "text_secondary": "#a0a0a0",
    "grid": "rgba(255,255,255,0.05)",
}

# Chart color sequence
COLOR_SEQUENCE = [
    "#8b5cf6",  # Purple
    "#10b981",  # Green
    "#3b82f6",  # Blue
    "#f59e0b",  # Amber
    "#ef4444",  # Red
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
    "#84cc16",  # Lime
]


def get_dark_layout(title: str = "") -> dict:
    """Get common dark theme layout for Plotly charts."""
    return {
        "title": {"text": title, "font": {"color": COLORS["text"], "size": 16}} if title else None,
        "paper_bgcolor": COLORS["background"],
        "plot_bgcolor": COLORS["background"],
        "font": {"family": "Inter, sans-serif", "color": COLORS["text_secondary"]},
        "xaxis": {
            "showgrid": True,
            "gridcolor": COLORS["grid"],
            "linecolor": COLORS["grid"],
            "tickfont": {"color": COLORS["text_secondary"]},
            "title_font": {"color": COLORS["text_secondary"]},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": COLORS["grid"],
            "linecolor": COLORS["grid"],
            "tickfont": {"color": COLORS["text_secondary"]},
            "title_font": {"color": COLORS["text_secondary"]},
        },
        "legend": {
            "bgcolor": COLORS["background"],
            "font": {"color": COLORS["text_secondary"]},
            "bordercolor": COLORS["grid"],
            "borderwidth": 1,
        },
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "rgba(30, 30, 63, 0.95)",
            "font_size": 12,
            "font_family": "Inter, sans-serif",
        },
        "margin": {"l": 50, "r": 20, "t": 30, "b": 50},
    }


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
    title: str = "",
) -> go.Figure:
    """Create price chart with Plotly (dark theme).

    Args:
        prices: Price DataFrame
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for i, col in enumerate(prices.columns):
        color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices[col],
            name=col,
            mode="lines",
            line={"color": color, "width": 2},
            hovertemplate=f"<b>{col}</b><br>%{{y:,.2f}}<extra></extra>",
        ))

    layout = get_dark_layout(title)
    layout["xaxis"]["title"] = ""
    layout["yaxis"]["title"] = ""
    layout["legend"]["yanchor"] = "top"
    layout["legend"]["y"] = 0.99
    layout["legend"]["xanchor"] = "left"
    layout["legend"]["x"] = 0.01

    fig.update_layout(**layout)

    return fig


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "",
) -> go.Figure:
    """Create returns histogram (dark theme)."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Returns",
        marker_color=COLORS["primary"],
        opacity=0.8,
        hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
    ))

    # Add normal distribution overlay
    x_range = np.linspace(returns.min(), returns.max(), 100)
    from scipy import stats
    normal = stats.norm.pdf(x_range, returns.mean(), returns.std())

    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal * len(returns) * (returns.max() - returns.min()) / 50,
        name="Normal Dist",
        line={"color": COLORS["warning"], "dash": "dash", "width": 2},
        hoverinfo="skip",
    ))

    # Add mean line
    fig.add_vline(
        x=returns.mean(),
        line={"color": COLORS["success"], "dash": "dot", "width": 2},
        annotation_text=f"Mean: {returns.mean():.2%}",
        annotation_position="top",
        annotation_font_color=COLORS["success"],
    )

    layout = get_dark_layout(title)
    layout["xaxis"]["title"] = "Return"
    layout["yaxis"]["title"] = "Frequency"
    layout["xaxis"]["tickformat"] = ".1%"
    layout["bargap"] = 0.1

    fig.update_layout(**layout)

    return fig


def plot_drawdown(
    portfolio_values: pd.Series,
    title: str = "",
) -> go.Figure:
    """Create drawdown chart (dark theme)."""
    # Calculate drawdown
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        fill="tozeroy",
        name="Drawdown",
        line={"color": COLORS["danger"], "width": 1},
        fillcolor=f"rgba({int(COLORS['danger'][1:3], 16)}, {int(COLORS['danger'][3:5], 16)}, {int(COLORS['danger'][5:7], 16)}, 0.3)",
        hovertemplate="Date: %{x}<br>Drawdown: %{y:.1f}%<extra></extra>",
    ))

    # Add max drawdown annotation
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()

    fig.add_annotation(
        x=max_dd_idx,
        y=max_dd_val,
        text=f"Max DD: {max_dd_val:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["danger"],
        font={"color": COLORS["danger"], "size": 11},
        bgcolor="rgba(30, 30, 63, 0.9)",
        bordercolor=COLORS["danger"],
    )

    layout = get_dark_layout(title)
    layout["xaxis"]["title"] = ""
    layout["yaxis"]["title"] = "Drawdown (%)"
    layout["yaxis"]["autorange"] = "reversed"

    fig.update_layout(**layout)

    return fig


def plot_allocation_pie(
    weights: dict[str, float],
    title: str = "",
) -> go.Figure:
    """Create allocation pie/donut chart (dark theme)."""
    labels = list(weights.keys())
    values = list(weights.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        textinfo="label+percent",
        textposition="outside",
        textfont={"color": COLORS["text_secondary"], "size": 11},
        marker={
            "colors": COLOR_SEQUENCE[:len(labels)],
            "line": {"color": "rgba(0,0,0,0.3)", "width": 1},
        },
        hovertemplate="<b>%{label}</b><br>%{percent}<br>$%{value:,.0f}<extra></extra>",
    )])

    layout = get_dark_layout(title)
    layout["showlegend"] = False
    layout["margin"] = {"l": 20, "r": 20, "t": 30, "b": 20}

    # Add center text
    fig.add_annotation(
        text="<b>Allocation</b>",
        x=0.5, y=0.5,
        font={"size": 14, "color": COLORS["text"]},
        showarrow=False,
    )

    fig.update_layout(**layout)

    return fig


def plot_performance_comparison(
    strategy_values: pd.Series,
    benchmark_values: pd.Series,
    title: str = "",
) -> go.Figure:
    """Create performance comparison chart (dark theme)."""
    fig = go.Figure()

    # Normalize to starting value
    strategy_norm = strategy_values / strategy_values.iloc[0] * 100
    benchmark_norm = benchmark_values / benchmark_values.iloc[0] * 100

    fig.add_trace(go.Scatter(
        x=strategy_norm.index,
        y=strategy_norm,
        name="Strategy",
        line={"color": COLORS["primary"], "width": 2},
        hovertemplate="<b>Strategy</b><br>%{y:.1f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=benchmark_norm.index,
        y=benchmark_norm,
        name="Benchmark",
        line={"color": COLORS["text_secondary"], "dash": "dash", "width": 2},
        hovertemplate="<b>Benchmark</b><br>%{y:.1f}<extra></extra>",
    ))

    # Add area between strategy and benchmark
    fig.add_trace(go.Scatter(
        x=list(strategy_norm.index) + list(benchmark_norm.index[::-1]),
        y=list(strategy_norm.values) + list(benchmark_norm.values[::-1]),
        fill="toself",
        fillcolor="rgba(139, 92, 246, 0.1)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=False,
        hoverinfo="skip",
    ))

    layout = get_dark_layout(title)
    layout["xaxis"]["title"] = ""
    layout["yaxis"]["title"] = "Value (Indexed to 100)"

    fig.update_layout(**layout)

    return fig


def plot_factor_exposure(
    exposures: dict[str, float],
    title: str = "",
) -> go.Figure:
    """Create factor exposure bar chart (dark theme)."""
    factors = list(exposures.keys())
    values = list(exposures.values())
    colors = [COLORS["success"] if v > 0 else COLORS["danger"] for v in values]

    fig = go.Figure(data=[go.Bar(
        x=factors,
        y=values,
        marker_color=colors,
        marker_line_color=colors,
        marker_line_width=1,
        opacity=0.8,
        hovertemplate="<b>%{x}</b><br>Beta: %{y:.2f}<extra></extra>",
    )])

    # Add zero line
    fig.add_hline(
        y=0,
        line={"color": COLORS["text_secondary"], "width": 1},
    )

    layout = get_dark_layout(title)
    layout["xaxis"]["title"] = ""
    layout["yaxis"]["title"] = "Exposure (Beta)"
    layout["bargap"] = 0.3

    fig.update_layout(**layout)

    return fig


def plot_regime_timeline(
    regimes: pd.Series,
    title: str = "",
) -> go.Figure:
    """Create regime timeline chart (dark theme)."""
    regime_colors = {
        "Expansion": COLORS["success"],
        "Peak": COLORS["warning"],
        "Contraction": COLORS["danger"],
        "Trough": COLORS["primary"],
        "Unknown": COLORS["text_secondary"],
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
                marker={"color": color, "size": 12, "symbol": "square"},
                hovertemplate=f"<b>{regime}</b><br>%{{x}}<extra></extra>",
            ))

    layout = get_dark_layout(title)
    layout["xaxis"]["title"] = ""
    layout["yaxis"]["title"] = ""

    fig.update_layout(**layout)

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
