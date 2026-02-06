"""Live Stock Analyzer - interactive ticker input and streaming analysis."""

import asyncio
import os
from functools import lru_cache
from pathlib import Path

from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import logging
from typing import Any
import yaml

from src.ui.services.realtime_hub import get_realtime_market_hub
from src.ui.services.paper_strategy_service import (
    build_prices_for_paper_run,
    latest_prices_for_symbols,
    run_paper_strategy_once,
    select_strategy_targets,
)

logger = logging.getLogger(__name__)


_BUILTIN_LIVE_WATCHLISTS: dict[str, list[str]] = {
    "us_mega": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO"],
    "semiconductor": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TSM", "ASML", "MU"],
    "etf_core": ["SPY", "QQQ", "IWM", "DIA", "VTI", "XLF", "XLK", "XLE"],
    "crypto_major": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    "mixed_macro": ["SPY", "QQQ", "TLT", "GLD", "BTCUSDT", "ETHUSDT", "DXY"],
}


def _format_watchlist_label(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


@lru_cache(maxsize=1)
def _load_live_watchlists() -> dict[str, list[str]]:
    """Load live watchlists from config + built-ins."""
    watchlists: dict[str, list[str]] = {
        key: list(symbols)
        for key, symbols in _BUILTIN_LIVE_WATCHLISTS.items()
    }

    config_path = Path("config/watchlist.yaml")
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                for key, value in loaded.items():
                    if not isinstance(value, list):
                        continue
                    symbols = [str(item).strip().upper() for item in value if str(item).strip()]
                    if symbols:
                        watchlists[str(key)] = symbols
        except Exception as exc:
            logger.warning("Failed to load watchlist config: %s", exc)

    return watchlists


def _watchlist_dropdown_options() -> list[dict[str, str]]:
    watchlists = _load_live_watchlists()
    return [
        {"label": _format_watchlist_label(name), "value": name}
        for name in sorted(watchlists.keys())
    ]


def _historical_request_timeout_sec() -> float:
    """Timeout bound for yfinance historical requests."""
    raw = os.getenv("AQ_YF_TIMEOUT_SEC", "8")
    try:
        timeout = float(raw)
    except ValueError:
        return 8.0
    return max(2.0, min(timeout, 30.0))


@lru_cache(maxsize=2)
def _load_ff5_factors_cached(frequency: str = "daily") -> pd.DataFrame:
    """Cache FF5 factors to avoid repeated network fetches per callback."""
    from src.factors.ff_data import FamaFrenchDataLoader

    ff_loader = FamaFrenchDataLoader()
    return ff_loader.load_ff5_factors(frequency=frequency)


def _dynamic_refresh_interval_ms(
    base_ms: int,
    analysis_type: str,
    resolution: str,
    ticker_count: int,
) -> int:
    """Adaptive refresh interval for lower latency and stable rendering."""
    interval = max(500, int(base_ms))

    if analysis_type in {"board", "candles", "screener"}:
        interval = min(interval, 1200)

    if resolution == "tick":
        interval = min(interval, 700)
    elif resolution == "1s":
        interval = min(interval, 950)
    elif resolution == "5s":
        interval = min(interval, 1400)
    else:
        interval = min(interval, 2200)

    # Protect browser render loop when many symbols are tracked.
    if ticker_count > 40:
        interval = max(interval, 2200)
    elif ticker_count > 24:
        interval = max(interval, 1700)
    elif ticker_count > 16:
        interval = max(interval, 1300)

    return max(500, min(interval, 5000))


def _limit_realtime_tickers(tickers: list[str], max_tickers: int = 48) -> list[str]:
    """Cap realtime subscription size to keep dashboard responsive."""
    deduped = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))
    return deduped[:max_tickers]


def _refresh_interval_ms() -> int:
    """Resolve live refresh interval from runtime environment."""
    raw = os.getenv("AQ_REFRESH_INTERVAL_MS", "3000")
    try:
        value = int(raw)
    except ValueError:
        return 3000
    return max(500, value)


def create_live_analyzer_layout() -> html.Div:
    """Create live analyzer page layout."""
    return html.Div([
        # Header
        html.Div([
            html.H2("Live Stock Analyzer", style={"color": "var(--text-primary)", "marginBottom": "0.5rem"}),
            html.P("Enter tickers to fetch historical/realtime data and run quant analysis "
                   "(US stocks realtime requires FINNHUB_API_KEY)",
                   style={"color": "var(--text-secondary)"}),
        ], style={"marginBottom": "1.5rem"}),

        # Input Section
        html.Div(
            className="chart-container",
            style={"marginBottom": "1.5rem", "position": "relative", "zIndex": 10, "overflow": "visible"},
            children=[
                html.Div(
                    className="row live-control-row",
                    children=[
                        # Ticker Input
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label("Tickers",
                                          style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                dcc.Input(
                                    id="live-ticker-input",
                                    type="text",
                                    value="AAPL MSFT NVDA GOOGL",
                                    placeholder="AAPL, MSFT, NVDA...",
                                    style={
                                        "width": "100%",
                                        "padding": "0.75rem",
                                        "marginTop": "0.5rem",
                                        "backgroundColor": "var(--bg-tertiary)",
                                        "border": "1px solid var(--border-color)",
                                        "borderRadius": "0.5rem",
                                        "color": "var(--text-primary)",
                                        "fontSize": "1rem",
                                    },
                                ),
                            ],
                        ),
                        # Data Mode
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label("Data Mode",
                                          style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                dcc.Dropdown(
                                    id="live-data-mode-dropdown",
                                    options=[
                                        {"label": "Historical (YFinance)", "value": "historical"},
                                        {"label": "Realtime Stream", "value": "realtime"},
                                    ],
                                    value="historical",
                                    clearable=False,
                                    style={"marginTop": "0.5rem"},
                                    className="dropdown-dark",
                                ),
                            ],
                        ),
                        # Interval
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label("Interval",
                                          style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                dcc.Dropdown(
                                    id="live-interval-dropdown",
                                    options=[
                                        {"label": "1 Min", "value": "1m"},
                                        {"label": "5 Min", "value": "5m"},
                                        {"label": "15 Min", "value": "15m"},
                                        {"label": "1 Hour", "value": "60m"},
                                        {"label": "1 Day", "value": "1d"},
                                        {"label": "1 Week", "value": "1wk"},
                                        {"label": "1 Month", "value": "1mo"},
                                    ],
                                    value="1d",
                                    clearable=False,
                                    style={"marginTop": "0.5rem"},
                                    className="dropdown-dark",
                                ),
                            ],
                        ),
                        # Period
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label("Period",
                                          style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                dcc.Dropdown(
                                    id="live-period-dropdown",
                                    options=[
                                        {"label": "1 Month", "value": "1mo"},
                                        {"label": "3 Months", "value": "3mo"},
                                        {"label": "6 Months", "value": "6mo"},
                                        {"label": "1 Year", "value": "1y"},
                                        {"label": "2 Years", "value": "2y"},
                                        {"label": "5 Years", "value": "5y"},
                                    ],
                                    value="1y",
                                    style={"marginTop": "0.5rem"},
                                    className="dropdown-dark",
                                ),
                            ],
                        ),
                        # Analysis Type
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label("Analysis",
                                          style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                dcc.Dropdown(
                                    id="live-analysis-dropdown",
                                    options=[
                                        {"label": "Price Chart", "value": "price"},
                                        {"label": "Realtime Board", "value": "board"},
                                        {"label": "Realtime Candles", "value": "candles"},
                                        {"label": "Returns", "value": "returns"},
                                        {"label": "Correlation", "value": "correlation"},
                                        {"label": "Factor Analysis", "value": "factor"},
                                        {"label": "Risk Metrics", "value": "risk"},
                                        {"label": "Quant Screener", "value": "screener"},
                                    ],
                                    value="price",
                                    style={"marginTop": "0.5rem"},
                                    className="dropdown-dark",
                                ),
                            ],
                        ),
                        # Fetch Button
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label("\u00a0", style={"fontSize": "0.875rem"}),
                                html.Button(
                                    [html.I(className="fas fa-search", style={"marginRight": "0.5rem"}), "Analyze"],
                                    id="live-fetch-btn",
                                    className="btn-primary",
                                    style={
                                        "width": "100%",
                                        "marginTop": "0.5rem",
                                        "padding": "0.75rem",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="row live-watchlist-row",
                    children=[
                        html.Div(
                            className="col col-3",
                            children=[
                                html.Label(
                                    "Quick Watchlist",
                                    style={
                                        "color": "var(--text-secondary)",
                                        "fontSize": "0.875rem",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="live-watchlist-dropdown",
                                    options=_watchlist_dropdown_options(),
                                    value="us_mega",
                                    clearable=False,
                                    style={"marginTop": "0.5rem"},
                                    className="dropdown-dark",
                                ),
                            ],
                        ),
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label(" ", style={"fontSize": "0.875rem"}),
                                html.Button(
                                    [
                                        html.I(
                                            className="fas fa-layer-group",
                                            style={"marginRight": "0.5rem"},
                                        ),
                                        "Load Preset",
                                    ],
                                    id="live-apply-watchlist-btn",
                                    className="btn-secondary",
                                    style={
                                        "width": "100%",
                                        "marginTop": "0.5rem",
                                        "padding": "0.75rem",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            className="col col-7",
                            children=[
                                html.Div(
                                    id="live-watchlist-status",
                                    children="Tip: choose a preset and load symbols before Analyze.",
                                    className="live-watchlist-status",
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="row live-resolution-row",
                    children=[
                        html.Div(
                            className="col col-3",
                            children=[
                                html.Label(
                                    "Realtime Resolution",
                                    style={
                                        "color": "var(--text-secondary)",
                                        "fontSize": "0.875rem",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="live-realtime-resolution-dropdown",
                                    options=[
                                        {"label": "Tick", "value": "tick"},
                                        {"label": "1 Second", "value": "1s"},
                                        {"label": "5 Seconds", "value": "5s"},
                                        {"label": "15 Seconds", "value": "15s"},
                                        {"label": "1 Minute", "value": "1m"},
                                    ],
                                    value="1s",
                                    clearable=False,
                                    style={"marginTop": "0.5rem"},
                                    className="dropdown-dark",
                                ),
                            ],
                        ),
                        html.Div(
                            className="col col-9",
                            children=[
                                html.Div(
                                    "Used when Data Mode is Realtime Stream (for Candles and Screener cadence).",
                                    style={
                                        "color": "var(--text-muted)",
                                        "fontSize": "0.82rem",
                                        "paddingTop": "1.95rem",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Status message
        html.Div(
            id="live-status",
            className="live-status-panel",
            style={"marginBottom": "1rem"},
        ),

        # Results area – no dcc.Loading wrapper to avoid content flash.
        # A thin CSS progress bar is shown via the ._live-updating class instead.
        html.Div(
            id="live-results-area",
            className="live-results-smooth",
        ),

        html.Div(
            className="chart-container",
            style={"marginTop": "1.5rem"},
            children=[
                html.Div(
                    [
                        html.H4(
                            "Paper Strategy Runner",
                            style={
                                "color": "var(--text-primary)",
                                "marginBottom": "0.35rem",
                            },
                        ),
                        html.P(
                            "Run a one-click paper execution using screened symbols.",
                            style={"color": "var(--text-secondary)", "marginBottom": "1rem"},
                        ),
                    ]
                ),
                html.Div(
                    className="row live-paper-row",
                    children=[
                        html.Div(
                            className="col col-3",
                            children=[
                                html.Label(
                                    "Strategy",
                                    style={
                                        "color": "var(--text-secondary)",
                                        "fontSize": "0.875rem",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="live-paper-strategy-dropdown",
                                    options=[
                                        {"label": "Momentum Top-N", "value": "momentum"},
                                        {"label": "Mean Reversion", "value": "mean_reversion"},
                                        {"label": "Low-Vol Momentum", "value": "low_vol_momentum"},
                                    ],
                                    value="momentum",
                                    clearable=False,
                                    style={"marginTop": "0.5rem"},
                                    className="dropdown-dark",
                                ),
                            ],
                        ),
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label(
                                    "Top N",
                                    style={
                                        "color": "var(--text-secondary)",
                                        "fontSize": "0.875rem",
                                    },
                                ),
                                dcc.Input(
                                    id="live-paper-topn-input",
                                    type="number",
                                    min=1,
                                    max=20,
                                    step=1,
                                    value=5,
                                    style={
                                        "width": "100%",
                                        "padding": "0.75rem",
                                        "marginTop": "0.5rem",
                                        "backgroundColor": "var(--bg-tertiary)",
                                        "border": "1px solid var(--border-color)",
                                        "borderRadius": "0.5rem",
                                        "color": "var(--text-primary)",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label(
                                    "Initial Capital",
                                    style={
                                        "color": "var(--text-secondary)",
                                        "fontSize": "0.875rem",
                                    },
                                ),
                                dcc.Input(
                                    id="live-paper-capital-input",
                                    type="number",
                                    min=1000,
                                    step=1000,
                                    value=100000,
                                    style={
                                        "width": "100%",
                                        "padding": "0.75rem",
                                        "marginTop": "0.5rem",
                                        "backgroundColor": "var(--bg-tertiary)",
                                        "border": "1px solid var(--border-color)",
                                        "borderRadius": "0.5rem",
                                        "color": "var(--text-primary)",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            className="col col-2",
                            children=[
                                html.Label(
                                    "Commission Rate",
                                    style={
                                        "color": "var(--text-secondary)",
                                        "fontSize": "0.875rem",
                                    },
                                ),
                                dcc.Input(
                                    id="live-paper-commission-input",
                                    type="number",
                                    min=0,
                                    max=0.02,
                                    step=0.0001,
                                    value=0.001,
                                    style={
                                        "width": "100%",
                                        "padding": "0.75rem",
                                        "marginTop": "0.5rem",
                                        "backgroundColor": "var(--bg-tertiary)",
                                        "border": "1px solid var(--border-color)",
                                        "borderRadius": "0.5rem",
                                        "color": "var(--text-primary)",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            className="col col-3",
                            children=[
                                html.Label(" ", style={"fontSize": "0.875rem"}),
                                html.Button(
                                    [
                                        html.I(
                                            className="fas fa-play",
                                            style={"marginRight": "0.5rem"},
                                        ),
                                        "Run on Screened Universe",
                                    ],
                                    id="live-run-paper-btn",
                                    className="btn-primary",
                                    style={
                                        "width": "100%",
                                        "marginTop": "0.5rem",
                                        "padding": "0.75rem",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="live-paper-status",
                    className="live-status-panel",
                    style={"marginTop": "1rem"},
                ),
                html.Div(id="live-paper-results", style={"marginTop": "1rem"}),
            ],
        ),

        # Store for data
        dcc.Store(id="live-data-store"),
        dcc.Store(id="live-stream-config-store"),
        dcc.Store(id="live-selected-symbol-store"),
        dcc.Interval(
            id="live-refresh-interval",
            interval=_refresh_interval_ms(),
            n_intervals=0,
            disabled=True,
        ),
    ])


def _to_list(series: pd.Series) -> list:
    """Convert pandas Series to plain Python list to avoid Plotly 6 bdata encoding."""
    return series.tolist()


def create_price_chart(prices_df: pd.DataFrame) -> go.Figure:
    """Create normalized price chart."""
    fig = go.Figure()

    # Normalize to 100
    normalized = prices_df / prices_df.iloc[0] * 100

    colors = px.colors.qualitative.Set2

    for i, col in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(
            x=list(normalized.index),
            y=_to_list(normalized[col]),
            name=col,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        template="algo_quant_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Normalized Price (100 = Start)",
        hovermode="x unified",
        transition=dict(duration=350, easing="cubic-in-out"),
    )

    fig.update_xaxes(gridcolor="rgba(72, 101, 129, 0.18)")
    fig.update_yaxes(gridcolor="rgba(72, 101, 129, 0.18)")

    return fig


def create_returns_chart(returns_df: pd.DataFrame) -> go.Figure:
    """Create cumulative returns chart."""
    fig = go.Figure()

    cumulative = (1 + returns_df).cumprod() - 1
    colors = px.colors.qualitative.Set2

    for i, col in enumerate(cumulative.columns):
        fig.add_trace(go.Scatter(
            x=list(cumulative.index),
            y=_to_list(cumulative[col] * 100),
            name=col,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            fill='tozeroy' if i == 0 else None,
        ))

    fig.update_layout(
        template="algo_quant_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
        transition=dict(duration=350, easing="cubic-in-out"),
    )

    fig.update_xaxes(gridcolor="rgba(72, 101, 129, 0.18)")
    fig.update_yaxes(gridcolor="rgba(72, 101, 129, 0.18)")

    return fig


def create_correlation_heatmap(returns_df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap."""
    corr = returns_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values.tolist(),
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr.values, 2).tolist(),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
    ))

    fig.update_layout(
        template="algo_quant_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=50, r=50, t=30, b=50),
        transition=dict(duration=350, easing="cubic-in-out"),
    )

    return fig


def _resample_ticks_to_candles(
    ticks_df: pd.DataFrame,
    resolution: str,
) -> pd.DataFrame:
    """Convert tick dataframe into OHLCV bars by resolution."""
    if ticks_df.empty or "price" not in ticks_df.columns:
        return pd.DataFrame()

    frame = ticks_df.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).set_index("timestamp")
    else:
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame[frame.index.notna()]

    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame.get("volume", 0.0), errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["price"]).sort_index()
    if frame.empty:
        return pd.DataFrame()

    if resolution == "tick":
        candles = pd.DataFrame(index=frame.index)
        candles["open"] = frame["price"]
        candles["high"] = frame["price"]
        candles["low"] = frame["price"]
        candles["close"] = frame["price"]
        candles["volume"] = frame["volume"]
        return candles

    freq_map = {
        "1s": "1s",
        "5s": "5s",
        "15s": "15s",
        "1m": "1min",
    }
    freq = freq_map.get(resolution, "1s")
    ohlc = frame["price"].resample(freq).ohlc()
    volume = frame["volume"].resample(freq).sum()
    candles = ohlc.join(volume.rename("volume"), how="left").dropna(subset=["open", "high", "low", "close"])
    return candles


def _build_realtime_candles(
    symbol: str,
    resolution: str,
    max_bars: int = 300,
) -> pd.DataFrame:
    """Fetch and build realtime OHLCV bars for a symbol."""
    hub = get_realtime_market_hub()
    ticks = hub.get_tick_history(symbol, max_points=max(1000, max_bars * 30))
    if ticks.empty:
        return pd.DataFrame()

    candles = _resample_ticks_to_candles(ticks, resolution=resolution)
    if candles.empty:
        return candles

    return candles.tail(max_bars)


def create_realtime_candlestick_chart(
    candles_df: pd.DataFrame,
    symbol: str,
    resolution: str,
) -> go.Figure:
    """Create realtime candlestick chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=list(candles_df.index),
            open=_to_list(candles_df["open"]),
            high=_to_list(candles_df["high"]),
            low=_to_list(candles_df["low"]),
            close=_to_list(candles_df["close"]),
            name=symbol,
            increasing_line_color="#0f766e",
            decreasing_line_color="#dc2626",
            increasing_fillcolor="rgba(15,118,110,0.35)",
            decreasing_fillcolor="rgba(220,38,38,0.32)",
        )
    )

    fig.update_layout(
        template="algo_quant_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=470,
        margin=dict(l=50, r=36, t=36, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title=f"{symbol} Price",
        xaxis_title=f"Resolution: {resolution}",
        xaxis_rangeslider_visible=False,
        transition=dict(duration=350, easing="cubic-in-out"),
    )
    fig.update_xaxes(gridcolor="rgba(72, 101, 129, 0.18)")
    fig.update_yaxes(gridcolor="rgba(72, 101, 129, 0.18)")
    return fig


def create_risk_metrics_table(returns_df: pd.DataFrame) -> html.Div:
    """Create risk metrics table."""
    metrics = []

    for col in returns_df.columns:
        r = returns_df[col].dropna()
        total_return = (1 + r).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(r)) - 1 if len(r) > 0 else 0
        volatility = r.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative = (1 + r).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        metrics.append({
            "Ticker": col,
            "Total Return": f"{total_return*100:.1f}%",
            "Annual Return": f"{annual_return*100:.1f}%",
            "Volatility": f"{volatility*100:.1f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd*100:.1f}%",
        })

    return html.Div([
        html.Table(
            className="metrics-table",
            style={
                "width": "100%",
                "borderCollapse": "collapse",
            },
            children=[
                html.Thead(
                    html.Tr([
                        html.Th(col, style={
                            "padding": "0.75rem",
                            "textAlign": "left",
                            "borderBottom": "2px solid var(--border-color)",
                            "color": "var(--text-secondary)",
                        }) for col in metrics[0].keys()
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(val, style={
                            "padding": "0.75rem",
                            "borderBottom": "1px solid var(--border-color)",
                            "color": "var(--text-primary)" if i == 0 else (
                                "#4ade80" if "%" in str(val) and float(val.replace("%", "")) > 0 else
                                "#f87171" if "%" in str(val) and float(val.replace("%", "")) < 0 else
                                "var(--text-primary)"
                            ),
                        }) for i, val in enumerate(row.values())
                    ]) for row in metrics
                ]),
            ],
        )
    ])


def create_factor_analysis_view(returns_df: pd.DataFrame, ff_factors: pd.DataFrame) -> html.Div:
    """Create factor analysis view with FF5 regression results."""
    from src.factors.ff5 import FamaFrench5

    results = []
    ff5_model = FamaFrench5()

    for col in returns_df.columns:
        try:
            stock_returns = returns_df[col].dropna()
            ff5_model.fit(stock_returns, ff_factors)
            r = ff5_model.result

            results.append({
                "Ticker": col,
                "Alpha (%)": f"{r.alpha*100:.3f}",
                "Market β": f"{r.loadings.get('Mkt-RF', 0):.2f}",
                "SMB": f"{r.loadings.get('SMB', 0):.2f}",
                "HML": f"{r.loadings.get('HML', 0):.2f}",
                "RMW": f"{r.loadings.get('RMW', 0):.2f}",
                "CMA": f"{r.loadings.get('CMA', 0):.2f}",
                "R²": f"{r.r_squared:.3f}",
            })
        except Exception as e:
            logger.warning(f"Factor analysis failed for {col}: {e}")
            results.append({
                "Ticker": col,
                "Alpha (%)": "N/A",
                "Market β": "N/A",
                "SMB": "N/A",
                "HML": "N/A",
                "RMW": "N/A",
                "CMA": "N/A",
                "R²": "N/A",
            })

    if not results:
        return html.Div("Factor analysis failed", style={"color": "var(--text-secondary)"})

    return html.Div([
        html.Div([
            html.H4("Fama-French 5 Factor Analysis", style={"color": "var(--text-primary)", "marginBottom": "0.5rem"}),
            html.P("Factor exposures based on FF5 model regression", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
        ], style={"marginBottom": "1rem"}),

        html.Table(
            className="metrics-table",
            style={"width": "100%", "borderCollapse": "collapse"},
            children=[
                html.Thead(
                    html.Tr([
                        html.Th(col, style={
                            "padding": "0.75rem",
                            "textAlign": "left",
                            "borderBottom": "2px solid var(--border-color)",
                            "color": "var(--text-secondary)",
                        }) for col in results[0].keys()
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(val, style={
                            "padding": "0.75rem",
                            "borderBottom": "1px solid var(--border-color)",
                            "color": "var(--text-primary)",
                        }) for val in row.values()
                    ]) for row in results
                ]),
            ],
        ),

        html.Div([
            html.P("β > 1: More volatile than market | SMB > 0: Small cap tilt | HML > 0: Value tilt",
                   style={"color": "var(--text-secondary)", "fontSize": "0.75rem", "marginTop": "1rem"}),
        ]),
    ])


def _parse_tickers(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]


# yfinance interval → max allowed period mapping
_INTERVAL_MAX_PERIODS: dict[str, list[str]] = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "5d", "1mo"],
    "15m": ["1d", "5d", "1mo"],
    "60m": ["1d", "5d", "1mo", "3mo", "6mo"],
    "1d": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    "1mo": ["3mo", "6mo", "1y", "2y", "5y"],
}


def _clamp_period_for_interval(interval: str, period: str) -> str:
    """Ensure period is valid for the selected interval."""
    valid = _INTERVAL_MAX_PERIODS.get(interval)
    if not valid:
        return period
    if period in valid:
        return period
    # Return the largest valid period as fallback
    return valid[-1]


def _get_period_options_for_interval(interval: str) -> list[dict[str, str]]:
    """Return dropdown options that are valid for the given interval."""
    _PERIOD_LABELS = {
        "1d": "1 Day",
        "5d": "5 Days",
        "1mo": "1 Month",
        "3mo": "3 Months",
        "6mo": "6 Months",
        "1y": "1 Year",
        "2y": "2 Years",
        "5y": "5 Years",
    }
    valid = _INTERVAL_MAX_PERIODS.get(interval, ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    return [{"label": _PERIOD_LABELS.get(p, p), "value": p} for p in valid]


_CLOSE_COLUMN_NAMES = (
    "Close",
    "close",
    "Adj Close",
    "adj close",
    "AdjClose",
    "adjclose",
)


def _coerce_close_series(candidate: Any) -> pd.Series | None:
    """Normalize close-like payload into a numeric Series."""
    if candidate is None:
        return None

    series: pd.Series | None = None
    if isinstance(candidate, pd.Series):
        series = candidate
    elif isinstance(candidate, pd.DataFrame):
        if candidate.empty:
            return None
        # yfinance can return a one-column DataFrame for Close in MultiIndex mode.
        for col_name in candidate.columns:
            col = candidate[col_name]
            if isinstance(col, pd.Series):
                series = col
                if not series.dropna().empty:
                    break
    else:
        return None

    if series is None:
        return None

    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    return series


def _extract_close_series(download_df: pd.DataFrame, ticker: str | None = None) -> pd.Series | None:
    """Extract close series from yfinance download output across column layouts."""
    if download_df is None or download_df.empty:
        return None

    columns = download_df.columns
    if getattr(columns, "nlevels", 1) == 1:
        for name in _CLOSE_COLUMN_NAMES:
            if name in columns:
                return _coerce_close_series(download_df[name])
        return None

    # Layout A: (PriceType, Ticker) e.g. ("Close", "AAPL")
    for name in _CLOSE_COLUMN_NAMES:
        try:
            sub = download_df[name]
        except Exception:
            continue

        if isinstance(sub, pd.DataFrame) and ticker and ticker in sub.columns:
            series = _coerce_close_series(sub[ticker])
            if series is not None:
                return series

        series = _coerce_close_series(sub)
        if series is not None:
            return series

    # Layout B: (Ticker, PriceType) e.g. ("AAPL", "Close")
    if ticker:
        try:
            sub = download_df[ticker]
        except Exception:
            sub = None

        if isinstance(sub, pd.DataFrame):
            for name in _CLOSE_COLUMN_NAMES:
                if name in sub.columns:
                    series = _coerce_close_series(sub[name])
                    if series is not None:
                        return series

        series = _coerce_close_series(sub)
        if series is not None:
            return series

    # Last-resort: pick any column tuple containing "close"
    for col_idx, col in enumerate(columns):
        labels = (col,) if not isinstance(col, tuple) else col
        norm = [str(level).strip().lower() for level in labels]
        if any(level in {"close", "adj close", "adjclose"} for level in norm):
            series = _coerce_close_series(download_df.iloc[:, col_idx])
            if series is not None:
                return series

    return None


def _fetch_historical_prices(
    tickers: list[str], period: str, interval: str = "1d",
) -> tuple[pd.DataFrame, list[str]]:
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

    capped = tickers[:20]
    period = _clamp_period_for_interval(interval, period)
    timeout_sec = _historical_request_timeout_sec()
    total_timeout_sec = max(10.0, min(90.0, timeout_sec * max(1, len(capped))))

    # Try bulk download first (single HTTP request for all symbols)
    try:
        bulk_df = yf.download(
            capped, period=period, interval=interval,
            group_by="ticker", progress=False, auto_adjust=True,
            threads=False, timeout=timeout_sec,
        )
        if not bulk_df.empty:
            prices_dict: dict[str, pd.Series] = {}
            failed: list[str] = []
            for ticker in capped:
                try:
                    series = _extract_close_series(bulk_df, ticker=ticker)
                    if series is not None:
                        prices_dict[ticker] = series
                    else:
                        failed.append(ticker)
                except Exception:
                    failed.append(ticker)
            if prices_dict:
                return pd.DataFrame(prices_dict), failed
    except Exception as e:
        logger.debug("Bulk download failed, falling back to concurrent fetch: %s", e)

    # Fallback: concurrent per-ticker fetch
    prices_dict = {}
    failed = []

    def _fetch_one(ticker: str) -> tuple[str, pd.Series | None]:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=timeout_sec,
            )
            if df.empty:
                return ticker, None

            series = _extract_close_series(df, ticker=ticker)
            if series is not None:
                return ticker, series
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", ticker, e)
        return ticker, None

    pending = set(capped)
    with ThreadPoolExecutor(max_workers=min(6, len(capped))) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in capped}
        try:
            for future in as_completed(futures, timeout=total_timeout_sec):
                ticker, series = future.result()
                pending.discard(ticker)
                if series is not None:
                    prices_dict[ticker] = series
                else:
                    failed.append(ticker)
        except FuturesTimeout:
            logger.warning(
                "Historical fetch timed out after %.1fs (interval=%s period=%s)",
                total_timeout_sec,
                interval,
                period,
            )

        for future in futures:
            if not future.done():
                future.cancel()

    for ticker in capped:
        if ticker in pending and ticker not in failed:
            failed.append(ticker)

    return pd.DataFrame(prices_dict), failed


def _create_screener_table(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    snapshot_df: pd.DataFrame | None = None,
) -> html.Div:
    if prices_df.empty or returns_df.empty:
        return html.Div(
            "Not enough data points yet for screening.",
            style={"color": "var(--text-secondary)"},
        )

    lookback = min(20, len(prices_df) - 1)
    if lookback < 2:
        return html.Div(
            "Waiting for more tick updates to calculate screener metrics...",
            style={"color": "var(--text-secondary)"},
        )

    momentum = prices_df.iloc[-1] / prices_df.iloc[-lookback] - 1
    volatility = returns_df.tail(lookback).std()
    sharpe_like = returns_df.tail(lookback).mean() / volatility.replace(0, np.nan)
    score = (
        momentum.rank(pct=True).fillna(0) * 0.6
        + sharpe_like.rank(pct=True).fillna(0) * 0.3
        + (1 - volatility.rank(pct=True)).fillna(0) * 0.1
    )

    volume_map = {}
    source_map = {}
    if snapshot_df is not None and not snapshot_df.empty:
        volume_map = {
            str(row["symbol"]): float(row["volume"])
            for _, row in snapshot_df.iterrows()
        }
        source_map = {
            str(row["symbol"]): str(row["source"])
            for _, row in snapshot_df.iterrows()
        }

    screener = pd.DataFrame(
        {
            "Ticker": momentum.index,
            "Price": prices_df.iloc[-1].values,
            f"{lookback}-Tick Mom%": momentum.values * 100,
            f"{lookback}-Tick Vol%": volatility.values * 100,
            "Sharpe-Like": sharpe_like.values,
            "Score": score.values,
            "Volume": [volume_map.get(sym, np.nan) for sym in momentum.index],
            "Source": [source_map.get(sym, "-") for sym in momentum.index],
        }
    ).sort_values("Score", ascending=False)

    screener = screener.head(20)
    if screener.empty:
        return html.Div(
            "No screener candidates yet. Waiting for more updates...",
            style={"color": "var(--text-secondary)"},
        )

    def _fmt(value: Any, col: str) -> str:
        if pd.isna(value):
            return "-"
        if col == "Price":
            return f"{float(value):,.2f}"
        if "Mom%" in col or "Vol%" in col:
            return f"{float(value):.2f}%"
        if col in ("Score", "Sharpe-Like"):
            return f"{float(value):.3f}"
        if col == "Volume":
            return f"{float(value):,.0f}"
        return str(value)

    rows = []
    for _, row in screener.iterrows():
        rows.append({col: _fmt(row[col], col) for col in screener.columns})

    return html.Div(
        className="chart-container",
        children=[
            html.Div("Realtime Quant Screener (Top Ranked)", className="chart-title"),
            html.Table(
                className="metrics-table",
                style={"width": "100%", "borderCollapse": "collapse"},
                children=[
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(
                                    col,
                                    style={
                                        "padding": "0.75rem",
                                        "textAlign": "left",
                                        "borderBottom": "2px solid var(--border-color)",
                                        "color": "var(--text-secondary)",
                                    },
                                )
                                for col in rows[0].keys()
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(
                                        val,
                                        style={
                                            "padding": "0.75rem",
                                            "borderBottom": "1px solid var(--border-color)",
                                            "color": "var(--text-primary)",
                                        },
                                    )
                                    for val in row.values()
                                ]
                            )
                            for row in rows
                        ]
                    ),
                ],
            ),
        ],
    )


def _build_market_watch_rows(
    prices_df: pd.DataFrame,
    snapshot_df: pd.DataFrame | None = None,
    lookback_points: int = 30,
) -> list[dict[str, Any]]:
    """Build ranked realtime market-watch rows from price frame."""
    if prices_df.empty:
        return []

    frame = prices_df.dropna(axis=1, how="all")
    if frame.empty:
        return []

    lookback = min(max(2, int(lookback_points)), len(frame))
    recent = frame.tail(lookback)

    latest = pd.to_numeric(recent.iloc[-1], errors="coerce")
    base = pd.to_numeric(recent.iloc[0], errors="coerce").replace(0, np.nan)
    change_pct = (latest / base - 1.0) * 100.0

    returns_df = recent.pct_change().dropna(how="all")
    volatility_pct = (
        returns_df.std() * 100.0 if not returns_df.empty else pd.Series(index=frame.columns, dtype=float)
    )

    volume_map: dict[str, float] = {}
    source_map: dict[str, str] = {}
    if snapshot_df is not None and not snapshot_df.empty:
        volume_map = {
            str(row["symbol"]): float(row["volume"])
            for _, row in snapshot_df.iterrows()
            if pd.notna(row.get("volume"))
        }
        source_map = {
            str(row["symbol"]): str(row["source"])
            for _, row in snapshot_df.iterrows()
            if pd.notna(row.get("source"))
        }

    rows: list[dict[str, Any]] = []
    for symbol in frame.columns:
        price_val = latest.get(symbol, np.nan)
        if pd.isna(price_val):
            continue

        chg = float(change_pct.get(symbol, np.nan))
        vol = float(volatility_pct.get(symbol, np.nan))
        score = chg if pd.isna(vol) or vol <= 0 else chg / vol

        rows.append(
            {
                "symbol": str(symbol),
                "price": float(price_val),
                "change_pct": chg,
                "volatility_pct": vol,
                "score": float(score) if np.isfinite(score) else float(chg),
                "volume": volume_map.get(str(symbol), np.nan),
                "source": source_map.get(str(symbol), "-"),
            }
        )

    rows.sort(key=lambda row: row["score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def _create_sparkline_figure(series: pd.Series, symbol: str) -> go.Figure:
    """Create tiny sparkline chart for market-watch card."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        values = pd.Series([0.0], index=[pd.Timestamp.utcnow()])

    x_index = values.index if isinstance(values.index, pd.DatetimeIndex) else list(range(len(values)))
    perf = 0.0
    if len(values) >= 2 and float(values.iloc[0]) != 0.0:
        perf = float(values.iloc[-1] / values.iloc[0] - 1.0)

    line_color = "#16a34a" if perf >= 0 else "#dc2626"
    fill_color = "rgba(22,163,74,0.14)" if perf >= 0 else "rgba(220,38,38,0.14)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(x_index),
            y=_to_list(values),
            mode="lines",
            line=dict(color=line_color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate=f"<b>{symbol}</b><br>%{{y:,.4f}}<extra></extra>",
            name=symbol,
        )
    )

    fig.update_layout(
        template="algo_quant_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=110,
        margin=dict(l=6, r=6, t=4, b=4),
        showlegend=False,
        transition=dict(duration=250, easing="cubic-in-out"),
    )
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    return fig


def _render_realtime_board(
    prices_df: pd.DataFrame,
    snapshot_df: pd.DataFrame | None,
    realtime_resolution: str,
    selected_symbol: str | None = None,
) -> html.Div:
    """Render tradingview-like realtime watch board."""
    rows = _build_market_watch_rows(
        prices_df=prices_df,
        snapshot_df=snapshot_df,
        lookback_points=40,
    )
    if not rows:
        return html.Div(
            "Waiting for realtime stream data...",
            style={"color": "var(--text-secondary)"},
        )

    symbols = [str(row["symbol"]) for row in rows]
    current_symbol = (
        str(selected_symbol).upper()
        if selected_symbol and str(selected_symbol).upper() in symbols
        else symbols[0]
    )

    cards = []
    for row in rows[:12]:
        symbol = row["symbol"]
        series = pd.to_numeric(prices_df[symbol], errors="coerce").dropna().tail(120)
        change_pct = float(row["change_pct"])
        change_text = "N/A" if not np.isfinite(change_pct) else f"{change_pct:+.2f}%"
        change_class = "positive" if change_pct >= 0 else "negative"
        card_class = (
            "market-watch-card is-selected"
            if symbol == current_symbol
            else "market-watch-card"
        )

        cards.append(
            html.Div(
                className=card_class,
                children=[
                    html.Div(
                        className="market-watch-card-top",
                        children=[
                            html.Div(symbol, className="market-watch-symbol"),
                            html.Div(f"{row['price']:,.4f}", className="market-watch-price"),
                        ],
                    ),
                    html.Div(
                        className=f"market-watch-change {change_class}",
                        children=change_text,
                    ),
                    dcc.Graph(
                        id={"type": "live-board-spark", "symbol": symbol},
                        figure=_create_sparkline_figure(series, symbol),
                        config={"displayModeBar": False, "staticPlot": True},
                        className="market-watch-spark",
                    ),
                    html.Div(
                        className="market-watch-meta",
                        children=[
                            html.Span(f"Vol: {row['volatility_pct']:.2f}%"
                                      if np.isfinite(row["volatility_pct"]) else "Vol: -"),
                            html.Span(
                                f"Volume: {row['volume']:,.0f}"
                                if pd.notna(row["volume"])
                                else "Volume: -"
                            ),
                            html.Span(f"Source: {row['source']}"),
                        ],
                    ),
                ],
            )
        )

    table_rows = []
    for row in rows[:40]:
        change_pct = row["change_pct"]
        vol_pct = row["volatility_pct"]
        table_rows.append(
            html.Tr(
                [
                    html.Td(str(row["rank"]), style={"padding": "0.62rem"}),
                    html.Td(row["symbol"], style={"padding": "0.62rem"}),
                    html.Td(f"{row['price']:,.4f}", style={"padding": "0.62rem"}),
                    html.Td(
                        "N/A" if not np.isfinite(change_pct) else f"{change_pct:+.2f}%",
                        style={
                            "padding": "0.62rem",
                            "color": "#16a34a" if np.isfinite(change_pct) and change_pct >= 0 else "#dc2626",
                        },
                    ),
                    html.Td(
                        "N/A" if not np.isfinite(vol_pct) else f"{vol_pct:.2f}%",
                        style={"padding": "0.62rem"},
                    ),
                    html.Td(
                        f"{row['volume']:,.0f}" if pd.notna(row["volume"]) else "-",
                        style={"padding": "0.62rem"},
                    ),
                    html.Td(str(row["source"]), style={"padding": "0.62rem"}),
                ]
            )
        )

    detail_candles = _build_realtime_candles(
        symbol=current_symbol,
        resolution=realtime_resolution,
        max_bars=220,
    )
    detail_chart_block: html.Div | html.P
    if detail_candles.empty or len(detail_candles) < 2:
        detail_chart_block = html.P(
            f"Collecting ticks for {current_symbol}...",
            style={"color": "var(--text-secondary)"},
        )
    else:
        detail_chart_block = dcc.Graph(
            figure=create_realtime_candlestick_chart(
                candles_df=detail_candles,
                symbol=current_symbol,
                resolution=realtime_resolution,
            ),
            config={"displayModeBar": False},
        )

    return html.Div(
        children=[
            html.Div(
                className="market-watch-summary",
                children=[
                    html.Span(f"Tracked: {len(rows)}"),
                    html.Span(f"Selected: {current_symbol}"),
                    html.Span(f"Resolution: {realtime_resolution}"),
                    html.Span("Top cards: 12 | Tape rows: 40"),
                    html.Span("Tip: click sparkline card to pin symbol"),
                ],
            ),
            html.Div(className="market-watch-grid", children=cards),
            html.Div(
                className="chart-container",
                style={"marginTop": "1rem"},
                children=[
                    html.Div(f"Pinned Symbol Detail ({current_symbol})", className="chart-title"),
                    detail_chart_block,
                ],
            ),
            html.Div(
                className="chart-container",
                style={"marginTop": "1rem"},
                children=[
                    html.Div("Realtime Market Tape", className="chart-title"),
                    html.Table(
                        className="metrics-table",
                        style={"width": "100%", "borderCollapse": "collapse"},
                        children=[
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Rank", style={"padding": "0.62rem", "textAlign": "left"}),
                                        html.Th("Symbol", style={"padding": "0.62rem", "textAlign": "left"}),
                                        html.Th("Last", style={"padding": "0.62rem", "textAlign": "left"}),
                                        html.Th("Change", style={"padding": "0.62rem", "textAlign": "left"}),
                                        html.Th("Volatility", style={"padding": "0.62rem", "textAlign": "left"}),
                                        html.Th("Volume", style={"padding": "0.62rem", "textAlign": "left"}),
                                        html.Th("Source", style={"padding": "0.62rem", "textAlign": "left"}),
                                    ]
                                )
                            ),
                            html.Tbody(table_rows),
                        ],
                    ),
                ],
            ),
        ]
    )


def _screened_universe_from_prices(prices_df: pd.DataFrame, top_n: int = 10) -> list[str]:
    """Select top symbols by short-horizon momentum for strategy handoff."""
    if prices_df.empty or len(prices_df) < 3:
        return []

    lookback = min(20, len(prices_df) - 1)
    momentum = (prices_df.iloc[-1] / prices_df.iloc[-lookback] - 1).sort_values(
        ascending=False
    )
    return list(momentum.head(top_n).index)


def _select_strategy_targets(
    prices_df: pd.DataFrame,
    strategy: str,
    top_n: int,
) -> list[str]:
    """Compatibility wrapper for strategy target selection."""
    return select_strategy_targets(
        prices_df=prices_df,
        strategy=strategy,
        top_n=top_n,
    )


def _latest_prices_for_symbols(
    prices_df: pd.DataFrame,
    symbols: list[str],
) -> dict[str, float]:
    """Compatibility wrapper for latest price extraction."""
    return latest_prices_for_symbols(
        prices_df=prices_df,
        symbols=symbols,
    )


def _build_prices_for_paper_run(
    symbols: list[str],
    data_mode: str,
    period: str,
) -> pd.DataFrame:
    """Compatibility wrapper for paper-run price frame builder."""
    return build_prices_for_paper_run(
        symbols=symbols,
        data_mode=data_mode,
        period=period,
    )


async def _run_paper_strategy_once(
    prices_df: pd.DataFrame,
    strategy: str,
    top_n: int,
    initial_capital: float,
    commission_rate: float,
) -> dict[str, Any]:
    """Compatibility wrapper for paper strategy execution."""
    return await run_paper_strategy_once(
        prices_df=prices_df,
        strategy=strategy,
        top_n=top_n,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
    )


def _render_paper_result(result: dict[str, Any]) -> html.Div:
    """Render paper strategy execution output."""
    if not result.get("ok"):
        return html.Div(
            result.get("message", "Paper strategy run failed."),
            style={"color": "#f87171"},
        )

    summary = result.get("summary", {})
    orders = result.get("orders", [])
    positions = result.get("positions", [])

    summary_box = html.Div(
        [
            html.Span(
                f"Strategy: {result.get('strategy', '-')}",
                style={"color": "var(--text-secondary)"},
            ),
            html.Span(
                f" | Selected: {', '.join(result.get('selected', []))}",
                style={"color": "var(--text-secondary)"},
            ),
            html.Span(
                f" | Trades: {summary.get('num_trades', 0)}",
                style={"color": "var(--text-secondary)"},
            ),
            html.Span(
                f" | Portfolio: {summary.get('portfolio_value', 0.0):,.2f}",
                style={"color": "var(--text-secondary)"},
            ),
            html.Span(
                f" | Cash: {summary.get('cash', 0.0):,.2f}",
                style={"color": "var(--text-secondary)"},
            ),
        ],
        style={"marginBottom": "0.75rem"},
    )

    order_rows = [
        html.Tr(
            [
                html.Td(row["symbol"], style={"padding": "0.6rem"}),
                html.Td(row["status"], style={"padding": "0.6rem"}),
                html.Td(f"{row['quantity']:.4f}", style={"padding": "0.6rem"}),
                html.Td(f"{row['price']:.4f}", style={"padding": "0.6rem"}),
                html.Td(f"{row['commission']:.4f}", style={"padding": "0.6rem"}),
                html.Td(row["message"] or "-", style={"padding": "0.6rem"}),
            ]
        )
        for row in orders
    ]

    position_rows = [
        html.Tr(
            [
                html.Td(row["symbol"], style={"padding": "0.6rem"}),
                html.Td(f"{row['quantity']:.4f}", style={"padding": "0.6rem"}),
                html.Td(f"{row['avg_cost']:.4f}", style={"padding": "0.6rem"}),
                html.Td(f"{row['market_value']:.2f}", style={"padding": "0.6rem"}),
            ]
        )
        for row in positions
    ]

    return html.Div(
        className="chart-container",
        children=[
            html.Div("Paper Execution Result", className="chart-title"),
            summary_box,
            html.Div(
                [
                    html.Div(
                        "Order Fills",
                        style={
                            "color": "var(--text-secondary)",
                            "fontSize": "0.875rem",
                            "marginBottom": "0.4rem",
                        },
                    ),
                    html.Table(
                        className="metrics-table",
                        style={"width": "100%", "borderCollapse": "collapse"},
                        children=[
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Symbol", style={"padding": "0.6rem", "textAlign": "left"}),
                                        html.Th("Status", style={"padding": "0.6rem", "textAlign": "left"}),
                                        html.Th("Qty", style={"padding": "0.6rem", "textAlign": "left"}),
                                        html.Th("Price", style={"padding": "0.6rem", "textAlign": "left"}),
                                        html.Th("Commission", style={"padding": "0.6rem", "textAlign": "left"}),
                                        html.Th("Message", style={"padding": "0.6rem", "textAlign": "left"}),
                                    ]
                                )
                            ),
                            html.Tbody(order_rows),
                        ],
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Div(
                        "Current Positions",
                        style={
                            "color": "var(--text-secondary)",
                            "fontSize": "0.875rem",
                            "marginBottom": "0.4rem",
                        },
                    ),
                    html.Table(
                        className="metrics-table",
                        style={"width": "100%", "borderCollapse": "collapse"},
                        children=[
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Symbol", style={"padding": "0.6rem", "textAlign": "left"}),
                                        html.Th("Qty", style={"padding": "0.6rem", "textAlign": "left"}),
                                        html.Th("Avg Cost", style={"padding": "0.6rem", "textAlign": "left"}),
                                        html.Th("Market Value", style={"padding": "0.6rem", "textAlign": "left"}),
                                    ]
                                )
                            ),
                            html.Tbody(position_rows),
                        ],
                    ),
                ]
            ),
        ],
    )


def _render_analysis_view(
    prices_df: pd.DataFrame,
    analysis_type: str,
    mode: str,
    snapshot_df: pd.DataFrame | None = None,
    realtime_resolution: str = "1s",
    selected_symbol: str | None = None,
) -> html.Div:
    if prices_df.empty:
        return html.Div(
            "Waiting for price data...",
            style={"color": "var(--text-secondary)"},
        )

    returns_df = prices_df.pct_change().dropna()

    if analysis_type == "price":
        chart = create_price_chart(prices_df)
        return html.Div(
            className="chart-container",
            children=[
                html.Div("Normalized Price Chart", className="chart-title"),
                dcc.Graph(figure=chart, config={"displayModeBar": False}),
            ],
        )

    if analysis_type == "returns":
        if returns_df.empty:
            return html.Div(
                "Need more data points for return analysis.",
                style={"color": "var(--text-secondary)"},
            )
        chart = create_returns_chart(returns_df)
        return html.Div(
            className="chart-container",
            children=[
                html.Div("Cumulative Returns", className="chart-title"),
                dcc.Graph(figure=chart, config={"displayModeBar": False}),
            ],
        )

    if analysis_type == "correlation":
        if returns_df.empty:
            return html.Div(
                "Need more data points for correlation analysis.",
                style={"color": "var(--text-secondary)"},
            )
        chart = create_correlation_heatmap(returns_df)
        return html.Div(
            className="chart-container",
            children=[
                html.Div("Correlation Matrix", className="chart-title"),
                dcc.Graph(figure=chart, config={"displayModeBar": False}),
            ],
        )

    if analysis_type == "risk":
        if returns_df.empty:
            return html.Div(
                "Need more data points for risk analysis.",
                style={"color": "var(--text-secondary)"},
            )
        return html.Div(
            className="chart-container",
            children=[
                html.Div("Risk Metrics", className="chart-title"),
                create_risk_metrics_table(returns_df),
            ],
        )

    if analysis_type == "factor":
        if mode == "realtime":
            return html.Div(
                className="chart-container",
                children=[
                    html.Div("Factor Analysis", className="chart-title"),
                    html.P(
                        "Realtime stream is tick-level. Factor analysis requires "
                        "daily factor data alignment. Switch to Historical mode.",
                        style={"color": "var(--text-secondary)"},
                    ),
                ],
            )
        try:
            ff_factors = _load_ff5_factors_cached("daily")
        except Exception as exc:
            logger.warning("Failed to load FF5 factors: %s", exc)
            return html.Div(
                className="chart-container",
                children=[
                    html.Div("Factor Analysis", className="chart-title"),
                    html.P(
                        "Failed to load Fama-French factors. Retry later or use another analysis.",
                        style={"color": "var(--text-secondary)"},
                    ),
                ],
            )
        return html.Div(
            className="chart-container",
            children=[create_factor_analysis_view(returns_df, ff_factors)],
        )

    if analysis_type == "board":
        if mode != "realtime":
            return html.Div(
                className="chart-container",
                children=[
                    html.Div("Realtime Board", className="chart-title"),
                    html.P(
                        "TradingView-like board is available in Realtime Stream mode.",
                        style={"color": "var(--text-secondary)"},
                    ),
                ],
            )
        return _render_realtime_board(
            prices_df=prices_df,
            snapshot_df=snapshot_df,
            realtime_resolution=realtime_resolution,
            selected_symbol=selected_symbol,
        )

    if analysis_type == "candles":
        if mode != "realtime":
            return html.Div(
                className="chart-container",
                children=[
                    html.Div("Realtime Candles", className="chart-title"),
                    html.P(
                        "Candlestick view is available in Realtime Stream mode.",
                        style={"color": "var(--text-secondary)"},
                    ),
                ],
            )

        symbol = ""
        if selected_symbol and selected_symbol in prices_df.columns:
            symbol = str(selected_symbol)
        elif len(prices_df.columns) > 0:
            symbol = str(prices_df.columns[0])
        if not symbol:
            return html.Div(
                "Waiting for symbol stream...",
                style={"color": "var(--text-secondary)"},
            )

        candles_df = _build_realtime_candles(
            symbol=symbol,
            resolution=realtime_resolution,
        )
        if candles_df.empty or len(candles_df) < 2:
            return html.Div(
                "Waiting for enough realtime ticks to build candles...",
                style={"color": "var(--text-secondary)"},
            )

        chart = create_realtime_candlestick_chart(
            candles_df=candles_df,
            symbol=symbol,
            resolution=realtime_resolution,
        )

        hint = None
        if len(prices_df.columns) > 1:
            hint = html.Div(
                f"Candles currently shown for {symbol}. Additional symbols tracked: "
                + ", ".join([str(c) for c in prices_df.columns[1:4]])
                + ("..." if len(prices_df.columns) > 4 else ""),
                style={
                    "marginTop": "0.6rem",
                    "fontSize": "0.8rem",
                    "color": "var(--text-muted)",
                },
            )

        return html.Div(
            className="chart-container",
            children=[
                html.Div(f"Realtime Candles ({symbol})", className="chart-title"),
                dcc.Graph(figure=chart, config={"displayModeBar": False}),
                hint,
            ],
        )

    if analysis_type == "screener":
        return _create_screener_table(prices_df, returns_df, snapshot_df=snapshot_df)

    return html.Div("Unknown analysis type")


def _historical_status(
    prices_df: pd.DataFrame, failed: list[str], interval: str = "1d",
) -> html.Div:
    return html.Div(
        [
            html.Span("Historical data loaded", style={"color": "#4ade80"}),
            html.Span(
                f" | YFinance | {len(prices_df.columns)} tickers"
                f" | {len(prices_df)} bars | interval: {interval}",
                style={"color": "var(--text-secondary)"},
            ),
            html.Span(
                f" | Failed: {', '.join(failed)}",
                style={"color": "#f87171"},
            )
            if failed
            else None,
        ]
    )


def _realtime_status(status: dict[str, Any], snapshot_df: pd.DataFrame) -> html.Div:
    providers = ", ".join(status.get("providers", [])) or "-"
    tracked = len(snapshot_df) if snapshot_df is not None else 0
    latest = status.get("latest_update") or "waiting"
    errors = status.get("errors", [])

    return html.Div(
        [
            html.Span(
                "✓ Realtime stream active" if status.get("running") else "Realtime stream starting...",
                style={"color": "#4ade80" if status.get("running") else "#facc15"},
            ),
            html.Span(
                f" | Providers: {providers} | Tracked: {tracked} | Last update: {latest}",
                style={"color": "var(--text-secondary)"},
            ),
            html.Span(
                f" | Warning: {errors[-1]}",
                style={"color": "#f87171"},
            )
            if errors
            else None,
        ]
    )


def register_live_analyzer_callbacks(app):
    """Register callbacks for live analyzer."""

    @app.callback(
        [
            Output("live-ticker-input", "value"),
            Output("live-watchlist-status", "children"),
        ],
        [Input("live-apply-watchlist-btn", "n_clicks")],
        [State("live-watchlist-dropdown", "value"), State("live-ticker-input", "value")],
        prevent_initial_call=True,
    )
    def apply_watchlist_preset(n_clicks, preset_name, current_value):
        """Load a predefined watchlist into ticker input."""
        if not n_clicks:
            raise PreventUpdate

        watchlists = _load_live_watchlists()
        symbols = watchlists.get(str(preset_name or ""), [])
        if not symbols:
            return (
                current_value,
                html.Span(
                    "Preset not found. Select another watchlist.",
                    style={"color": "#f87171"},
                ),
            )

        ticker_text = " ".join(symbols)
        return (
            ticker_text,
            html.Span(
                f"Loaded preset '{_format_watchlist_label(str(preset_name))}' "
                f"({len(symbols)} symbols).",
                style={"color": "#16a34a"},
            ),
        )

    @app.callback(
        [
            Output("live-period-dropdown", "options"),
            Output("live-period-dropdown", "value"),
        ],
        [Input("live-interval-dropdown", "value")],
        [State("live-period-dropdown", "value")],
        prevent_initial_call=True,
    )
    def update_period_for_interval(interval, current_period):
        """Dynamically constrain period options based on selected interval."""
        options = _get_period_options_for_interval(interval or "1d")
        valid_values = [o["value"] for o in options]
        # Keep current period if still valid, otherwise use last (largest)
        new_value = current_period if current_period in valid_values else valid_values[-1]
        return options, new_value

    @app.callback(
        Output("live-selected-symbol-store", "data"),
        [Input({"type": "live-board-spark", "symbol": ALL}, "clickData")],
        [
            State({"type": "live-board-spark", "symbol": ALL}, "id"),
            State("live-selected-symbol-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def select_board_symbol(click_datas, graph_ids, current_symbol):
        """Pin symbol when user clicks a board sparkline."""
        del current_symbol
        trigger = callback_context.triggered_id
        if isinstance(trigger, dict) and trigger.get("symbol"):
            return str(trigger["symbol"])

        for click_data, graph_id in zip(click_datas or [], graph_ids or []):
            if click_data and isinstance(graph_id, dict) and graph_id.get("symbol"):
                return str(graph_id["symbol"])

        raise PreventUpdate

    @app.callback(
        [
            Output("live-results-area", "children"),
            Output("live-status", "children"),
            Output("live-data-store", "data"),
            Output("live-stream-config-store", "data"),
            Output("live-refresh-interval", "disabled"),
            Output("store-screened-universe", "data"),
            Output("live-refresh-interval", "interval"),
        ],
        [
            Input("live-fetch-btn", "n_clicks"),
            Input("live-refresh-interval", "n_intervals"),
            Input("live-analysis-dropdown", "value"),
            Input("live-realtime-resolution-dropdown", "value"),
            Input("live-selected-symbol-store", "data"),
        ],
        [
            State("live-ticker-input", "value"),
            State("live-period-dropdown", "value"),
            State("live-interval-dropdown", "value"),
            State("live-data-mode-dropdown", "value"),
            State("live-stream-config-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def fetch_and_analyze(
        n_clicks,
        n_intervals,
        analysis_type,
        realtime_resolution,
        selected_symbol,
        tickers_str,
        period,
        chart_interval,
        data_mode,
        stream_config,
    ):
        del n_clicks, n_intervals
        trigger = callback_context.triggered_id
        base_interval = _refresh_interval_ms()
        if not trigger:
            raise PreventUpdate

        hub = get_realtime_market_hub()
        try:
            if trigger == "live-fetch-btn":
                tickers = _parse_tickers(tickers_str)
                if not tickers:
                    return (
                        html.Div(),
                        html.Div("Please enter valid tickers", style={"color": "#f87171"}),
                        None,
                        None,
                        True,
                        no_update,
                        base_interval,
                    )

                if data_mode == "historical":
                    # Explicitly stop stream when user switches to historical mode
                    hub.stop()

                    interval = chart_interval or "1d"
                    prices_df, failed = _fetch_historical_prices(tickers, period, interval=interval)
                    if prices_df.empty:
                        return (
                            html.Div(),
                            html.Div(
                                "Failed to fetch historical data. "
                                "Try fewer symbols or a larger interval.",
                                style={"color": "#f87171"},
                            ),
                            None,
                            None,
                            True,
                            no_update,
                            base_interval,
                        )

                    result = _render_analysis_view(
                        prices_df,
                        analysis_type=analysis_type,
                        mode="historical",
                        realtime_resolution=realtime_resolution,
                        selected_symbol=selected_symbol,
                    )
                    status = _historical_status(prices_df, failed, interval=interval)
                    payload = {
                        "tickers": list(prices_df.columns),
                        "mode": "historical",
                        "analysis": analysis_type,
                    }
                    screened = no_update
                    if analysis_type == "screener":
                        screened = {
                            "tickers": _screened_universe_from_prices(prices_df),
                            "mode": "historical",
                            "updated_at": pd.Timestamp.utcnow().isoformat(),
                        }
                    return result, status, payload, None, True, screened, base_interval

                # Realtime mode
                tickers = _limit_realtime_tickers(tickers)
                start_status = hub.start(tickers)
                prices_df = hub.get_price_frame(tickers)
                snapshot_df = hub.get_latest_snapshot(tickers)
                dynamic_interval = _dynamic_refresh_interval_ms(
                    base_ms=base_interval,
                    analysis_type=analysis_type,
                    resolution=realtime_resolution,
                    ticker_count=len(tickers),
                )
                result = _render_analysis_view(
                    prices_df,
                    analysis_type=analysis_type,
                    mode="realtime",
                    snapshot_df=snapshot_df,
                    realtime_resolution=realtime_resolution,
                    selected_symbol=selected_symbol,
                )
                status = _realtime_status(start_status, snapshot_df)
                hub_status = hub.get_status()
                stream_payload = {
                    "tickers": tickers,
                    "mode": "realtime",
                    "analysis": analysis_type,
                    "resolution": realtime_resolution,
                    "last_update": hub_status.get("latest_update"),
                }
                data_payload = {"tickers": tickers, "mode": "realtime"}
                screened = no_update
                if analysis_type == "screener":
                    screened = {
                        "tickers": _screened_universe_from_prices(prices_df),
                        "mode": "realtime",
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                    }
                return (
                    result,
                    status,
                    data_payload,
                    stream_payload,
                    False,
                    screened,
                    dynamic_interval,
                )

            if trigger in (
                "live-refresh-interval",
                "live-analysis-dropdown",
                "live-realtime-resolution-dropdown",
                "live-selected-symbol-store",
            ):
                if not stream_config or stream_config.get("mode") != "realtime":
                    raise PreventUpdate

                tickers = stream_config.get("tickers", [])
                if not tickers:
                    raise PreventUpdate

                hub_status = hub.get_status()
                latest_update = hub_status.get("latest_update")
                if (
                    trigger == "live-refresh-interval"
                    and latest_update
                    and latest_update == stream_config.get("last_update")
                ):
                    # Skip expensive redraw when no fresh tick has arrived.
                    raise PreventUpdate

                prices_df = hub.get_price_frame(tickers)
                snapshot_df = hub.get_latest_snapshot(tickers)
                status = _realtime_status(hub_status, snapshot_df)
                dynamic_interval = _dynamic_refresh_interval_ms(
                    base_ms=base_interval,
                    analysis_type=analysis_type,
                    resolution=realtime_resolution,
                    ticker_count=len(tickers),
                )
                result = _render_analysis_view(
                    prices_df,
                    analysis_type=analysis_type,
                    mode="realtime",
                    snapshot_df=snapshot_df,
                    realtime_resolution=realtime_resolution,
                    selected_symbol=selected_symbol,
                )
                stream_config["analysis"] = analysis_type
                stream_config["resolution"] = realtime_resolution
                stream_config["last_update"] = latest_update
                return (
                    result,
                    status,
                    {"tickers": tickers, "mode": "realtime"},
                    stream_config,
                    False,
                    {
                        "tickers": _screened_universe_from_prices(prices_df),
                        "mode": "realtime",
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                    }
                    if analysis_type == "screener"
                    else no_update,
                    dynamic_interval,
                )
        except PreventUpdate:
            raise
        except Exception as exc:
            logger.exception(
                "Live analyzer callback failed (trigger=%s mode=%s analysis=%s)",
                trigger,
                data_mode,
                analysis_type,
            )
            message = str(exc).strip() or "unknown error"
            return (
                html.Div(),
                html.Div(
                    f"Live analyzer request failed: {message}",
                    style={"color": "#f87171"},
                ),
                no_update,
                no_update,
                False if stream_config and stream_config.get("mode") == "realtime" else True,
                no_update,
                base_interval,
            )

        raise PreventUpdate

    @app.callback(
        [
            Output("live-paper-status", "children"),
            Output("live-paper-results", "children"),
        ],
        [Input("live-run-paper-btn", "n_clicks")],
        [
            State("store-screened-universe", "data"),
            State("live-data-mode-dropdown", "value"),
            State("live-period-dropdown", "value"),
            State("live-paper-strategy-dropdown", "value"),
            State("live-paper-topn-input", "value"),
            State("live-paper-capital-input", "value"),
            State("live-paper-commission-input", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_paper_strategy(
        n_clicks,
        screened_data,
        data_mode,
        period,
        strategy,
        top_n,
        initial_capital,
        commission_rate,
    ):
        """Run one-click paper strategy from screened universe."""
        if not n_clicks:
            raise PreventUpdate

        tickers = screened_data.get("tickers", []) if screened_data else []
        if not tickers:
            return (
                html.Div(
                    "No screened universe found. Run Quant Screener first.",
                    style={"color": "#f87171"},
                ),
                html.Div(),
            )

        top_n = int(top_n or 5)
        if top_n <= 0:
            return (
                html.Div("Top N must be greater than zero.", style={"color": "#f87171"}),
                html.Div(),
            )

        initial_capital = float(initial_capital or 0)
        if initial_capital <= 0:
            return (
                html.Div(
                    "Initial capital must be greater than zero.",
                    style={"color": "#f87171"},
                ),
                html.Div(),
            )

        commission_rate = float(commission_rate or 0)
        if commission_rate < 0:
            return (
                html.Div(
                    "Commission rate cannot be negative.",
                    style={"color": "#f87171"},
                ),
                html.Div(),
            )

        prices_df = _build_prices_for_paper_run(
            symbols=tickers,
            data_mode=data_mode,
            period=period,
        )
        if prices_df.empty:
            return (
                html.Div(
                    "No price data available for paper execution. "
                    "Wait for stream updates or re-run analysis.",
                    style={"color": "#f87171"},
                ),
                html.Div(),
            )

        try:
            result = asyncio.run(
                _run_paper_strategy_once(
                    prices_df=prices_df,
                    strategy=(strategy or "momentum"),
                    top_n=top_n,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                )
            )
        except Exception as e:
            logger.exception("Paper strategy execution failed")
            return (
                html.Div(
                    f"Paper execution failed: {e}",
                    style={"color": "#f87171"},
                ),
                html.Div(),
            )

        status_color = "#4ade80" if result.get("ok") else "#facc15"
        return (
            html.Div(
                result.get("message", "Paper execution finished."),
                style={"color": status_color},
            ),
            _render_paper_result(result),
        )
