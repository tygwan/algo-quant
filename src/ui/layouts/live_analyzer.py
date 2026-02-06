"""Live Stock Analyzer - interactive ticker input and streaming analysis."""

from dash import html, dcc, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import logging
from typing import Any

from src.ui.services.realtime_hub import get_realtime_market_hub

logger = logging.getLogger(__name__)


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
            style={"marginBottom": "1.5rem"},
            children=[
                html.Div(
                    className="row",
                    children=[
                        # Ticker Input
                        html.Div(
                            className="col col-3",
                            children=[
                                html.Label("Tickers (space or comma separated)",
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
                        # Date Range
                        html.Div(
                            className="col col-3",
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
                                html.Label(" ", style={"fontSize": "0.875rem"}),
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
            ],
        ),

        # Loading indicator
        dcc.Loading(
            id="live-loading",
            type="circle",
            children=[
                # Status message
                html.Div(id="live-status", style={"marginBottom": "1rem"}),

                # Results area
                html.Div(id="live-results-area"),
            ],
        ),

        # Store for data
        dcc.Store(id="live-data-store"),
        dcc.Store(id="live-stream-config-store"),
        dcc.Interval(
            id="live-refresh-interval",
            interval=3000,  # 3 seconds
            n_intervals=0,
            disabled=True,
        ),
    ])


def create_price_chart(prices_df: pd.DataFrame) -> go.Figure:
    """Create normalized price chart."""
    fig = go.Figure()

    # Normalize to 100
    normalized = prices_df / prices_df.iloc[0] * 100

    colors = px.colors.qualitative.Set2

    for i, col in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[col],
            name=col,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Normalized Price (100 = Start)",
        hovermode="x unified",
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")

    return fig


def create_returns_chart(returns_df: pd.DataFrame) -> go.Figure:
    """Create cumulative returns chart."""
    fig = go.Figure()

    cumulative = (1 + returns_df).cumprod() - 1
    colors = px.colors.qualitative.Set2

    for i, col in enumerate(cumulative.columns):
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative[col] * 100,
            name=col,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            fill='tozeroy' if i == 0 else None,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")

    return fig


def create_correlation_heatmap(returns_df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap."""
    corr = returns_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=50, r=50, t=30, b=50),
    )

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


def _fetch_historical_prices(
    tickers: list[str], period: str
) -> tuple[pd.DataFrame, list[str]]:
    from src.data.yfinance_client import YFinanceClient

    yf_client = YFinanceClient()
    prices_dict = {}
    failed = []

    for ticker in tickers[:20]:
        try:
            df = yf_client.get_historical_prices(ticker, period=period)
            if not df.empty and "close" in df.columns:
                prices_dict[ticker] = df["close"]
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
            logger.warning(f"Failed to fetch {ticker}: {e}")

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


def _screened_universe_from_prices(prices_df: pd.DataFrame, top_n: int = 10) -> list[str]:
    """Select top symbols by short-horizon momentum for strategy handoff."""
    if prices_df.empty or len(prices_df) < 3:
        return []

    lookback = min(20, len(prices_df) - 1)
    momentum = (prices_df.iloc[-1] / prices_df.iloc[-lookback] - 1).sort_values(
        ascending=False
    )
    return list(momentum.head(top_n).index)


def _render_analysis_view(
    prices_df: pd.DataFrame,
    analysis_type: str,
    mode: str,
    snapshot_df: pd.DataFrame | None = None,
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
        from src.factors.ff_data import FamaFrenchDataLoader

        ff_loader = FamaFrenchDataLoader()
        ff_factors = ff_loader.load_ff5_factors(frequency="daily")
        return html.Div(
            className="chart-container",
            children=[create_factor_analysis_view(returns_df, ff_factors)],
        )

    if analysis_type == "screener":
        return _create_screener_table(prices_df, returns_df, snapshot_df=snapshot_df)

    return html.Div("Unknown analysis type")


def _historical_status(prices_df: pd.DataFrame, failed: list[str]) -> html.Div:
    return html.Div(
        [
            html.Span("✓ Historical data loaded", style={"color": "#4ade80"}),
            html.Span(
                f" | Source: Yahoo Finance | Tickers: {len(prices_df.columns)} | Rows: {len(prices_df)}",
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
            Output("live-results-area", "children"),
            Output("live-status", "children"),
            Output("live-data-store", "data"),
            Output("live-stream-config-store", "data"),
            Output("live-refresh-interval", "disabled"),
            Output("store-screened-universe", "data"),
        ],
        [
            Input("live-fetch-btn", "n_clicks"),
            Input("live-refresh-interval", "n_intervals"),
            Input("live-analysis-dropdown", "value"),
        ],
        [
            State("live-ticker-input", "value"),
            State("live-period-dropdown", "value"),
            State("live-data-mode-dropdown", "value"),
            State("live-stream-config-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def fetch_and_analyze(
        n_clicks,
        n_intervals,
        analysis_type,
        tickers_str,
        period,
        data_mode,
        stream_config,
    ):
        del n_clicks, n_intervals
        trigger = callback_context.triggered_id
        if not trigger:
            raise PreventUpdate

        hub = get_realtime_market_hub()

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
                )

            if data_mode == "historical":
                # Explicitly stop stream when user switches to historical mode
                hub.stop()

                prices_df, failed = _fetch_historical_prices(tickers, period)
                if prices_df.empty:
                    return (
                        html.Div(),
                        html.Div(
                            "Failed to fetch historical data for all tickers",
                            style={"color": "#f87171"},
                        ),
                        None,
                        None,
                        True,
                        no_update,
                    )

                result = _render_analysis_view(
                    prices_df,
                    analysis_type=analysis_type,
                    mode="historical",
                )
                status = _historical_status(prices_df, failed)
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
                return result, status, payload, None, True, screened

            # Realtime mode
            start_status = hub.start(tickers)
            prices_df = hub.get_price_frame(tickers)
            snapshot_df = hub.get_latest_snapshot(tickers)
            result = _render_analysis_view(
                prices_df,
                analysis_type=analysis_type,
                mode="realtime",
                snapshot_df=snapshot_df,
            )
            status = _realtime_status(start_status, snapshot_df)
            stream_payload = {
                "tickers": tickers,
                "mode": "realtime",
                "analysis": analysis_type,
            }
            data_payload = {"tickers": tickers, "mode": "realtime"}
            screened = no_update
            if analysis_type == "screener":
                screened = {
                    "tickers": _screened_universe_from_prices(prices_df),
                    "mode": "realtime",
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                }
            return result, status, data_payload, stream_payload, False, screened

        if trigger in ("live-refresh-interval", "live-analysis-dropdown"):
            if not stream_config or stream_config.get("mode") != "realtime":
                raise PreventUpdate

            tickers = stream_config.get("tickers", [])
            if not tickers:
                raise PreventUpdate

            prices_df = hub.get_price_frame(tickers)
            snapshot_df = hub.get_latest_snapshot(tickers)
            status = _realtime_status(hub.get_status(), snapshot_df)
            result = _render_analysis_view(
                prices_df,
                analysis_type=analysis_type,
                mode="realtime",
                snapshot_df=snapshot_df,
            )
            stream_config["analysis"] = analysis_type
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
            )

        raise PreventUpdate
