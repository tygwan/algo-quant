"""Live Stock Analyzer - Interactive ticker input and analysis."""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import date, timedelta
import logging

logger = logging.getLogger(__name__)


def create_live_analyzer_layout() -> html.Div:
    """Create live analyzer page layout."""
    return html.Div([
        # Header
        html.Div([
            html.H2("Live Stock Analyzer", style={"color": "var(--text-primary)", "marginBottom": "0.5rem"}),
            html.P("Enter tickers to fetch real-time data and run analysis",
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
                            className="col col-4",
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
                            className="col col-3",
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


# Register callbacks
def register_live_analyzer_callbacks(app):
    """Register callbacks for live analyzer."""

    @app.callback(
        [Output("live-results-area", "children"),
         Output("live-status", "children"),
         Output("live-data-store", "data")],
        [Input("live-fetch-btn", "n_clicks")],
        [State("live-ticker-input", "value"),
         State("live-period-dropdown", "value"),
         State("live-analysis-dropdown", "value")],
        prevent_initial_call=True,
    )
    def fetch_and_analyze(n_clicks, tickers_str, period, analysis_type):
        if not tickers_str:
            return html.Div(), html.Div("Please enter tickers", style={"color": "#f87171"}), None

        # Parse tickers
        tickers = [t.strip().upper() for t in tickers_str.replace(",", " ").split() if t.strip()]

        if not tickers:
            return html.Div(), html.Div("No valid tickers", style={"color": "#f87171"}), None

        try:
            # Import yfinance
            from src.data.yfinance_client import YFinanceClient
            yf_client = YFinanceClient()

            # Fetch data
            prices_dict = {}
            failed = []

            for ticker in tickers[:10]:  # Limit to 10
                try:
                    df = yf_client.get_historical_prices(ticker, period=period)
                    if not df.empty and 'close' in df.columns:
                        prices_dict[ticker] = df['close']
                except Exception as e:
                    failed.append(ticker)
                    logger.warning(f"Failed to fetch {ticker}: {e}")

            if not prices_dict:
                return html.Div(), html.Div("Failed to fetch data for all tickers", style={"color": "#f87171"}), None

            # Create DataFrame
            prices_df = pd.DataFrame(prices_dict)
            returns_df = prices_df.pct_change().dropna()

            # Status message
            status_msg = html.Div([
                html.Span(f"✓ Loaded {len(prices_dict)} tickers", style={"color": "#4ade80"}),
                html.Span(f" | {len(prices_df)} days of data", style={"color": "var(--text-secondary)"}),
                html.Span(f" | Failed: {', '.join(failed)}", style={"color": "#f87171"}) if failed else None,
            ])

            # Generate analysis based on type
            if analysis_type == "price":
                chart = create_price_chart(prices_df)
                result = html.Div(
                    className="chart-container",
                    children=[
                        html.Div("Normalized Price Chart", className="chart-title"),
                        dcc.Graph(figure=chart, config={"displayModeBar": False}),
                    ],
                )

            elif analysis_type == "returns":
                chart = create_returns_chart(returns_df)
                result = html.Div(
                    className="chart-container",
                    children=[
                        html.Div("Cumulative Returns", className="chart-title"),
                        dcc.Graph(figure=chart, config={"displayModeBar": False}),
                    ],
                )

            elif analysis_type == "correlation":
                chart = create_correlation_heatmap(returns_df)
                result = html.Div(
                    className="chart-container",
                    children=[
                        html.Div("Correlation Matrix", className="chart-title"),
                        dcc.Graph(figure=chart, config={"displayModeBar": False}),
                    ],
                )

            elif analysis_type == "risk":
                result = html.Div(
                    className="chart-container",
                    children=[
                        html.Div("Risk Metrics", className="chart-title"),
                        create_risk_metrics_table(returns_df),
                    ],
                )

            elif analysis_type == "factor":
                # Load FF factors
                from src.factors.ff_data import FamaFrenchDataLoader
                ff_loader = FamaFrenchDataLoader()
                ff_factors = ff_loader.load_ff5_factors(frequency="daily")

                result = html.Div(
                    className="chart-container",
                    children=[
                        create_factor_analysis_view(returns_df, ff_factors),
                    ],
                )

            else:
                result = html.Div("Unknown analysis type")

            return result, status_msg, {"tickers": list(prices_dict.keys())}

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return html.Div(), html.Div(f"Error: {str(e)}", style={"color": "#f87171"}), None
