"""Dashboard page layout."""

from dash import html, dcc
import pandas as pd
import numpy as np

from src.ui.components import (
    create_metric_row,
    create_status_badge,
    create_line_chart,
    create_pie_chart,
    create_chart_container,
)
from src.ui.services import DataService


def create_dashboard_layout() -> html.Div:
    """Create the main dashboard layout."""
    # Get demo data
    service = DataService(demo_mode=True)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    prices = service.get_prices(symbols, periods=252)
    returns = prices.pct_change().dropna()

    portfolio_values = 100000 * (1 + returns.mean(axis=1)).cumprod()
    total_return = (portfolio_values.iloc[-1] / 100000 - 1) * 100
    vol = returns.mean(axis=1).std() * np.sqrt(252) * 100
    sharpe = (returns.mean(axis=1).mean() * 252) / (returns.mean(axis=1).std() * np.sqrt(252))

    # Create metrics
    metrics = [
        {
            "label": "Portfolio Value",
            "value": f"${portfolio_values.iloc[-1]:,.0f}",
            "delta": f"{total_return:+.1f}%",
            "delta_type": "positive" if total_return > 0 else "negative",
        },
        {
            "label": "Annual Volatility",
            "value": f"{vol:.1f}%",
        },
        {
            "label": "Sharpe Ratio",
            "value": f"{sharpe:.2f}",
        },
    ]

    # Regime metric (special)
    regime_metric = html.Div(
        className="metric-card",
        children=[
            html.Div("Current Regime", className="metric-label"),
            html.Div(
                create_status_badge("Expansion", "expansion"),
                style={"marginTop": "0.5rem"},
            ),
        ],
    )

    # Allocation weights
    weights = {"AAPL": 0.30, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.20}

    return html.Div([
        # Metrics row
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col col-3",
                    children=html.Div(
                        className="metric-card",
                        children=[
                            html.Div(m["label"], className="metric-label"),
                            html.Div(m["value"], className="metric-value"),
                            html.Div(
                                f"{'↑' if m.get('delta_type') == 'positive' else '↓'} {m['delta']}",
                                className=f"metric-delta {m.get('delta_type', '')}",
                            ) if m.get("delta") else None,
                        ],
                    ),
                )
                for m in metrics
            ] + [html.Div(className="col col-3", children=regime_metric)],
        ),

        # Divider
        html.Div(className="section-divider"),

        # Charts row
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col col-6",
                    children=create_chart_container(
                        "Portfolio Performance",
                        create_line_chart(
                            portfolio_values.to_frame("Portfolio"),
                            height=350,
                        ),
                    ),
                ),
                html.Div(
                    className="col col-6",
                    children=create_chart_container(
                        "Asset Allocation",
                        create_pie_chart(
                            list(weights.keys()),
                            list(weights.values()),
                            height=350,
                        ),
                    ),
                ),
            ],
        ),

        # Full width price chart
        create_chart_container(
            "Asset Prices (Indexed)",
            create_line_chart(prices, height=400),
            subtitle="Historical price data for portfolio assets",
        ),
    ])
