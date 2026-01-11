"""Regime Monitor page layout."""

from dash import html, dcc
import pandas as pd
import numpy as np

from src.ui.components import (
    create_metric_card,
    create_status_badge,
    create_regime_timeline,
    create_chart_container,
)


def create_regime_monitor_layout() -> html.Div:
    """Create regime monitor page layout."""
    # Generate sample regime data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=48, freq="ME")
    regime_values = np.random.choice([0, 1, 2, 3], 48, p=[0.5, 0.2, 0.15, 0.15])
    regime_names = ["Expansion", "Peak", "Contraction", "Trough"]
    regimes = [regime_names[r] for r in regime_values]

    return html.Div([
        # Metrics row
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col col-4",
                    children=html.Div(
                        className="metric-card",
                        children=[
                            html.Div("Current Regime", className="metric-label"),
                            html.Div(
                                create_status_badge("Expansion", "expansion"),
                                style={"marginTop": "0.5rem"},
                            ),
                        ],
                    ),
                ),
                html.Div(
                    className="col col-4",
                    children=create_metric_card("Confidence", "78%"),
                ),
                html.Div(
                    className="col col-4",
                    children=create_metric_card("Duration", "8 months"),
                ),
            ],
        ),

        html.Div(className="section-divider"),

        # Tabs
        dcc.Tabs(
            id="regime-tabs",
            value="indicators",
            className="custom-tabs",
            children=[
                dcc.Tab(label="Economic Indicators", value="indicators", className="tab"),
                dcc.Tab(label="Regime History", value="history", className="tab"),
            ],
        ),

        html.Div(id="regime-tab-content", style={"marginTop": "1.5rem"}),
    ])


def create_indicators_tab() -> html.Div:
    """Create economic indicators tab."""
    indicators = [
        {"name": "GDP Growth", "value": "+2.3%", "signal": "positive", "label": "Positive"},
        {"name": "Unemployment", "value": "3.8%", "signal": "positive", "label": "Low"},
        {"name": "10Y-2Y Spread", "value": "+0.45%", "signal": "positive", "label": "Normal"},
        {"name": "Fed Funds Rate", "value": "5.25%", "signal": "warning", "label": "High"},
        {"name": "PMI", "value": "52.1", "signal": "positive", "label": "Expanding"},
    ]

    return html.Div(
        className="chart-container",
        children=[
            html.Div("Key Economic Indicators", className="chart-title"),
            html.Div([
                html.Div(
                    className="indicator-row",
                    children=[
                        html.Div(ind["name"], className="indicator-name"),
                        html.Div([
                            html.Span(ind["value"], className="indicator-value"),
                            html.Span(
                                ind["label"],
                                className=f"indicator-signal {ind['signal']}",
                                style={"marginLeft": "1rem"},
                            ),
                        ], style={"display": "flex", "alignItems": "center", "gap": "1rem"}),
                    ],
                )
                for ind in indicators
            ]),
        ],
    )


def create_history_tab() -> html.Div:
    """Create regime history tab."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=48, freq="ME")
    regime_values = np.random.choice([0, 1, 2, 3], 48, p=[0.5, 0.2, 0.15, 0.15])
    regime_names = ["Expansion", "Peak", "Contraction", "Trough"]
    regimes = [regime_names[r] for r in regime_values]

    return create_chart_container(
        "Regime Timeline",
        create_regime_timeline(dates, regimes, height=250),
        subtitle="Historical market regime classification",
    )
