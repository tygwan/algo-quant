"""Backtest page layout."""

from dash import html, dcc
import pandas as pd
import numpy as np

from src.ui.components import (
    create_metric_card,
    create_line_chart,
    create_pie_chart,
    create_histogram,
    create_area_chart,
    create_chart_container,
    create_empty_state,
    get_table_style,
)


def create_backtest_layout() -> html.Div:
    """Create backtest page layout."""
    return html.Div([
        dcc.Tabs(
            id="backtest-tabs",
            value="configure",
            className="custom-tabs",
            children=[
                dcc.Tab(label="Configure", value="configure", className="tab"),
                dcc.Tab(label="Results", value="results", className="tab"),
                dcc.Tab(label="Analysis", value="analysis", className="tab"),
            ],
        ),

        html.Div(id="backtest-tab-content", style={"marginTop": "1.5rem"}),

        # Store for backtest results
        dcc.Store(id="backtest-results-store"),
    ])


def create_configure_tab() -> html.Div:
    """Create backtest configuration tab."""
    return html.Div(
        className="row",
        children=[
            # Strategy settings
            html.Div(
                className="col col-6",
                children=html.Div(
                    className="chart-container",
                    children=[
                        html.Div("Strategy Settings", className="chart-title"),

                        html.Div([
                            html.Label("Strategy Type", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                            dcc.Dropdown(
                                id="strategy-dropdown",
                                options=[
                                    {"label": "Equal Weight", "value": "equal_weight"},
                                    {"label": "Risk Parity", "value": "risk_parity"},
                                    {"label": "Momentum", "value": "momentum"},
                                    {"label": "Mean-Variance", "value": "mean_variance"},
                                ],
                                value="equal_weight",
                                clearable=False,
                                style={"marginTop": "0.5rem"},
                            ),
                        ], style={"marginBottom": "1rem"}),

                        html.Div([
                            html.Label("Universe", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                            dcc.Input(
                                id="universe-input",
                                type="text",
                                value="AAPL,MSFT,GOOGL,AMZN,META",
                                className="form-control",
                                style={"width": "100%", "marginTop": "0.5rem"},
                            ),
                        ], style={"marginBottom": "1rem"}),

                        html.Div([
                            html.Label("Rebalance Frequency", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                            dcc.Dropdown(
                                id="rebalance-dropdown",
                                options=[
                                    {"label": "Weekly", "value": "weekly"},
                                    {"label": "Monthly", "value": "monthly"},
                                    {"label": "Quarterly", "value": "quarterly"},
                                ],
                                value="monthly",
                                clearable=False,
                                style={"marginTop": "0.5rem"},
                            ),
                        ]),
                    ],
                ),
            ),

            # Backtest settings
            html.Div(
                className="col col-6",
                children=html.Div(
                    className="chart-container",
                    children=[
                        html.Div("Backtest Settings", className="chart-title"),

                        html.Div([
                            html.Label("Initial Capital ($)", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                            dcc.Input(
                                id="capital-input",
                                type="number",
                                value=100000,
                                step=10000,
                                className="form-control",
                                style={"width": "100%", "marginTop": "0.5rem"},
                            ),
                        ], style={"marginBottom": "1rem"}),

                        html.Div([
                            html.Label("Commission (%)", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                            dcc.Slider(
                                id="commission-slider",
                                min=0,
                                max=1,
                                step=0.01,
                                value=0.1,
                                marks={0: "0%", 0.5: "0.5%", 1: "1%"},
                            ),
                        ], style={"marginBottom": "1rem"}),

                        html.Div([
                            html.Label("Backtest Period (days)", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                            dcc.Slider(
                                id="periods-slider",
                                min=60,
                                max=756,
                                step=1,
                                value=252,
                                marks={60: "60", 252: "1Y", 504: "2Y", 756: "3Y"},
                            ),
                        ]),
                    ],
                ),
            ),
        ],
    ) , html.Div(className="section-divider"), html.Button(
            "Run Backtest",
            id="run-backtest-btn",
            className="btn-primary",
            style={"width": "100%", "marginTop": "1rem"},
        )


def create_results_tab(results: dict = None) -> html.Div:
    """Create backtest results tab."""
    if not results:
        return create_empty_state(
            icon="📈",
            message="Configure and run a backtest to see results",
        )

    metrics = results.get("metrics", {})

    return html.Div([
        # Metrics
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col col-3",
                    children=create_metric_card(
                        "Total Return",
                        f"{metrics.get('total_return', 0):.1%}",
                    ),
                ),
                html.Div(
                    className="col col-3",
                    children=create_metric_card(
                        "CAGR",
                        f"{metrics.get('cagr', 0):.1%}",
                    ),
                ),
                html.Div(
                    className="col col-3",
                    children=create_metric_card(
                        "Sharpe Ratio",
                        f"{metrics.get('sharpe_ratio', 0):.2f}",
                    ),
                ),
                html.Div(
                    className="col col-3",
                    children=create_metric_card(
                        "Max Drawdown",
                        f"{metrics.get('max_drawdown', 0):.1%}",
                    ),
                ),
            ],
        ),

        html.Div(className="section-divider"),

        # Charts
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col col-6",
                    id="portfolio-value-chart",
                ),
                html.Div(
                    className="col col-6",
                    id="allocation-chart",
                ),
            ],
        ),
    ])


def create_analysis_tab(results: dict = None) -> html.Div:
    """Create backtest analysis tab."""
    if not results:
        return create_empty_state(
            icon="📊",
            message="Run a backtest first to view analysis",
        )

    return html.Div([
        html.Div([
            html.Label("Analysis Type", style={"color": "var(--text-secondary)", "marginRight": "1rem"}),
            dcc.Dropdown(
                id="analysis-type-dropdown",
                options=[
                    {"label": "Returns Distribution", "value": "distribution"},
                    {"label": "Drawdown", "value": "drawdown"},
                    {"label": "Rolling Metrics", "value": "rolling"},
                ],
                value="distribution",
                style={"width": "250px"},
                clearable=False,
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "1.5rem"}),

        html.Div(id="analysis-chart-area"),
    ])
