"""Factor Analysis page layout."""

from dash import html, dcc
import pandas as pd
import numpy as np

from src.ui.components import (
    create_line_chart,
    create_bar_chart,
    create_chart_container,
    get_table_style,
)
from src.ui.services import DataService


def create_factor_analysis_layout() -> html.Div:
    """Create factor analysis page layout."""
    return html.Div([
        dcc.Tabs(
            id="factor-tabs",
            value="returns",
            className="custom-tabs",
            children=[
                dcc.Tab(label="Factor Returns", value="returns", className="tab"),
                dcc.Tab(label="Factor Exposure", value="exposure", className="tab"),
            ],
        ),

        html.Div(id="factor-tab-content", style={"marginTop": "1.5rem"}),
    ])


def create_factor_returns_tab() -> html.Div:
    """Create factor returns tab."""
    service = DataService(demo_mode=True)
    factor_data = service.get_factor_data(252)

    return html.Div([
        # Model selector
        html.Div([
            html.Label(
                "Model",
                style={"color": "var(--text-secondary)", "fontSize": "0.875rem", "marginRight": "1rem"},
            ),
            dcc.Dropdown(
                id="factor-model-dropdown",
                options=[
                    {"label": "FF3 (3-Factor)", "value": "ff3"},
                    {"label": "FF5 (5-Factor)", "value": "ff5"},
                ],
                value="ff3",
                style={"width": "200px"},
                clearable=False,
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "1.5rem"}),

        # Chart
        html.Div(id="factor-returns-chart"),

        # Stats table
        html.Div(
            id="factor-stats-table",
            style={"marginTop": "1.5rem"},
        ),
    ])


def create_factor_exposure_tab() -> html.Div:
    """Create factor exposure tab."""
    exposures = {
        "Market": 1.05,
        "Size (SMB)": -0.15,
        "Value (HML)": 0.25,
        "Profitability": 0.10,
        "Investment": -0.05,
    }

    return html.Div([
        create_chart_container(
            "Portfolio Factor Exposure",
            create_bar_chart(
                list(exposures.keys()),
                list(exposures.values()),
                height=400,
            ),
        ),

        # Interpretation guide
        html.Div(
            className="chart-container",
            style={"marginTop": "1rem"},
            children=[
                html.Div("Interpretation Guide", className="chart-title"),
                html.Div([
                    html.Div([
                        html.Strong("Market β > 1:", style={"color": "var(--text-secondary)"}),
                        " Portfolio is more volatile than market",
                    ], style={"marginBottom": "0.5rem", "color": "var(--text-muted)"}),
                    html.Div([
                        html.Strong("Negative SMB:", style={"color": "var(--text-secondary)"}),
                        " Tilted toward large-cap stocks",
                    ], style={"marginBottom": "0.5rem", "color": "var(--text-muted)"}),
                    html.Div([
                        html.Strong("Positive HML:", style={"color": "var(--text-secondary)"}),
                        " Tilted toward value stocks",
                    ], style={"color": "var(--text-muted)"}),
                ]),
            ],
        ),
    ])
