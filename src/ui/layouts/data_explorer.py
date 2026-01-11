"""Data Explorer page layout."""

from dash import html, dcc
import pandas as pd

from src.ui.components import (
    create_line_chart,
    create_chart_container,
    create_empty_state,
)
from src.ui.services import DataService


def create_data_explorer_layout() -> html.Div:
    """Create data explorer page layout."""
    service = DataService(demo_mode=True)

    # Get sample data
    crypto_prices = service.get_prices(["BTC", "ETH", "BNB"], periods=365, start_date="2024-01-01")
    crypto_prices = crypto_prices * 300  # Scale

    macro_data = service.get_macro_data(48)

    return html.Div([
        # Tabs
        dcc.Tabs(
            id="data-explorer-tabs",
            value="stocks",
            className="custom-tabs",
            children=[
                dcc.Tab(label="Stocks", value="stocks", className="tab"),
                dcc.Tab(label="Macro Indicators", value="macro", className="tab"),
                dcc.Tab(label="Cryptocurrency", value="crypto", className="tab"),
            ],
        ),

        # Tab content
        html.Div(id="data-explorer-content", style={"marginTop": "1.5rem"}),

        # Store for generated data
        dcc.Store(id="stock-data-store"),
    ])


def create_stocks_tab() -> html.Div:
    """Create stocks tab content."""
    return html.Div(
        className="row",
        children=[
            # Config panel
            html.Div(
                className="col col-3",
                children=html.Div(
                    className="chart-container",
                    children=[
                        html.Div("Configuration", className="chart-title"),
                        html.Div([
                            html.Label(
                                "Symbols (comma-separated)",
                                style={"color": "var(--text-secondary)", "fontSize": "0.875rem"},
                            ),
                            dcc.Input(
                                id="stock-symbols-input",
                                type="text",
                                value="AAPL,MSFT,GOOGL,AMZN",
                                className="form-control",
                                style={"width": "100%", "marginTop": "0.5rem"},
                            ),
                        ], style={"marginBottom": "1rem"}),
                        html.Div([
                            html.Label(
                                "History (days)",
                                style={"color": "var(--text-secondary)", "fontSize": "0.875rem"},
                            ),
                            dcc.Slider(
                                id="stock-periods-slider",
                                min=60,
                                max=500,
                                value=252,
                                marks={60: "60", 252: "252", 500: "500"},
                            ),
                        ], style={"marginBottom": "1.5rem"}),
                        html.Button(
                            "Generate Data",
                            id="generate-stock-btn",
                            className="btn-primary",
                            style={"width": "100%"},
                        ),
                    ],
                ),
            ),

            # Chart area
            html.Div(
                className="col col-9",
                id="stock-chart-area",
                children=create_empty_state(
                    icon="📊",
                    message="Click 'Generate Data' to view stock prices",
                ),
            ),
        ],
    )


def create_macro_tab() -> html.Div:
    """Create macro indicators tab content."""
    service = DataService(demo_mode=True)
    macro_data = service.get_macro_data(48)

    return html.Div([
        html.Div([
            html.Label(
                "Select Indicators",
                style={"color": "var(--text-secondary)", "fontSize": "0.875rem", "marginRight": "1rem"},
            ),
            dcc.Dropdown(
                id="macro-indicator-dropdown",
                options=[{"label": col, "value": col} for col in macro_data.columns],
                value=["GDP Growth (%)"],
                multi=True,
                style={"minWidth": "300px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "1.5rem"}),

        html.Div(id="macro-chart-area"),
    ])


def create_crypto_tab() -> html.Div:
    """Create cryptocurrency tab content."""
    service = DataService(demo_mode=True)
    crypto_prices = service.get_prices(["BTC", "ETH", "BNB"], periods=365, start_date="2024-01-01")
    crypto_prices = crypto_prices * 300

    return create_chart_container(
        "Cryptocurrency Prices",
        create_line_chart(crypto_prices, height=450),
        subtitle="Demo data - Connect API for real prices",
    )
