"""Portfolio page layout."""

from dash import html, dcc, dash_table
import pandas as pd

from src.ui.components import (
    create_pie_chart,
    create_chart_container,
    create_empty_state,
    get_table_style,
)


def create_portfolio_layout() -> html.Div:
    """Create portfolio management page layout."""
    return html.Div([
        dcc.Tabs(
            id="portfolio-tabs",
            value="holdings",
            className="custom-tabs",
            children=[
                dcc.Tab(label="Current Portfolio", value="holdings", className="tab"),
                dcc.Tab(label="Optimization", value="optimization", className="tab"),
            ],
        ),

        html.Div(id="portfolio-tab-content", style={"marginTop": "1.5rem"}),

        # Store for portfolio data
        dcc.Store(id="portfolio-store"),
    ])


def create_holdings_tab(portfolio: dict = None) -> html.Div:
    """Create holdings tab."""
    if not portfolio:
        return html.Div([
            create_empty_state(
                icon="💼",
                message="Run a backtest or add positions manually",
            ),

            # Manual entry form
            html.Div(
                className="chart-container",
                style={"marginTop": "2rem", "maxWidth": "500px"},
                children=[
                    html.Div("Add Position", className="chart-title"),

                    html.Div(
                        className="row",
                        children=[
                            html.Div(
                                className="col col-6",
                                children=[
                                    html.Label("Symbol", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                    dcc.Input(
                                        id="add-symbol-input",
                                        type="text",
                                        placeholder="AAPL",
                                        className="form-control",
                                        style={"width": "100%", "marginTop": "0.5rem"},
                                    ),
                                ],
                            ),
                            html.Div(
                                className="col col-6",
                                children=[
                                    html.Label("Weight (%)", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                    dcc.Input(
                                        id="add-weight-input",
                                        type="number",
                                        min=0,
                                        max=100,
                                        value=25,
                                        className="form-control",
                                        style={"width": "100%", "marginTop": "0.5rem"},
                                    ),
                                ],
                            ),
                        ],
                    ),

                    html.Button(
                        "Add Position",
                        id="add-position-btn",
                        className="btn-primary",
                        style={"marginTop": "1rem"},
                    ),
                ],
            ),
        ])

    # Display portfolio
    holdings_df = pd.DataFrame([
        {"Symbol": symbol, "Weight": f"{weight:.1%}", "Value": f"${weight * 100000:,.0f}"}
        for symbol, weight in portfolio.items()
    ])

    table_style = get_table_style()

    return html.Div(
        className="row",
        children=[
            html.Div(
                className="col col-8",
                children=html.Div(
                    className="chart-container",
                    children=[
                        html.Div("Holdings", className="chart-title"),
                        dash_table.DataTable(
                            id="holdings-table",
                            columns=[{"name": c, "id": c} for c in holdings_df.columns],
                            data=holdings_df.to_dict("records"),
                            **table_style,
                        ),
                    ],
                ),
            ),
            html.Div(
                className="col col-4",
                children=create_chart_container(
                    "Allocation",
                    create_pie_chart(
                        list(portfolio.keys()),
                        list(portfolio.values()),
                        height=300,
                    ),
                ),
            ),
        ],
    )


def create_optimization_tab() -> html.Div:
    """Create optimization tab."""
    return html.Div(
        className="row",
        children=[
            # Config
            html.Div(
                className="col col-4",
                children=html.Div(
                    className="chart-container",
                    children=[
                        html.Div("Optimization Settings", className="chart-title"),

                        html.Div([
                            html.Label("Method", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                            dcc.Dropdown(
                                id="opt-method-dropdown",
                                options=[
                                    {"label": "Equal Weight", "value": "equal_weight"},
                                    {"label": "Minimum Variance", "value": "min_variance"},
                                    {"label": "Maximum Sharpe", "value": "max_sharpe"},
                                    {"label": "Risk Parity", "value": "risk_parity"},
                                ],
                                value="equal_weight",
                                clearable=False,
                                style={"marginTop": "0.5rem"},
                            ),
                        ], style={"marginBottom": "1rem"}),

                        html.Div([
                            html.Label("Assets", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                            dcc.Input(
                                id="opt-assets-input",
                                type="text",
                                value="AAPL,MSFT,GOOGL,AMZN",
                                className="form-control",
                                style={"width": "100%", "marginTop": "0.5rem"},
                            ),
                        ], style={"marginBottom": "1.5rem"}),

                        html.Button(
                            "Optimize",
                            id="optimize-btn",
                            className="btn-primary",
                            style={"width": "100%"},
                        ),
                    ],
                ),
            ),

            # Results
            html.Div(
                className="col col-8",
                id="optimization-results",
                children=create_empty_state(
                    icon="⚖️",
                    message="Configure and run optimization",
                ),
            ),
        ],
    )
