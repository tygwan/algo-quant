"""Dashboard page callbacks."""

from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np

from src.ui.components import (
    create_line_chart,
    create_chart_container,
)
from src.ui.services import DataService


def register_dashboard_callbacks(app):
    """Register dashboard callbacks."""

    @app.callback(
        Output("dashboard-portfolio-chart", "children"),
        [Input("url", "pathname")],
        [State("store-demo-mode", "data")],
    )
    def update_portfolio_chart(pathname, demo_mode):
        """Update portfolio value chart on dashboard."""
        if pathname not in ["/", "/dashboard"]:
            raise PreventUpdate

        service = DataService(demo_mode=demo_mode if demo_mode is not None else True)
        prices = service.get_prices(["Portfolio"], periods=252)

        # Simulate portfolio value
        portfolio_value = 100000 * (1 + prices.cumsum())

        chart = create_line_chart(portfolio_value, height=300)
        return create_chart_container("Portfolio Value", chart)

    @app.callback(
        Output("dashboard-returns-chart", "children"),
        [Input("url", "pathname")],
        [State("store-demo-mode", "data")],
    )
    def update_returns_chart(pathname, demo_mode):
        """Update returns distribution chart."""
        if pathname not in ["/", "/dashboard"]:
            raise PreventUpdate

        service = DataService(demo_mode=demo_mode if demo_mode is not None else True)
        returns = service.get_returns(["Portfolio"], periods=252)

        from src.ui.components import create_histogram

        chart = create_histogram(returns["Portfolio"].values, height=300)
        return create_chart_container("Returns Distribution", chart)
