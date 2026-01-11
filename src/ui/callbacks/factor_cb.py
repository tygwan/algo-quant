"""Factor Analysis page callbacks."""

from dash import Input, Output, html, dash_table
from dash.exceptions import PreventUpdate
import pandas as pd

from src.ui.components import (
    create_line_chart,
    create_chart_container,
    get_table_style,
)
from src.ui.services import DataService
from src.ui.layouts.factor_analysis import (
    create_factor_returns_tab,
    create_factor_exposure_tab,
)


def register_factor_callbacks(app):
    """Register factor analysis callbacks."""

    @app.callback(
        Output("factor-tab-content", "children"),
        [Input("factor-tabs", "value")],
    )
    def render_factor_tab(tab):
        """Render factor analysis tab content."""
        if tab == "returns":
            return create_factor_returns_tab()
        elif tab == "exposure":
            return create_factor_exposure_tab()
        return create_factor_returns_tab()

    @app.callback(
        [
            Output("factor-returns-chart", "children"),
            Output("factor-stats-table", "children"),
        ],
        [Input("factor-model-dropdown", "value")],
    )
    def update_factor_returns(model):
        """Update factor returns chart and stats."""
        service = DataService(demo_mode=True)

        if model == "ff3":
            factor_data = service.get_factor_data(252)
            factors = ["Mkt-RF", "SMB", "HML"]
        else:
            factor_data = service.get_factor_data(252)
            factors = list(factor_data.columns)

        # Filter to selected factors
        chart_data = factor_data[factors] if model == "ff3" else factor_data

        # Create chart
        cumulative = (1 + chart_data / 100).cumprod()
        chart = create_line_chart(cumulative, height=350)
        chart_container = create_chart_container(
            f"Factor Cumulative Returns ({model.upper()})",
            chart,
        )

        # Calculate stats
        stats = pd.DataFrame({
            "Factor": factors,
            "Mean (%)": [f"{chart_data[f].mean():.2f}" for f in factors],
            "Std (%)": [f"{chart_data[f].std():.2f}" for f in factors],
            "Sharpe": [f"{chart_data[f].mean() / chart_data[f].std() * (252**0.5):.2f}" for f in factors],
        })

        table_style = get_table_style()
        stats_table = html.Div(
            className="chart-container",
            children=[
                html.Div("Factor Statistics", className="chart-title"),
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in stats.columns],
                    data=stats.to_dict("records"),
                    **table_style,
                ),
            ],
        )

        return chart_container, stats_table
