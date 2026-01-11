"""Dash application entry point."""

from dash import Dash, html, dcc, page_container
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from src.ui.components import create_navbar
from src.ui.layouts import (
    create_dashboard_layout,
    create_data_explorer_layout,
    create_factor_analysis_layout,
    create_regime_monitor_layout,
    create_backtest_layout,
    create_portfolio_layout,
)
from src.ui.state import PAGE_CONFIG

# Initialize app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="algo-quant Dashboard",
    update_title="Loading...",
)

server = app.server

# Main layout
app.layout = html.Div(
    className="app-container",
    children=[
        # URL routing
        dcc.Location(id="url", refresh=False),

        # Sidebar
        html.Div(id="sidebar-container"),

        # Main content
        html.Div(
            className="main-content",
            children=[
                html.Div(id="page-content"),
            ],
        ),

        # Global stores
        dcc.Store(id="store-demo-mode", data=True),
        dcc.Store(id="store-backtest-result"),
        dcc.Store(id="store-portfolio"),
        dcc.Store(id="store-theme", data="dark"),
    ],
)


# Page routing
@app.callback(
    [Output("sidebar-container", "children"), Output("page-content", "children")],
    [Input("url", "pathname")],
)
def display_page(pathname):
    """Route to appropriate page based on URL."""
    if pathname is None or pathname == "/":
        pathname = "/dashboard"

    # Remove leading slash for lookup
    page_key = pathname.lstrip("/") or "dashboard"

    # Get page config
    page_info = PAGE_CONFIG.get(page_key, PAGE_CONFIG["dashboard"])

    # Create navbar with current page
    navbar = create_navbar(page_key)

    # Route to page layout
    if page_key == "dashboard":
        content = create_dashboard_layout()
    elif page_key == "data-explorer":
        content = create_data_explorer_layout()
    elif page_key == "factor-analysis":
        content = create_factor_analysis_layout()
    elif page_key == "regime-monitor":
        content = create_regime_monitor_layout()
    elif page_key == "backtest":
        content = create_backtest_layout()
    elif page_key == "portfolio":
        content = create_portfolio_layout()
    else:
        content = create_dashboard_layout()

    return navbar, content


# Import callbacks to register them
from src.ui.callbacks import register_callbacks

register_callbacks(app)


def run_dashboard(debug: bool = True, port: int = 8050):
    """Run the Dash dashboard."""
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_dashboard()
