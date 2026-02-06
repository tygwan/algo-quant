"""Dash application entry point."""

from dash import Dash, html, dcc
from dash.dependencies import Input, Output

from src.ui.components import create_navbar
from src.ui.layouts import (
    create_dashboard_layout,
    create_data_explorer_layout,
    create_factor_analysis_layout,
    create_regime_monitor_layout,
    create_backtest_layout,
    create_portfolio_layout,
)
from src.ui.layouts.live_analyzer import create_live_analyzer_layout, register_live_analyzer_callbacks
from src.ui.state import PAGE_CONFIG

# Initialize app (no Bootstrap theme to avoid CSS conflicts)
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="algo-quant Dashboard",
    update_title=None,
)

server = app.server


def create_page_header(page_key: str) -> html.Div:
    """Create consistent page header shell."""
    page_info = PAGE_CONFIG.get(page_key, PAGE_CONFIG["dashboard"])

    return html.Div(
        className="page-header",
        children=[
            html.Div("ALGO QUANT WORKSPACE", className="page-kicker"),
            html.H1(page_info["title"], className="page-title"),
            html.P(page_info["subtitle"], className="page-subtitle"),
            html.Div(
                className="page-meta",
                children=[
                    html.Span(page_info.get("icon", "◉"), className="page-chip"),
                    html.Span(f"/{page_key}", className="page-chip"),
                    html.Span("US Stocks + Crypto", className="page-chip"),
                ],
            ),
        ],
    )


# Main layout
app.layout = html.Div(
    className="app-shell",
    children=[
        # URL routing
        dcc.Location(id="url", refresh=False),

        # Background ambience
        html.Div(className="ambient ambient-a"),
        html.Div(className="ambient ambient-b"),
        html.Div(className="ambient-grid"),

        # Sidebar
        html.Div(id="sidebar-container"),

        # Main content
        html.Div(
            className="main-content",
            children=[
                html.Div(id="page-header-container"),
                html.Div(id="page-content"),
            ],
        ),

        # Global stores
        dcc.Store(id="store-demo-mode", data=True),
        dcc.Store(id="store-backtest-result"),
        dcc.Store(id="store-portfolio"),
        dcc.Store(id="store-screened-universe"),
        dcc.Store(id="store-theme", data="dark"),
    ],
)


# Page routing
@app.callback(
    [
        Output("sidebar-container", "children"),
        Output("page-header-container", "children"),
        Output("page-content", "children"),
    ],
    [Input("url", "pathname")],
)
def display_page(pathname):
    """Route to appropriate page based on URL."""
    if pathname is None or pathname == "/":
        pathname = "/dashboard"

    # Remove leading slash for lookup
    page_key = pathname.lstrip("/") or "dashboard"

    # Create navbar with current page
    navbar = create_navbar(page_key)
    header = create_page_header(page_key)

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
    elif page_key == "live-analyzer":
        content = create_live_analyzer_layout()
    else:
        content = create_dashboard_layout()

    return navbar, header, content


# Import callbacks to register them
from src.ui.callbacks import register_callbacks

register_callbacks(app)
register_live_analyzer_callbacks(app)


def run_dashboard(debug: bool = True, port: int = 8050, host: str = "127.0.0.1"):
    """Run the Dash dashboard."""
    app.run(debug=debug, port=port, host=host)


if __name__ == "__main__":
    run_dashboard()
