"""Sidebar navigation component."""

from dash import html, dcc


def create_navbar(current_page: str = "dashboard") -> html.Div:
    """Create sidebar navigation component.

    Args:
        current_page: Currently active page ID

    Returns:
        Dash HTML component for sidebar
    """
    nav_items = [
        {"id": "dashboard", "icon": "◉", "label": "Dashboard"},
        {"id": "data-explorer", "icon": "◈", "label": "Data Explorer"},
        {"id": "factor-analysis", "icon": "◇", "label": "Factor Analysis"},
        {"id": "regime-monitor", "icon": "◆", "label": "Regime Monitor"},
        {"id": "backtest", "icon": "▣", "label": "Backtest"},
        {"id": "portfolio", "icon": "◧", "label": "Portfolio"},
    ]

    return html.Div(
        className="sidebar",
        children=[
            # Brand
            html.Div(
                className="sidebar-brand",
                children=[
                    html.Div("◉", className="sidebar-brand-icon"),
                    html.Div([
                        html.Div("Algo-Quant", className="sidebar-brand-text"),
                        html.Div("Quantitative Investing", className="sidebar-brand-subtitle"),
                    ]),
                ],
            ),

            # Navigation section
            html.Div("Navigation", className="nav-section-title"),

            # Nav items
            html.Div([
                html.Div(
                    id={"type": "nav-item", "index": item["id"]},
                    className=f"nav-item {'active' if item['id'] == current_page else ''}",
                    children=[
                        html.Span(item["icon"], className="nav-icon"),
                        html.Span(item["label"]),
                    ],
                    n_clicks=0,
                )
                for item in nav_items
            ]),

            # Divider
            html.Div(className="section-divider", style={"marginTop": "auto"}),

            # Settings section
            html.Div("Settings", className="nav-section-title"),

            # Demo mode toggle
            html.Div([
                dcc.Checklist(
                    id="demo-mode-toggle",
                    options=[{"label": " Demo Mode", "value": "demo"}],
                    value=["demo"],
                    style={"color": "var(--text-secondary)"},
                ),
                html.Div(
                    id="demo-mode-badge",
                    className="demo-badge",
                    children=[
                        html.Div("✓ Demo Mode Active", className="demo-badge-title"),
                        html.Div("Using simulated market data", className="demo-badge-text"),
                    ],
                ),
            ]),

            # Version
            html.Div(
                style={
                    "marginTop": "2rem",
                    "paddingTop": "1rem",
                    "borderTop": "1px solid var(--border)",
                    "color": "var(--text-muted)",
                    "fontSize": "0.75rem",
                },
                children="Version 0.2.0",
            ),
        ],
    )


def create_breadcrumb(items: list[dict]) -> html.Div:
    """Create breadcrumb navigation.

    Args:
        items: List of {"label": str, "href": str} dicts

    Returns:
        Breadcrumb component
    """
    children = []
    for i, item in enumerate(items):
        if i > 0:
            children.append(html.Span(" / ", style={"color": "var(--text-muted)", "margin": "0 0.5rem"}))

        if item.get("href"):
            children.append(
                dcc.Link(
                    item["label"],
                    href=item["href"],
                    style={"color": "var(--text-secondary)", "textDecoration": "none"},
                )
            )
        else:
            children.append(
                html.Span(item["label"], style={"color": "var(--text)"})
            )

    return html.Div(children, style={"marginBottom": "1rem", "fontSize": "0.875rem"})
