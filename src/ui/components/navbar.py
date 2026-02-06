"""Sidebar navigation component."""

from dash import html, dcc


def create_navbar(current_page: str = "dashboard") -> html.Div:
    """Create sidebar navigation component.

    Args:
        current_page: Currently active page ID

    Returns:
        Dash HTML component for sidebar
    """
    # Navigation items grouped by category
    nav_groups = [
        {
            "id": "overview",
            "label": "Overview",
            "icon": "⬡",
            "items": [
                {"id": "dashboard", "icon": "◉", "label": "Dashboard"},
            ],
        },
        {
            "id": "data",
            "label": "Data",
            "icon": "⬢",
            "items": [
                {"id": "data-explorer", "icon": "◈", "label": "Data Explorer"},
            ],
        },
        {
            "id": "analysis",
            "label": "Analysis",
            "icon": "◇",
            "items": [
                {"id": "live-analyzer", "icon": "⚡", "label": "Live Analyzer"},
                {"id": "factor-analysis", "icon": "◇", "label": "Factor Analysis"},
                {"id": "regime-monitor", "icon": "◆", "label": "Regime Monitor"},
            ],
        },
        {
            "id": "trading",
            "label": "Trading",
            "icon": "▣",
            "items": [
                {"id": "backtest", "icon": "▣", "label": "Backtest"},
                {"id": "portfolio", "icon": "◧", "label": "Portfolio"},
            ],
        },
    ]

    def create_nav_group(group: dict) -> html.Div:
        """Create a navigation group with header and items."""
        return html.Div(
            className="nav-group",
            children=[
                # Group header with icon
                html.Div(
                    className="nav-group-header",
                    children=[
                        html.Span(group["icon"], className="nav-group-icon"),
                        html.Span(group["label"], className="nav-group-label"),
                    ],
                ),
                # Group items
                html.Div(
                    className="nav-group-items",
                    children=[
                        dcc.Link(
                            className=f"nav-item {'active' if item['id'] == current_page else ''}",
                            href=f"/{item['id']}",
                            title=f"Navigate to {item['label']}",
                            children=[
                                html.Span(item["icon"], className="nav-icon"),
                                html.Span(item["label"]),
                            ],
                        )
                        for item in group["items"]
                    ],
                ),
            ],
        )

    return html.Div(
        children=[
            # Mobile sidebar overlay
            html.Div(
                id="sidebar-overlay",
                className="sidebar-overlay",
            ),

            # Sidebar container
            html.Div(
                id="sidebar",
                className="sidebar",
                role="navigation",
                children=[
                    # Brand with hamburger toggle for mobile
                    html.Div(
                        className="sidebar-brand",
                        children=[
                            # Hamburger toggle button (shown on mobile via CSS)
                            html.Button(
                                id="sidebar-toggle",
                                className="sidebar-toggle",
                                title="Toggle navigation menu",
                                children=[
                                    html.Span(className="hamburger-line"),
                                    html.Span(className="hamburger-line"),
                                    html.Span(className="hamburger-line"),
                                ],
                            ),
                            html.Div("◉", className="sidebar-brand-icon"),
                            html.Div([
                                html.Div("Algo-Quant", className="sidebar-brand-text"),
                                html.Div("Quantitative Investing", className="sidebar-brand-subtitle"),
                            ]),
                        ],
                    ),

                    # Navigation groups
                    html.Nav(
                        className="nav-container",
                        children=[create_nav_group(group) for group in nav_groups],
                    ),

                    # Divider
                    html.Div(className="section-divider", style={"marginTop": "auto"}),

                    # Settings section
                    html.Div(
                        className="nav-group-header",
                        children=[
                            html.Span("⚙", className="nav-group-icon"),
                            html.Span("Settings", className="nav-group-label"),
                        ],
                    ),

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
                                html.Div("Demo Mode Active", className="demo-badge-title"),
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
